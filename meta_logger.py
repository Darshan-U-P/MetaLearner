import torch
import time
import json
import numpy as np
import multiprocessing as mp
import queue
import os


def _logger_worker(log_queue, file_path):
    with open(file_path, "a") as f:
        while True:
            try:
                item = log_queue.get(timeout=5)
            except queue.Empty:
                continue

            if item == "STOP":
                break

            f.write(json.dumps(item) + "\n")
            f.flush()


class MetaLogger:
    def __init__(self, model, optimizer, file_path="logs/meta_logs.jsonl"):
        self.model = model
        self.optimizer = optimizer
        self.prev_loss = None

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        self.queue = mp.Queue(maxsize=2000)
        self.process = mp.Process(
            target=_logger_worker,
            args=(self.queue, file_path),
            daemon=True
        )
        self.process.start()

    def _global_grad_norm(self):
        total = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total += p.grad.detach().norm(2).item() ** 2
        return total ** 0.5

    def _global_weight_norm(self):
        total = 0.0
        for p in self.model.parameters():
            total += p.detach().norm(2).item() ** 2
        return total ** 0.5

    def _update_ratio(self, grad_norm, weight_norm, lr):
        if weight_norm == 0:
            return 0.0
        return (lr * grad_norm) / weight_norm

    def _layer_grad_stats(self):
        norms = []
        for p in self.model.parameters():
            if p.grad is not None:
                norms.append(p.grad.detach().norm(2).item())
        if not norms:
            return 0.0, 0.0
        return float(np.mean(norms)), float(np.std(norms))

    @staticmethod
    def attention_entropy(attn_weights):
        probs = torch.clamp(attn_weights, min=1e-9)
        entropy = -(probs * torch.log(probs)).sum(dim=-1)
        return entropy.mean().item(), entropy.std().item()

    @staticmethod
    def expert_load_entropy(expert_indices, num_experts):
        counts = torch.bincount(
            expert_indices.flatten(),
            minlength=num_experts
        ).float()

        if counts.sum() == 0:
            return 0.0

        probs = counts / counts.sum()
        probs = torch.clamp(probs, min=1e-9)
        entropy = -(probs * torch.log(probs)).sum()
        return entropy.item()

    def log_step(
        self,
        loss,
        attention_weights=None,
        expert_indices=None,
        num_experts=None,
        activation_sparsity=0.0,
        fp16_overflow=False
    ):

        loss_val = float(loss.item())
        delta_loss = (
            loss_val - self.prev_loss
            if self.prev_loss is not None else 0.0
        )
        self.prev_loss = loss_val

        grad_norm = self._global_grad_norm()
        weight_norm = self._global_weight_norm()
        lr = self.optimizer.param_groups[0]["lr"]
        update_ratio = self._update_ratio(grad_norm, weight_norm, lr)
        layer_grad_mean, layer_grad_std = self._layer_grad_stats()

        attn_mean, attn_std = 0.0, 0.0
        if attention_weights is not None:
            attn_mean, attn_std = self.attention_entropy(attention_weights)

        expert_entropy = 0.0
        if expert_indices is not None and num_experts is not None:
            expert_entropy = self.expert_load_entropy(
                expert_indices,
                num_experts
            )

        log_dict = {
            "timestamp": time.time(),
            "loss": loss_val,
            "delta_loss": delta_loss,
            "lr": lr,
            "grad_norm": grad_norm,
            "update_ratio": update_ratio,
            "layer_grad_mean": layer_grad_mean,
            "layer_grad_std": layer_grad_std,
            "attn_entropy_mean": attn_mean,
            "attn_entropy_std": attn_std,
            "expert_load_entropy": expert_entropy,
            "activation_sparsity": float(activation_sparsity),
            "fp16_overflow": int(fp16_overflow),
        }

        try:
            self.queue.put_nowait(log_dict)
        except queue.Full:
            pass

    def close(self):
        self.queue.put("STOP")
        self.process.join()