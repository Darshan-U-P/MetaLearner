
# 🧠 MetaLearner – Learning-to-Learn Optimization

A meta-optimization research project that trains models to predict and improve training dynamics on MNIST.

## 🎯 Quick Start

**Goal:** Train multiple MNIST models with varied architectures, log per-step dynamics, then train a meta-learner to predict optimization behavior and reduce convergence epochs.

**Success metric:** Reduce training from ~10 epochs to 4–6 epochs while maintaining accuracy.

---

## 🔬 How It Works

1. **Phase 1: Base Model Training**
    - Train 30–100 MNIST models with varying depth, width, activations, and optimizers
    - Log gradient norms, weight updates, loss deltas, and accuracy per step

2. **Phase 2: Dataset Construction**
    - Convert logs into structured rows: `[loss, grad_norm, weight_norm, update_ratio, lr, depth, width, activation, next_loss]`
    - Target: predict next loss, convergence speed, or ideal LR adjustment

3. **Phase 3: MetaLearner Model**
    - Train an MLP, Transformer, or PB-ANN model on optimization sequences
    - Learn patterns: gradient explosions, plateaus, optimal LR scaling, early stopping signals

---

## 📊 Critical Logging Requirements

Per training step, capture:
- **Loss:** current loss, ΔLoss
- **Accuracy:** train & validation
- **Gradients:** norm ‖g‖₂, per-layer norms
- **Weights:** norm ‖W‖₂, update norm ‖ΔW‖₂, update-to-weight ratio
- **Metadata:** LR, optimizer, batch/epoch index, architecture encoding

---

## 💡 Why It Matters

You're building a **dataset of optimization physics**. MetaLearner learns when gradients explode, when convergence accelerates, and how architecture affects optimization speed—enabling intelligent training acceleration.

---

## 🖥 Your Hardware

- i7 11th Gen, RTX 3050 Ti (4GB), 16GB RAM
- Sufficient for 50–100 MNIST experiments with efficient logging

---

## 🚀 Future Extensions

- CIFAR-10, tiny language models
- Integration with MoE routing, PB-ANN inhibition control, BrahmaLLM optimization
- Publishable research in learning-to-learn systems
