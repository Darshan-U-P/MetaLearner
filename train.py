import torch
import torch.nn as nn
import torch.optim as optim
from meta_logger import MetaLogger
from model import SimpleModel
from utils import compute_activation_sparsity


device = "cuda" if torch.cuda.is_available() else "cpu"

model = SimpleModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler()

logger = MetaLogger(model, optimizer)

for step in range(200):

    inputs = torch.randn(32, 128).to(device)
    targets = torch.randint(0, 10, (32,)).to(device)

    with torch.cuda.amp.autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

    sparsity = compute_activation_sparsity(model)
    fp16_overflow = scaler.get_scale() < 1

    logger.log_step(
        loss,
        activation_sparsity=sparsity,
        fp16_overflow=fp16_overflow
    )

logger.close()