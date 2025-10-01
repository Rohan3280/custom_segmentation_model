import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from src.models.vit_backbone import build_vit
from src.dataset.dataloader import Sentinel2Dataset

def train_model(config):
    dataset = Sentinel2Dataset(config["data"]["train_dir"])
    loader = DataLoader(dataset, batch_size=config["train"]["batch_size"], shuffle=True)

    model = build_vit(num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["train"]["learning_rate"])

    for epoch in range(config["train"]["epochs"]):
        for images, labels in loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}: loss = {loss.item()}")
    return model
