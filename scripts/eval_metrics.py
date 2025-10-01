import torch
from sklearn.metrics import classification_report
from src.dataset.dataloader import Sentinel2Dataset
from torch.utils.data import DataLoader
from src.models.vit_backbone import build_vit

def evaluate(checkpoint, data_dir):
    dataset = Sentinel2Dataset(data_dir)
    loader = DataLoader(dataset, batch_size=8)

    model = build_vit(num_classes=2, checkpoint=checkpoint)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())

    print(classification_report(y_true, y_pred, target_names=["Normal", "GLOF-prone"]))

if __name__ == "__main__":
    evaluate("experiments/weights/vit_glof_custom.pt", "data/processed/test")
