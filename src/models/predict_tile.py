import torch
from PIL import Image
from torchvision import transforms
from src.models.vit_backbone import build_vit

def predict(image_path, checkpoint):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    model = build_vit(num_classes=2, checkpoint=checkpoint)
    model.eval()

    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1).item()
    return "GLOF-prone lake" if pred == 1 else "Normal lake"

if __name__ == "__main__":
    result = predict("examples/sample_input/test.jpg", "experiments/weights/vit_glof_custom.pt")
    print("Prediction:", result)
