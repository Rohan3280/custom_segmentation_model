from src.dataset.dataloader import Sentinel2Dataset

def test_dataloader():
    dataset = Sentinel2Dataset("data/processed/train")
    image, label = dataset[0]
    assert image.shape[0] == 3, "Image should have 3 channels (RGB)"
    assert label in [0, 1], "Label must be 0 or 1"
