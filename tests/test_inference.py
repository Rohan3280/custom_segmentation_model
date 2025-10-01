from src.inference.predict_tile import predict
import os

def test_inference():
    checkpoint = "experiments/weights/vit_glof_custom.pt"
    sample_img = "examples/sample_input/test.jpg"
    if not os.path.exists(checkpoint) or not os.path.exists(sample_img):
        return  # skip test if files aren't available
    result = predict(sample_img, checkpoint)
    assert result in ["Normal lake", "GLOF-prone lake"], "Unexpected prediction output"
