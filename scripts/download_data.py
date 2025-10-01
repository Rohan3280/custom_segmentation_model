import os
import requests

def download_example_data(save_dir="data/raw"):
    os.makedirs(save_dir, exist_ok=True)
    url = "https://example.com/sample_glof_lake_tile.jpg"  # replace with real dataset link
    save_path = os.path.join(save_dir, "sample_glof_lake_tile.jpg")

    print(f"Downloading {url}...")
    r = requests.get(url)
    with open(save_path, "wb") as f:
        f.write(r.content)
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    download_example_data()
