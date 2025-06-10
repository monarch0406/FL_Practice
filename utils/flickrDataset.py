from PIL import Image
import torch
from torch.utils.data import Dataset


# ------------------------
# Flickr8k Dataset Loader with OpenCLIP Tokenizer
# ------------------------
class Flickr8kDataset(Dataset):
    def __init__(self, image_dir, caption_file, tokenizer, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.data = []
        with open(caption_file, "r") as f:
            for line in f:
                img_id, caption = line.strip().split("\t")
                img_file = img_id.split("#")[0].strip()
                if not img_file.endswith('.jpg'):
                    img_file += ".jpg"
                self.data.append((img_file, caption))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_file, caption = self.data[idx]
        # print(f"Loading image: {img_file}")
        # print(f"Caption: {caption}")
        img_path = f"{self.image_dir}/{img_file}"
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)

        tokens = self.tokenizer([caption])[0]
        return image, tokens