import os
import yaml
from torch.utils.data import Dataset
from PIL import Image

def load_config(base_path):
    with open(os.path.join(base_path, "configs/diseases.yaml")) as f:
        return yaml.safe_load(f)["diseases"]

def find_class_dir(root, class_name):
    direct = os.path.join(root, class_name)
    if os.path.isdir(direct):
        return direct
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            if d.lower() == class_name.lower():
                return os.path.join(dirpath, d)
    raise FileNotFoundError(
        f"Could not find folder '{class_name}' under '{root}'. "
        f"Available folders: {[d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]}"
    )

def gather_files(root, classes):
    paths, labels = [], []
    for i, c in enumerate(classes):
        cdir = find_class_dir(root, c)
        print(f"  using '{c}' -> {cdir}")
        for fn in os.listdir(cdir):
            if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                paths.append(os.path.join(cdir, fn))
                labels.append(i)
    return paths, labels

class ImgListDataset(Dataset):
    def __init__(self, paths, labels, tf):
        self.paths, self.labels, self.tf = paths, labels, tf
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.tf(img), self.labels[idx]
