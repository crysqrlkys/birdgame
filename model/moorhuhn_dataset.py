import json
import os

import cv2
import torch
import torchvision.transforms.v2 as v2
from PIL import Image
from torch.utils.data import Dataset
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat


class MoorhuhnDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        annotation_file: str,
        transform: v2.Compose = None,
    ):
        self.images_dir = images_dir
        self.transform = transform

        with open(annotation_file, "r") as f:
            self.coco_data = json.load(f)

        self.image_id_to_info = {img["id"]: img for img in self.coco_data["images"]}
        self.image_id_to_annotations = {}

        for ann in self.coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[img_id] = []
            self.image_id_to_annotations[img_id].append(ann)

        self.category_map = {
            cat["id"]: cat["name"] for cat in self.coco_data["categories"]
        }

        self.image_ids = list(self.image_id_to_info.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.image_id_to_info[image_id]

        img_path = os.path.join(self.images_dir, image_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        width, height = image.size

        # image = cv2.imread(img_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        # image = torch.from_numpy(image).permute(2, 0, 1)  # .float() / 255.0

        annotations = self.image_id_to_annotations.get(image_id, [])

        boxes = []
        labels = []

        for ann in annotations:
            x, y, w, h = ann["bbox"]
            category_id = ann["category_id"]

            x1, y1, x2, y2 = x, y, x + w, y + h
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(category_id)

        boxes = BoundingBoxes(
            boxes, format=BoundingBoxFormat.XYXY, canvas_size=(height, width)
        )
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id]),
        }

        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target
