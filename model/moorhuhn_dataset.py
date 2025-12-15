import json
import os

import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from transformations import simple_resize


class MoorhuhnDataset(Dataset):
    def __init__(self, images_dir, annotation_file, img_size=640, transform=None):
        self.images_dir = images_dir
        self.img_size = img_size

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
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

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

        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id]),
            "area": (
                (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
                if len(boxes) > 0
                else torch.zeros((0,), dtype=torch.float32)
            ),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }

        if self.transform is not None:
            self.transform(image)
        else:
            image, target = simple_resize(image, target=target)
            image = T.ToTensor()(image)

        return image, target
