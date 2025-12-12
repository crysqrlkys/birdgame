import numpy as np
import torch
from PIL import Image


# works like ass
def letterbox(image, target=None, img_size=640):
    original_w, original_h = image.size

    scale = min(img_size / original_w, img_size / original_h)
    new_w = int(original_w * scale)
    new_h = int(original_h * scale)

    resized_image = image.resize((new_w, new_h), Image.BILINEAR)

    padded_image = Image.new("RGB", (img_size, img_size), (0, 0, 0))
    dx = (img_size - new_w) // 2
    dy = (img_size - new_h) // 2
    padded_image.paste(resized_image, (dx, dy))

    if target is not None and "boxes" in target:
        boxes = target["boxes"]

        scaled_boxes = boxes * scale
        scaled_boxes[:, 0] += dx
        scaled_boxes[:, 1] += dy
        scaled_boxes[:, 2] += dx
        scaled_boxes[:, 3] += dy

        scaled_boxes[:, 0::2] = scaled_boxes[:, 0::2].clamp(0, img_size)
        scaled_boxes[:, 1::2] = scaled_boxes[:, 1::2].clamp(0, img_size)

        valid_mask = (scaled_boxes[:, 2] > scaled_boxes[:, 0]) & (
            scaled_boxes[:, 3] > scaled_boxes[:, 1]
        )

        target["boxes"] = scaled_boxes[valid_mask]
        target["labels"] = target["labels"][valid_mask]
        target["area"] = target["area"][valid_mask]
        target["iscrowd"] = target["iscrowd"][valid_mask]

        target["orig_size"] = torch.tensor([original_h, original_w])
        target["pad_info"] = torch.tensor([scale, dx, dy])

    return padded_image, target


def simple_resize(image, target=None, target_size=640):
    if isinstance(image, np.ndarray):
        orig_height, orig_width, _ = image.shape
        image = Image.fromarray(image)
    else:
        orig_width, orig_height = image.size

    resized_image = image.resize((target_size, target_size), Image.BILINEAR)

    scale_x = target_size / orig_width
    scale_y = target_size / orig_height

    if target is not None:
        boxes = target["boxes"]
        if len(boxes) > 0:
            scaled_boxes = boxes.clone()
            scaled_boxes[:, 0] *= scale_x
            scaled_boxes[:, 2] *= scale_x
            scaled_boxes[:, 1] *= scale_y
            scaled_boxes[:, 3] *= scale_y

            target["boxes"] = scaled_boxes

            return resized_image, target

    return resized_image, target


def inverse_simple_resize_boxes(original_frame, boxes, target_size=640):
    if isinstance(original_frame, np.ndarray):
        orig_height, orig_width, _ = original_frame.shape
        original_frame = Image.fromarray(original_frame)
    else:
        orig_width, orig_height = original_frame.size
    orig_width, orig_height = original_frame.size

    scale_x = target_size / orig_width
    scale_y = target_size / orig_height

    if boxes is None or len(boxes) == 0:
        return boxes

    original_boxes = boxes.clone()
    original_boxes[:, 0] /= scale_x
    original_boxes[:, 2] /= scale_x
    original_boxes[:, 1] /= scale_y
    original_boxes[:, 3] /= scale_y

    return original_boxes
