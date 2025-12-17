import cv2
import numpy as np
import torch


# works like ass, small birds become too small (needs precise anchor adjustment)
def letterbox(image, target_size=640, target=None):
    original_w, original_h = image.shape[:2]

    scale = min(target_size / original_w, target_size / original_h)
    new_w = int(original_w * scale)
    new_h = int(original_h * scale)

    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    padded_image = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    dx = (target_size - new_w) // 2
    dy = (target_size - new_h) // 2
    padded_image[dy : dy + new_h, dx : dx + new_w] = resized_image

    if target is not None and "boxes" in target:
        boxes = target["boxes"]

        scaled_boxes = boxes * scale
        scaled_boxes[:, [0, 2]] += dx
        scaled_boxes[:, [1, 3]] += dy

        scaled_boxes[:, 0::2] = scaled_boxes[:, 0::2].clamp(0, target_size)
        scaled_boxes[:, 1::2] = scaled_boxes[:, 1::2].clamp(0, target_size)

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


def restore_letterbox_boxes(image, boxes, target_size=640):
    if boxes.numel() == 0:
        return boxes

    original_w, original_h = image.shape[:2]

    scale = min(target_size / original_w, target_size / original_h)
    new_w = int(original_w * scale)
    new_h = int(original_h * scale)

    dx = (target_size - new_w) // 2
    dy = (target_size - new_h) // 2

    restored = boxes.clone()

    restored[:, [0, 2]] -= dx
    restored[:, [1, 3]] -= dy

    restored = restored / scale

    restored[:, 0::2] = restored[:, 0::2].clamp(0, original_w)
    restored[:, 1::2] = restored[:, 1::2].clamp(0, original_h)

    return restored


def get_scale_factors(orig_shape, target_size=640):
    h, w = orig_shape
    return target_size / w, target_size / h


def scale_boxes(boxes, scale_x, scale_y):
    if boxes is None or len(boxes) == 0:
        return boxes

    scaled = boxes.clone()
    scaled[:, [0, 2]] *= scale_x
    scaled[:, [1, 3]] *= scale_y
    return scaled


def simple_resize(image, target_size=640, target=None):
    h, w = image.shape[:2]
    resized = cv2.resize(image, (target_size, target_size), cv2.INTER_AREA)
    scale_x, scale_y = get_scale_factors((h, w), target_size)

    if target is not None and "boxes" in target:
        target["boxes"] = scale_boxes(target["boxes"], scale_x, scale_y)

    return resized, target


def restore_simple_resize_boxes(image, boxes, target_size=640):
    h, w = image.shape[:2]
    if boxes is None or len(boxes) == 0:
        return boxes

    scale_x, scale_y = get_scale_factors((h, w), target_size)

    restored = boxes.clone()
    restored[:, [0, 2]] /= scale_x
    restored[:, [1, 3]] /= scale_y

    return restored
