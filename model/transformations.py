import torch
import torchvision.transforms.v2 as v2


# works like ass, small birds become too small (needs precise anchor adjustment)
class Letterbox(v2.Transform):
    def __init__(self, target_size: int = 640, fill=0):
        super().__init__()
        self.target_size = target_size
        self.fill = fill

    def __call__(self, image, target=None):
        h, w = image.shape[-2:]
        scale = min(self.target_size / w, self.target_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resize = v2.Resize((new_h, new_w))
        image, target = resize(image, target)

        pad_h = self.target_size - new_h
        pad_w = self.target_size - new_w

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        pad = v2.Pad((pad_left, pad_top, pad_right, pad_bottom), fill=self.fill)
        image, target = pad(image, target)

        if target is not None:
            target = dict(target)
            target["letterbox_meta"] = {
                "scale": scale,
                "pad": (pad_left, pad_top, pad_right, pad_bottom),
                "orig_size": (h, w),
            }

        return image, target


def infer_simple_resize(target_size: int = 640) -> v2.Compose:
    return v2.Compose(
        [
            v2.Resize(size=(target_size, target_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )


def create_simple_resize(target_size: int = 640) -> v2.Compose:
    return v2.Compose(
        [
            v2.RandomHorizontalFlip(),
            v2.Resize(size=(target_size, target_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )


def infer_letterbox(target_size: int = 640) -> v2.Compose:
    return v2.Compose(
        [
            Letterbox(target_size=target_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )


def create_letterbox(target_size: int = 640) -> v2.Compose:
    return v2.Compose(
        [
            v2.RandomHorizontalFlip(),
            Letterbox(target_size=target_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )


def restore_simple_resize_boxes(image, boxes, target_size=640):
    orig_h, orig_w = image.shape[:2]

    scale_x = orig_w / target_size
    scale_y = orig_h / target_size

    boxes = boxes.clone()
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y
    return boxes
