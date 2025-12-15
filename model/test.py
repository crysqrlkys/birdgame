import matplotlib.pyplot as plt
import torch
import torchvision
from PIL import Image
from torchvision import transforms as T
from transformations import simple_resize

from model.models import load_retinanet_model


def test_model():
    model, device = load_retinanet_model()

    test_img_path = "dataset/test/test_1.jpg"
    test_img = Image.open(test_img_path).convert("RGB")
    test_img, _ = simple_resize(test_img)
    transform = T.ToTensor()
    test_tensor = transform(test_img).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(test_tensor)

    boxes = predictions[0]["boxes"]
    scores = predictions[0]["scores"]
    labels = predictions[0]["labels"]

    conf_threshold = 0.4
    conf_mask = scores > conf_threshold

    boxes = boxes[conf_mask]
    scores = scores[conf_mask]
    labels = labels[conf_mask]

    if len(boxes) > 0:
        keep_indices = torchvision.ops.nms(
            boxes=boxes, scores=scores, iou_threshold=0.2
        )

        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        labels = labels[keep_indices]

    plt.imshow(test_img)
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        plt.gca().add_patch(
            plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, fill=False, color="red", linewidth=2
            )
        )
        plt.text(x1, y1, f"{label}:{score:.2f}", color="yellow")

    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    test_model()
