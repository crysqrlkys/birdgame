import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as T

from dataset import MoorhuhnDataset
from model import create_retinanet_model
from transformations import simple_resize

IMAGES_FOLDER = "dataset/images"
ANNOTATION_FILE = "dataset/annotations/coco.json"


def train_model(model, train_loader, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.0001,
        weight_decay=0.0005,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,
        T_mult=2,
        eta_min=1e-6,
    )

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        for images, targets in train_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += losses.item()
            num_batches += 1

        avg_train_loss = train_loss / num_batches if num_batches > 0 else 0

        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        print("-" * 50)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epochs,
        },
        "moorhuhn_retinanet.pth",
    )

    return model


def test_model():
    model = create_retinanet_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load("moorhuhn_retinanet.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    test_img_path = "dataset/test/test_1.jpg"
    test_img = Image.open(test_img_path).convert("RGB")
    test_img, _ = simple_resize(test_img)
    transform = T.ToTensor()
    test_tensor = transform(test_img).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(test_tensor)

    boxes = predictions[0]["boxes"].cpu().numpy()
    scores = predictions[0]["scores"].cpu().numpy()
    labels = predictions[0]["labels"].cpu().numpy()

    conf_threshold = 0.4
    conf_mask = scores > conf_threshold

    boxes = boxes[conf_mask]
    scores = scores[conf_mask]
    labels = labels[conf_mask]

    if len(boxes) > 0:
        if not isinstance(boxes, torch.Tensor):
            boxes = torch.tensor(boxes)
        if not isinstance(scores, torch.Tensor):
            scores = torch.tensor(scores)

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
    dataset = MoorhuhnDataset(images_dir=IMAGES_FOLDER, annotation_file=ANNOTATION_FILE)
    train_loader = DataLoader(
        dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x))
    )
    model = create_retinanet_model()

    train_model(model, train_loader=train_loader)
