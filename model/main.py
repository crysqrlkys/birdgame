import torch
import torch.optim as optim
from dataset import MoorhuhnDataset
from torch.utils.data import DataLoader

from model import create_retinanet_model

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


if __name__ == "__main__":
    dataset = MoorhuhnDataset(images_dir=IMAGES_FOLDER, annotation_file=ANNOTATION_FILE)
    train_loader = DataLoader(
        dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x))
    )
    model = create_retinanet_model()

    train_model(model, train_loader=train_loader)
