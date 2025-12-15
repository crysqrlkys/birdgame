import torch
import torchvision
from torchvision import transforms as T

from detector.base.ai_detector import AiDetector
from model.models import load_retinanet_model
from model.transformations import restore_simple_resize_boxes, simple_resize


class RetinaNetDetector(AiDetector):

    CONFIDENCE_THRESHOLD = 0.2
    IOU_THRESHOLD = 0.2

    def __init__(self, monitor_index: int | None = None, *args, **kwargs):
        super().__init__(monitor_index=monitor_index, *args, **kwargs)
        self.model, self.device = load_retinanet_model()
        self.transform = T.ToTensor()

    def process_frame(self, frame):
        frame_resized, _ = simple_resize(frame.copy())
        frame_tensor = self.transform(frame_resized).unsqueeze(0).to(self.device)

        with torch.no_grad():
            predictions = self.model(frame_tensor)

        boxes = predictions[0]["boxes"]
        scores = predictions[0]["scores"]
        labels = predictions[0]["labels"]

        conf_mask = scores > self.CONFIDENCE_THRESHOLD
        boxes = boxes[conf_mask]
        scores = scores[conf_mask]
        labels = labels[conf_mask]

        if len(boxes) > 0:
            keep_indices = torchvision.ops.nms(
                boxes=boxes, scores=scores, iou_threshold=self.IOU_THRESHOLD
            )

            boxes = boxes[keep_indices]
            labels = labels[keep_indices]

            boxes = restore_simple_resize_boxes(frame, boxes)
            boxes = boxes.cpu().numpy()
            labels = labels.cpu().numpy()

        return boxes, labels
