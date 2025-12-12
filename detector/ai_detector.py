import cv2
import torch
import torchvision
from torchvision import transforms as T

from detector.base import AbstractDetector
from model.model import load_retinanet_model
from model.transformations import inverse_simple_resize_boxes, simple_resize


# slow as Christmas
class AiDetector(AbstractDetector):

    def __init__(self, monitor_index: int | None = None, *args, **kwargs):
        super().__init__(monitor_index=monitor_index, *args, **kwargs)
        self.model, self.device = load_retinanet_model()

    def process_frame(self, frame):
        frame_resized, _ = simple_resize(frame.copy())
        transform = T.ToTensor()
        frame_tensor = transform(frame_resized).unsqueeze(0).to(self.device)

        with torch.no_grad():
            predictions = self.model(frame_tensor)

        bounding_boxes = predictions[0]["boxes"].cpu().numpy()
        scores = predictions[0]["scores"].cpu().numpy()
        labels = predictions[0]["labels"].cpu().numpy()

        conf_threshold = 0.35
        conf_mask = scores > conf_threshold

        bounding_boxes = bounding_boxes[conf_mask]
        scores = scores[conf_mask]
        labels = labels[conf_mask]

        if len(bounding_boxes) > 0:
            if not isinstance(bounding_boxes, torch.Tensor):
                bounding_boxes = torch.tensor(bounding_boxes)
            if not isinstance(scores, torch.Tensor):
                scores = torch.tensor(scores)

            keep_indices = torchvision.ops.nms(
                boxes=bounding_boxes, scores=scores, iou_threshold=0.2
            )

            bounding_boxes = bounding_boxes[keep_indices]
            scores = scores[keep_indices]
            labels = labels[keep_indices]

            bounding_boxes = inverse_simple_resize_boxes(frame, bounding_boxes)

        return bounding_boxes, None

    def draw_boxes(self, frame, boxes):
        # labels dead/alive
        for box in boxes:
            x1, y1, x2, y2 = [int(i.item()) for i in box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
            cv2.putText(
                frame,
                text="AI",
                org=(x1, y1 - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(255, 0, 0),
                thickness=3,
            )
        return frame
