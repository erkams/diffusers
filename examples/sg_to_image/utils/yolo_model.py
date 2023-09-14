from ultralytics import FastSAM
from ultralytics import YOLO
from PIL import Image
from huggingface_hub import hf_hub_download
import torch
import torchvision.transforms as T
import numpy as np


class ObjectDetectionMetrics:
    def __init__(self, iou=0.4, dataset_type='clevr') -> None:
        super().__init__()

        model_path = hf_hub_download(repo_id='erkam/yolo-clevr', filename='best.pt')
        self.iou = iou
        self.dataset_type = dataset_type
        self.yolo_model = YOLO(model_path)
        self.sam_model = FastSAM('FastSAM-s.pt')

    def calculate(self, img, boxes_gt, objects_gt):
        # check if img is Tensor or PIL Image
        # print(f'img type: {type(img)}')

        if isinstance(img, torch.Tensor):
            img = T.ToPILImage()(img)
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        elif not isinstance(img, Image.Image):
            raise TypeError("img should be Tensor, PIL Image or numpy array")

        img = img.resize((640, 640), Image.BILINEAR)

        results = self.sam_model(img, conf=0.75, iou=0.4, max_det=20, device='cpu', verbose=False)

        if len(results[0].boxes.data) == 0:
            return 0, 0, 0

        boxes = results[0].boxes.data / torch.tensor([img.size[0], img.size[1], img.size[0], img.size[1], 1, 1],
                                                     dtype=torch.float32)
        boxes = boxes.tolist()
        boxes = sorted(boxes, key=lambda bbox: (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        # print(boxes)
        boxes = [box[:-2] for box in boxes]
        boxes_pred = filter_boxes(boxes)

        if len(boxes_pred) == 0:
            return 0, 0, 0

        num_ground_truth = len(boxes_gt)
        num_predictions = len(boxes_pred)

        true_positives = np.zeros(num_predictions)
        false_positives = np.ones(num_predictions)

        iou_matrix = np.zeros((num_ground_truth, num_predictions))

        for i, gt_bbox in enumerate(boxes_gt):
            for j, pred_bbox in enumerate(boxes_pred):
                iou = calculate_iou(gt_bbox, pred_bbox[:4])
                iou_matrix[i, j] = iou

        assigned_preds = []
        perm = {}
        while np.any(iou_matrix > self.iou):
            max_i, max_j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            perm[max_i] = max_j
            assigned_preds.append(max_j)
            iou_matrix[max_i, :] = 0
            iou_matrix[:, max_j] = 0

        for i in assigned_preds:
            true_positives[i] = 1
            false_positives[i] = 0

        cumulative_true_positives = np.cumsum(true_positives)
        cumulative_false_positives = np.cumsum(false_positives)
        precision_box = []
        recall_box = []
        for i in range(num_predictions):
            precision_box.append(cumulative_true_positives[i] / (i + 1))
            recall_box.append(cumulative_true_positives[i] / num_ground_truth)

        ap_box = compute_average_precision(precision_box, recall_box)

        if len(perm) == 0 or self.dataset_type != 'clevr':
            return ap_box, 0, len(boxes_pred)

        # check if the objects are correct
        ordered_boxes_pred = [boxes_pred[perm[i]] for i in perm.keys()]

        num_preds = len(ordered_boxes_pred)
        objs = np.zeros(num_preds)
        for i, box in enumerate(ordered_boxes_pred):
            mask = create_binary_mask(box, 640, padding=5)

            masked_im = T.ToTensor()(img) * mask
            masked_im = T.ToPILImage()(masked_im)
            results = self.yolo_model.predict(masked_im, augment=True, conf=0.2, iou=0.4, max_det=1, agnostic_nms=True,
                                              verbose=False, device='cpu')
            if len(results[0].boxes.cls) > 0:
                objs[i] = int(results[0].boxes.cls[0].item())

        true_positives = np.zeros(num_preds)
        false_positives = np.ones(num_preds)

        for i in range(num_preds):
            if i >= len(objects_gt):
                continue
            if objs[i] == objects_gt[i]:
                true_positives[i] = 1
                false_positives[i] = 0

        cumulative_true_positives = np.cumsum(true_positives)
        cumulative_false_positives = np.cumsum(false_positives)

        precision_obj = []
        recall_obj = []
        for i in range(num_preds):
            precision_obj.append(cumulative_true_positives[i] / (i + 1))
            recall_obj.append(cumulative_true_positives[i] / num_ground_truth)

        ap_obj = compute_average_precision(precision_obj, recall_obj)

        return ap_box, ap_obj, len(boxes_pred)


def is_bbox_contained(bbox1, bbox2):
    """
    Check if bbox1 contains bbox2.

    Parameters:
    - bbox1: A tuple or list representing the first bounding box (x1, y1, x2, y2).
    - bbox2: A tuple or list representing the second bounding box (x1, y1, x2, y2).

    Returns:
    - True if bbox1 contains bbox2, False otherwise.
    """
    # check if they are torch tensor
    if isinstance(bbox1, torch.Tensor):
        bbox1 = bbox1.tolist()
    if isinstance(bbox2, torch.Tensor):
        bbox2 = bbox2.tolist()

    if bbox1 == bbox2:
        return False
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Check if bbox2 is completely inside bbox1
    if x1_1 <= x1_2 and y1_1 <= y1_2 and x2_1 >= x2_2 and y2_1 >= y2_2:
        return True
    else:
        return False


def filter_boxes(boxes):
    filtered_boxes = []
    for box in boxes:
        contained = False
        for container in boxes:
            if is_bbox_contained(container, box):
                # print(f'{box} is in {container}')
                contained = True
                break
        if not contained:
            x1, y1, x2, y2 = box
            if abs(x2 - x1) / abs(y2 - y1) < 1.7:
                filtered_boxes.append(box)

    return filtered_boxes


def create_binary_mask(bbox, image_size, padding=5):
    """
    Create a binary mask with ones inside the specified bounding box and zeros elsewhere.

    Args:
        padding: The number of pixels to add to each side of the bounding box.
        bbox (list): The bounding box coordinates [x1, y1, x2, y2] (normalized).
        image_size (int): The size of the image (assumed to be square).

    Returns:
        torch.Tensor: A binary mask with shape (1, image_size, image_size).
    """
    # Create an empty mask with zeros
    mask = torch.zeros((1, image_size, image_size))

    # Convert normalized coordinates to pixel coordinates
    x1, y1, x2, y2 = [int(coord * image_size) for coord in bbox]

    # Add padding to the bounding box
    x1 -= padding
    y1 -= padding
    x2 += padding
    y2 += padding

    # Ensure the bounding box stays within the image bounds
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, image_size)
    y2 = min(y2, image_size)

    # Set the region inside the bounding box to ones
    mask[0, y1:y2, x1:x2] = 1

    return mask


def calculate_iou(box1, box2):
    # Calculate IoU between two bounding boxes
    # box1 and box2 are represented as [x1, y1, x2, y2]
    x1_intersection = max(box1[0], box2[0])
    y1_intersection = max(box1[1], box2[1])
    x2_intersection = min(box1[2], box2[2])
    y2_intersection = min(box1[3], box2[3])

    if x1_intersection >= x2_intersection or y1_intersection >= y2_intersection:
        return 0.0

    intersection_area = (x2_intersection - x1_intersection) * (y2_intersection - y1_intersection)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou


def compute_average_precision(precision, recall):
    # Compute Average Precision (AP) using precision-recall values
    precision = [1, *precision]
    recall = [0, *recall]

    ap = 0
    for i in range(1, len(precision)):
        ap += (recall[i] - recall[i - 1]) * precision[i]
    return ap
