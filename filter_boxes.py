from typing import List, Tuple

def compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Compute Intersection over Union (IoU) between two boxes."""
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    xi1, yi1 = max(x1, x1_p), max(y1, y1_p)
    xi2, yi2 = min(x2, x2_p), min(y2, y2_p)

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area

def filter_detections(
    boxes: List[Tuple[int, int, int, int]],
    confidences: List[float]
) -> List[int]:
    """Filter bounding boxes based on aspect ratio, area, and IoU."""
    
    # Step 1: Aspect ratio & area filter
    filtered_indices = []
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        width = x2 - x1
        height = y2 - y1
        area = width * height

        if width > 3 * height:
            continue
        if area < 400:
            continue

        filtered_indices.append(i)

    # Step 2: IoU-based suppression
    final_indices = []
    filtered = sorted(filtered_indices, key=lambda i: confidences[i], reverse=True)

    while filtered:
        current = filtered.pop(0)
        final_indices.append(current)

        filtered = [
            idx for idx in filtered
            if compute_iou(boxes[current], boxes[idx]) <= 0.4
        ]

    return final_indices

## test usage 
# if __name__ == "__main__":
#     test_boxes = [(10, 20, 110, 60), (15, 25, 105, 55), (300, 400, 340, 420)]
#     test_confs = [0.9, 0.8, 0.95]
#     kept_indices = filter_detections(test_boxes, test_confs)
#     print("Kept:", kept_indices)
