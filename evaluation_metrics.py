import numpy as np
import utils
from collections import Counter

def intersection_over_union(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def get_TP_FP_total_boxes(detection_rects, ground_truth_rects, iou_treshold):

    if len(detection_rects) == 0:
        return 0, 0, len(ground_truth_rects)

    if len(ground_truth_rects) == 0:
        return 0, len(detection_rects), 0

    detections = []
    ground_truths = []

    for i, gt_rect in enumerate(ground_truth_rects):
        gt_corners = utils.bounding_rect_to_corners(gt_rect)
        ground_truths.append(gt_corners)

    for j, det_rect in enumerate(detection_rects):
        det_corners = utils.bounding_rect_to_corners(det_rect)
        detections.append(det_corners)

    gt_bboxes_matched = [0] * len(ground_truths)

    TP = 0
    FP = 0
    total_gt_boxes = len(ground_truths)

    for det_idx, detection in enumerate(detections):
        best_iou = -1

        for gt_idx, ground_truth in enumerate(ground_truths):
            iou = intersection_over_union(detection, ground_truth)

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou > iou_treshold:
            if gt_bboxes_matched[best_gt_idx] == 0:
                TP += 1
                gt_bboxes_matched[best_gt_idx] = 1
            else:
                FP += 1
        else:
            FP += 1

    return TP, FP, total_gt_boxes




