import numpy as np
import cv2
import matplotlib.pyplot as plt
import imutils
import evaluation_metrics
import utils


def detect_faceboxes(img, face_cascades, scale_step, size, img_num):

    face_boxes = viola_jones(img, face_cascades, scale_step, size, img_num)

    if len(face_boxes) != 0:
        max_facebox_surface = 0

        for i, face_box in enumerate(face_boxes):
            current_surface = face_box[2] * face_box[3]

            if current_surface > max_facebox_surface:
                max_facebox_surface = current_surface
                max_i = i

        x = face_boxes[max_i][0]
        y = face_boxes[max_i][1]
        w = face_boxes[max_i][2]
        h = face_boxes[max_i][3]

        cv2.rectangle(img, (x, y, x + w, y + h), (0, 0, 255), 2)
        roi = img[y:y + h, x:x + w]

    else:
        roi = img

        x = 0
        y = 0

    return roi, x, y, img


def viola_jones(img, cascades, scale_step, size, img_num):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #gray = cv2.equalizeHist(gray)
    detections = []

    for cascade in cascades:
        current_detection = cascade.detectMultiScale(gray, scale_step, size)
        if len(current_detection) != 0:
            detections.append(current_detection)

    if detections:
        detections_tuple = ()
        for detection in detections:
            # print(detection)
            detections_tuple = detections_tuple + (detection,)

        detections_np = np.concatenate(detections_tuple, axis=0)
        # print(img_num, detections_np, len(detections))
        return detections_np
    else:
        return []

def get_bbox_from_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]

    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2 = imutils.grab_contours(contours)

    bbox = np.empty([len(contours2), 4], dtype="int32")

    for i, contour in enumerate(contours2):
        x = contour[0][0][0]
        y = contour[0][0][1]

        h = contour[2][0][1] - contour[0][0][1]
        w = contour[2][0][0] - contour[0][0][0]

        # print(contour, "-->", x, y, h, w)
        bbox[i][0] = x
        bbox[i][1] = y
        bbox[i][2] = w
        bbox[i][3] = h

    return bbox


if __name__ == "__main__":

    img_num = 1

    left_ear_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_mcs_leftear.xml')

    right_ear_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_mcs_rightear.xml')

    face_cascade_1 = cv2.CascadeClassifier('./haarcascades/haarcascade_profileface.xml')
    face_cascade_2 = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')


    cascades = [left_ear_cascade, right_ear_cascade]
    face_cascades = [face_cascade_1, face_cascade_2]

    combined_TP = 0
    combined_FP = 0
    combined_gt_boxes = 0

    while img_num < 751:

        img_num_digits = len(str(img_num))
        zeros_to_add = 4 - img_num_digits
        img_num_str = zeros_to_add * "0" + str(img_num)

        img = cv2.imread('./AWEForSegmentation/train/' + img_num_str + '.png', 0)
        img_ear_mask = cv2.imread('./AWEForSegmentation/trainannot_rect/' + img_num_str + '.png')

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # viola jones parameters:
        scale_step = 1.05
        size = 2


        #scale_step = 1.3  # few faces
        #size = 5

        #scale_step = 1.5 # only one face
        #size = 10

        scale_step_facebox = 1.5
        size_facebox = 10

        use_faceboxes = False

        if use_faceboxes:
            roi, x, y, img2 = detect_faceboxes(img, face_cascades, scale_step_facebox, size_facebox, img_num)
            img = img2

        else:
            roi = img
            x = 0
            y = 0


        detections_bboxes = viola_jones(roi, cascades, scale_step, size, img_num)

        for detection_bbox in detections_bboxes:
            detection_bbox[0] += x
            detection_bbox[1] += y

        ground_truth_bboxes = get_bbox_from_mask(img_ear_mask)

        TP, FP, total_gt_boxes = evaluation_metrics.get_TP_FP_total_boxes(detections_bboxes, ground_truth_bboxes, 0.5)
        print(img_num, TP, FP, total_gt_boxes)
        combined_TP += TP
        combined_FP += FP
        combined_gt_boxes += total_gt_boxes

        for (x, y, w, h) in ground_truth_bboxes:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        for (x, y, w, h) in detections_bboxes:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


        #fig = plt.figure(figsize=(20, 12))
        #plt.suptitle("img:" + str(img_num))
        #ax = plt.subplot()
        #ax.imshow(img)
        #plt.show()


        img_num += 1

    try:
        precision = (combined_TP/(combined_TP + combined_FP))
    except ZeroDivisionError:
        precision = 0

    try:
        recall = (combined_TP/combined_gt_boxes)
    except ZeroDivisionError:
        recall = 0

    print("precision", precision)
    print("recall", recall )
    print("f score", (precision*recall)/(precision + recall))





