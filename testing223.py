import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascades2/HS.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
left_ear_cascade = cv2.CascadeClassifier('./haarcascades2/haarcascade_mcs_leftear.xml')
right_ear_cascade = cv2.CascadeClassifier('./haarcascades2/haarcascade_mcs_rightear.xml')

img_num = "0019"

img = cv2.imread('./AWEForSegmentation/train/' + img_num + '.png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

scale_step = 1.1  # most faces
size = 2

right_ears_detected = right_ear_cascade.detectMultiScale(gray, scale_step, size)
left_ears_detected = left_ear_cascade.detectMultiScale(gray, scale_step, size)

det = []

for i in range(3):
    det.append(right_ears_detected)

print(det[1][0])



#ears_detected = np.concatenate((left_ears_detected, right_ears_detected), axis=1)





'''
if (len(left_ears_detected) == 0):
    ears_detected = right_ears_detected
elif (len(right_ears_detected) == 0):
    ears_detected = left_ears_detected
elif (len(left_ears_detected) == 0 and len(right_ears_detected) == 0):
    ears_detected = [[]]
else:
    ears_detected = np.concatenate((left_ears_detected, right_ears_detected), axis=0)
'''