'''
face_cascade = cv2.CascadeClassifier('haarcascades2/HS.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
left_ear_cascade = cv2.CascadeClassifier('./haarcascades2/haarcascade_mcs_leftear.xml')
right_ear_cascade = cv2.CascadeClassifier('./haarcascades2/haarcascade_mcs_rightear.xml')


print(cv2.__version__)

counter = 53

while counter < 750:

    counter_digits = len(str(counter))
    zeros_to_add = 4 - counter_digits

    img_num = zeros_to_add * "0" + str(counter)
    print(img_num)


    img = cv2.imread('./AWEForSegmentation/train/' + img_num + '.png')

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    print(img.shape)

    # viola jones parameters:
    #scale_step = 1.1  # most faces
    #size = 2

    scale_step = 1.3 # few faces
    size = 5

    # scale_step = 1.5 # only one face
    # size = 10

    right_ears_detected = right_ear_cascade.detectMultiScale(gray, scale_step, size)
    left_ears_detected = left_ear_cascade.detectMultiScale(gray, scale_step, size)

    #ears_detected = face_cascade.detectMultiScale(gray, scale_step, size)

    print(left_ears_detected)
    print(right_ears_detected)

    if(len(left_ears_detected) == 0):
        ears_detected = right_ears_detected
    elif(len(right_ears_detected) == 0):
        ears_detected = left_ears_detected
    elif(len(left_ears_detected) == 0 and len(right_ears_detected) == 0):
        ears_detected = [[]]
    else:
        ears_detected = np.concatenate((left_ears_detected, right_ears_detected), axis=0)


    print(ears_detected)

    for (x, y, w, h) in ears_detected:
        print(x, y, w, h)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        # eyes = eye_cascade.detectMultiScale(roi_gray)
        # for (ex,ey,ew,eh) in eyes:
        #    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    images = [img]

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle("Viola-Jones ear detection", fontsize=16)

    for ix, image in enumerate(images):
        ax = plt.subplot("11" + str(ix + 1))
        # ax.set_title("k = " + str(len(ids)) + ", Expression: " + emo)
        ax.imshow(image)

    # show the generated faces
    plt.show()
    counter+=1
    '''
