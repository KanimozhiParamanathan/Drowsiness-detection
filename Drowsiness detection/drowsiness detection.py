import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer

mixer.init()
sound = mixer.Sound('D:/c#/phase/Drowsiness detection/alarm.wav')

face = cv2.CascadeClassifier('D:/c#/phase/Drowsiness detection/haar cascade files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('D:/c#/phase/Drowsiness detection/haar cascade files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('D:/c#/phase/Drowsiness detection/haar cascade files/haarcascade_righteye_2splits.xml')

model = load_model('models/cnncat2.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

open_score = 0
closed_score = 0
threshold = 15  # Threshold to trigger alarm
thicc = 2

while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    open_text = ''
    closed_text = ''

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y + h, x:x + w]
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (24, 24))
        r_eye = r_eye / 255.0
        r_eye = np.expand_dims(r_eye, axis=-1)
        r_eye = np.expand_dims(r_eye, axis=0)
        rpred = model.predict(r_eye)
        lbl = 'Closed' if rpred[0][0] > rpred[0][1] else 'Open'

        if lbl == 'Closed':
            closed_score += 1
            open_score = 0
            if closed_score > threshold:
                cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
                try:
                    sound.play()
                except:
                    pass
                if thicc < 16:
                    thicc += 2
                else:
                    thicc -= 2
                    if thicc < 2:
                        thicc = 2
                cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc) 
        else:
            open_score += 1
            closed_score = 0

    if open_score > 0:
        open_text = 'Open Score: ' + str(open_score)
    elif closed_score > 0:
        closed_text = 'Closed Score: ' + str(closed_score)

    # Draw scores on frame
    cv2.putText(frame, open_text, (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, closed_text, (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
