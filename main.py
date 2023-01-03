import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from pandas import read_csv

fr = True
path = 'images'
images = []
classNames = []
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def find_encodings(images):
    encode_list = []
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(image)[0]
        encode_list.append(encode)
    return encode_list


def mark_attendance(name):
    date = datetime.now().strftime('%d-%m-%Y')
    with open(f"Attendance_of_{date}.csv", "a+") as f:
        global fr
        if fr:
            if os.path.getsize(f"Attendance_of_{date}.csv") == 0:
                f.writelines("Name,Time,Date")
            fr = False

        else:
            data = read_csv(f"Attendance_of_{date}.csv")
            name_list = data["Name"].to_list()
            if name not in name_list:
                time_now = datetime.now()
                t_str = time_now.strftime('%H:%M:%S')
                d_str = time_now.strftime('%d/%m/%Y')
                f.writelines(f'\n{name},{t_str},{d_str}')


encodeListKnown = find_encodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            mark_attendance(name)

    cv2.imshow('Attendance Camera', img)
    cv2.waitKey(1)

# find . -name '.DS_Store' -type f -delete
