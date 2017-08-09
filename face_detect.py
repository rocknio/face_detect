# !/usr/bin/python3
# -*- coding: utf-8 -*-

import time

import cv2
import imutils

__author__ = "Neo"


def detect_faces(gray_frame):
    face_cascade = cv2.CascadeClassifier("/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    results = []
    for (x, y, width, height) in faces:
        results.append((x, y, x + width, y + height))

    return results


def draw_rectangle(frame, positions):
    for (x, y, x1, y1) in positions:
        cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)


def detect_eyes(gray_frame, faces):
    eye_cascade = cv2.CascadeClassifier("/usr/local/share/OpenCV/haarcascades/haarcascade_eye.xml")
    # eye_cascade = cv2.CascadeClassifier("/usr/local/share/OpenCV/haarcascades/haarcascade_eye_tree_eyeglasses.xml")
    results = []
    for (x1, y1, x2, y2) in faces:
        roi_gray = gray_frame[y1:y2, x1:x2]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 2)
        for (ex, ey, ew, eh) in eyes:
            results.append((x1 + ex, y1 + ey, x1 + ex + ew, y1 + ey + eh))

    return results


def detect_smiles(gray_frame):
    smile_cascade = cv2.CascadeClassifier("/usr/local/share/OpenCV/haarcascades/haarcascade_smile.xml")
    smiles = smile_cascade.detectMultiScale(gray_frame, 4, 5)
    results = []
    for (x, y, width, height) in smiles:
        results.append((x, y, x + width, y + height))

    return results


def camera_face_detect():
    camera = cv2.VideoCapture(0)
    time.sleep(0.5)

    # 获取视频每一帧
    while True:
        # 获取当前帧，初始化occupied/unoccupied文本
        (grabbed, frame) = camera.read()
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detect_faces(gray)
        draw_rectangle(frame, faces)

        eyes = detect_eyes(gray, faces)
        draw_rectangle(frame, eyes)

        # smiles = detect_smiles(gray)
        # draw_rectangle(frame, smiles)

        cv2.imshow("faces", frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


def picture_face_detect(file_name):
    img = cv2.imread(file_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detect_faces(gray)
    draw_rectangle(img, faces)

    # eyes = detect_eyes(gray, faces)
    # draw_rectangle(img, eyes)
    #
    # smiles = detect_smiles(gray)
    # draw_rectangle(img, smiles)

    cv2.imshow("dest_" + file_name, img)
    while True:
        key = cv2.waitKey(1)
        if key == ord("q"):
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    # camera_face_detect()
    picture_face_detect("./test.jpg")
    picture_face_detect("./test1.jpg")
    picture_face_detect("./test2.jpg")
    picture_face_detect("./test3.jpg")
    picture_face_detect("./test4.png")
    picture_face_detect("./test5.jpg")

