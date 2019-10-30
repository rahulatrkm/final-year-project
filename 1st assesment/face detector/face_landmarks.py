import cv2
import numpy as np
import dlib

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    _, video_input = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the face
    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        ## Mark the rectangle around the face.
        # cv2.rectangle(<image>, (top-left co-ordinate), (bottom-right co-ordinate), (b,g,r)<color of the rectangle, thickness)
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        landmarks = predictor(gray, face)

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        margin = 25
        eye_left_x1 = landmarks.part(37).x - margin - 4
        eye_left_x2 = landmarks.part(40).x + margin + 4
        eye_left_y1 = landmarks.part(39).y - margin - 4
        eye_left_y2 = landmarks.part(42).y + margin + 4

        eye_right_x1 = landmarks.part(43).x - margin
        eye_right_x2 = landmarks.part(46).x + margin
        eye_right_y1 = landmarks.part(44).y - margin
        eye_right_y2 = landmarks.part(47).y + margin
        cv2.rectangle(frame, (eye_left_x1, eye_left_y1), (eye_left_x2, eye_left_y2), (0, 0, 255), 2)
        cv2.rectangle(frame, (eye_right_x1, eye_right_y1), (eye_right_x2, eye_right_y2), (0, 0, 255), 2)

        left_eye = video_input[eye_left_y1:eye_left_y2, eye_left_x1:eye_left_x2]
        cv2.imshow("Left Eye", left_eye)
        right_eye = video_input[eye_right_y1:eye_right_y2, eye_right_x1:eye_right_x2]
        cv2.imshow("Right Eye", right_eye)
        #cv2.circle(frame, (landmarks.part(39).x, landmarks.part(39).y), 2, (0, 0, 255), -1)
        #cv2.line(frame, (landmarks.part(39).x, 0), (landmarks.part(39).x, landmarks.part(39).y), (0,255,0), 2)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
