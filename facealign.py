import cv2
from mtcnn import MTCNN
import math
cap = cv2.VideoCapture(0)
detector = MTCNN()

while True:
    ret, frame = cap.read()

    output = detector.detect_faces(frame)

    for single_output in output:
        x, y, width, height = single_output['box']
        keypoints = single_output['keypoints']

        # Draw bounding box
        cv2.rectangle(frame, pt1=(x, y), pt2=(x+width, y+height), color=(255, 0, 0), thickness=3)

        # Draw facial landmarks
        cv2.circle(frame, center=keypoints['left_eye'], radius=5, color=(0, 255, 0), thickness=-1)
        cv2.circle(frame, center=keypoints['right_eye'], radius=5, color=(0, 255, 0), thickness=-1)
        cv2.circle(frame, center=keypoints['nose'], radius=5, color=(0, 255, 0), thickness=-1)
        cv2.circle(frame, center=keypoints['mouth_left'], radius=5, color=(0, 255, 0), thickness=-1)
        cv2.circle(frame, center=keypoints['mouth_right'], radius=5, color=(0, 255, 0), thickness=-1)

        # Calculate eye line angle
        eye_line_angle = (keypoints['right_eye'][1] - keypoints['left_eye'][1]) / (keypoints['right_eye'][0] - keypoints['left_eye'][0])
        eye_line_angle_degrees = round(math.degrees(math.atan(eye_line_angle)), 2)

        # Display alignment information
        cv2.putText(frame, f'Alignment: {eye_line_angle_degrees} degrees', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow('win', frame)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cv2.destroyAllWindows()
