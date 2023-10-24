import cv2
from mtcnn import MTCNN

cap = cv2.VideoCapture(0)
detector = MTCNN()

while True:

    ret,frame = cap.read()

    output = detector.detect_faces(frame)

    for single_output in output: # for all the faces
        x,y,width,height = single_output['box']
        # cv2.rectangle(frame,pt1=(x,y),pt2=(x+width,y+height),color=(255,0,0),thickness=3)
        blurred_face = cv2.GaussianBlur(frame[y:y+height, x:x+width], (99,99), 0)
        frame[y:y+height, x:x+width]=blurred_face

    cv2.imshow('output',frame)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cv2.destroyAllWindows()