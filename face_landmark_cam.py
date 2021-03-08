import cv2
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        print('no frame.');break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for rect in faces:
        x,y = rect.left(), rect.top()
        w,h = rect.right()-x, rect.bottom()-y
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
    
        shape = predictor(gray, rect)
        for i in range(68):
            part = shape.part(i)
            cv2.circle(img, (part.x, part.y), 2, (0, 0, 255), -1)
    
    cv2.imshow("face landmark", img)
    if cv2.waitKey(1)== 27:
        break
cap.release()



