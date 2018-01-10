import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


def detect(gray, img):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # min=1.3, max=5
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes=eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    return img


# detect faces with webcam

video_capture=cv2.VideoCapture(0)

while True:
    _, img= video_capture.read()
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canvas=detect(gray, img)
    cv2.imshow("Video", canvas)
    if cv2.waitKey(1) and 0xFF == ord("q"):
        break
video_capture.release()