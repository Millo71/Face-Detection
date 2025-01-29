import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)

while True: 
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    num_faces = len(faces)
    contatore = 1

    for (x, y, w, h) in faces:

        font = cv2.FONT_HERSHEY_PLAIN
        text_x = x + 1
        text_y = y - 5

        if text_y < 0:
            text_y = y + h + 20 

        cv2.putText(frame, f'#{contatore}', (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        contatore += 1

        if contatore > 2:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.putText(frame, f'Persone rilevate: {num_faces}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if num_faces > 1:
        num_face_corretto = num_faces - 1
        cv2.putText(frame, f'Intrusi: {num_face_corretto}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

video_capture.release()
cv2.destroyAllWindows()