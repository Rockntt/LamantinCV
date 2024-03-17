import cv2
import pytesseract

reader = easyocr.Reader(['en'])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = reader.readtext(frame)

    for detection in result:
        top_left = tuple(detection[0][0])
        bottom_right = tuple(detection[0][2])
        text = detection[1]
        font = cv2.FONT_HERSHEY_SIMPLEX
        frame = cv2.rectangle(frame, top_left, bottom_right, (0,255,0), 3)
        frame = cv2.putText(frame, text, top_left, font, 1, (255,255,255), 2, cv2.LINE_AA)

    cv2.imshow('Text Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()