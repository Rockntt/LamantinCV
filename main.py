import cv2
import pytesseract

cap = cv2.VideoCapture(0)

pytesseract.pytesseract.tesseract_cmd = 'Tesseract/tesseract.exe'

alphabet1 = "ԽՁՒՅԷՐՏԵԸԻՈՕՊՌՉԺՋՓԹԼԿՃՀՔՖԴՍԱԶՑԳՎԲՆՄՇՂԾ"
alphabet2 = "ԱԲՑԷՌԵԽՈՒԹԵՅՈՒՄԵՆԻՔՈՒՄեՆԻՅոՒՅուՆԻքՈՒՅոՂՈՒԹՅՈւՆԻ"

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    text_data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, lang='hye', config='--psm 6')

    for i in range(len(text_data['text'])):
        if text_data['text'][i] and any(
                char in text_data['text'][i] for char in alphabet1):
            (x, y, w, h) = (text_data['left'][i], text_data['top'][i], text_data['width'][i], text_data['height'][i])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
