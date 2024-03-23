import cv2
import numpy as np
import pytesseract
from PIL import ImageFont, ImageDraw, Image

cap = cv2.VideoCapture(0)

pytesseract.pytesseract.tesseract_cmd = 'Tesseract-OCR/tesseract.exe'

alphabet1 = "ԽՁՒՅԷՐՏԵԸԻՈՕՊՌՉԺՋՓԹԼԿՃՀՔՖԴՍԱԶՑԳՎԲՆՄՇՂԾ"
alphabet2 = "ԱԲՑԷՌԵԽՈՒԹԵՅՈՒՄԵՆԻՔՈՒՄեՆԻՅոՒՅուՆԻքՈՒՅոՂՈՒԹՅՈւՆԻ"

fontpath = "FreeSerif.ttf"

counter = 0


def frame_preprocessing(f):
    f = cv2.GaussianBlur(f, (5, 5), 0)
    gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    invert = 255 - thresh
    return invert


while True:
    ret, frame = cap.read()

    # img_h, img_w, ch = frame.shape

    frame_p = frame_preprocessing(frame)

    text_data = pytesseract.image_to_data(frame_p, output_type=pytesseract.Output.DICT, lang='arm+hye', config='--psm 10')

    for i in range(len(text_data['text'])):
        if text_data['text'][i] and any(char in text_data['text'][i] for char in alphabet2):
            (x, y, w, h) = (text_data['left'][i], text_data['top'][i], text_data['width'][i], text_data['height'][i])
            if len(text_data['text'][i]) == 1 and text_data['text'][i] in alphabet2:
                print(text_data['text'][i], w, h)
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

                font = ImageFont.truetype(fontpath, 130)
                img_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(img_pil)
                draw.text((x, y), text_data['text'][i], font=font, fill=(255,255,0,0))
                frame = np.array(img_pil)

    cv2.imshow('Video', frame)
    cv2.imshow('Preprocessed Video', frame_p)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
