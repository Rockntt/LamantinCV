# pylint: disable=no-name-in-module
# pylint: disable=no-member

"""
Подключение необходимых библиотек (pip install -r requirements.txt)
"""
import cv2
import numpy as np
import pytesseract
from PIL import ImageFont, ImageDraw, Image

# Инициализация камеры
cap = cv2.VideoCapture(0)

# Инициализация бинарника установленного Tesseract-OCR
pytesseract.pytesseract.tesseract_cmd = 'Tesseract-OCR/tesseract.exe'

# Список букв армянского алфавита для сравнения вывода
ALPHABET = "ԱԲԳԴԵԶԷԸԹԺԻԼԽԾԿՀՁՂՃՄՅՆՇՈՉՊՋՌՍՎՏՐՑՒՓՔ"

# Инициализация шрифта, поддерживающего Unicode (для вывода армянских букв)
UNICODE_FONT = ImageFont.truetype("FreeSerif.ttf", 130)


def frame_preprocessing(f):
    """
    Функция frame_preprocessing обрабатывает изображение, делая его более пригодным для OCR
    Параметры:
        f: Изображение, которое нужно обработать
    Возвращает:
        inverted_f: Обработанное изображение
    """
    f = cv2.GaussianBlur(f, (5, 5), 0)
    gray_f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    thresh_f = cv2.threshold(gray_f, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    inverted_f = 255 - thresh_f
    return inverted_f


# Цикл обработки кадров
while True:
    ret, frame = cap.read()  # Чтение кадра из потока камеры

    frame_p = frame_preprocessing(frame)  # Обработка полученного кадра

    # Распознавание текста с помощью библиотеки Tesseract
    text_data = pytesseract.image_to_data(
                                        frame_p,
                                        output_type=pytesseract.Output.DICT,
                                        lang='arm+hye',  # Базовый язык + Дообучение
                                        config='--psm 10')  # Ищет 1 символ

    for i in range(len(text_data['text'])):
        if len(text_data['text'][i]) == 1 and text_data['text'][i] in ALPHABET:
            (x, y, w, h) = (text_data['left'][i],
                            text_data['top'][i],
                            text_data['width'][i],
                            text_data['height'][i])
            # print(text_data['text'][i], w, h) - Можно использовать для отладки

            # Отрисовка сектора с найденным текстом. Отключено, поскольку бесполезно для задачи,
            # поскольку Tesseract часто возвращает сектор по площади больший, чем сама буква в кадре
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

            # Отрисовка распознанной буквы в координатах, переданных из image_to_data()
            img_pil = Image.fromarray(frame)
            draw = ImageDraw.Draw(img_pil)
            draw.text((x, y), text_data['text'][i], font=UNICODE_FONT, fill=(255, 255, 0, 0))
            frame = np.array(img_pil)

    # Отображаются 2 потока: 1) с распознаванием; 2) без, но обработанный
    cv2.imshow('Video', frame)
    cv2.imshow('Preprocessed Video', frame_p)

    # Завершение работы при нажатии клавиши "q"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Завершение работы
cap.release()
cv2.destroyAllWindows()
