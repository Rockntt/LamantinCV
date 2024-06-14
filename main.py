# pylint: disable=no-name-in-module
# pylint: disable=no-member

"""
Подключение необходимых библиотек (pip install -r requirements.txt)
"""
import sys

# Импорт необходимых библиотек
# pip install -r requirements.txt
import cv2
import numpy as np
import pytesseract
from PIL import ImageFont, ImageDraw, Image
import imutils


def crop_by_color(_frame, height, width):
    """
    Функция crop_by_color() обрезает изображение до прямоугольника заданного цвета
    :param _frame: кадр для обрезки
    :param height: высота выходного кадра
    :param width: ширина выходного кадра
    :return: обрезанный кадр
    """

    # Ищем контур прямоугольника
    cnts = cv2.findContours(_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    _x, _y, _w, _h = cv2.boundingRect(c)

    # Обрезаем изображение по координатам прямоугольника
    _frame = _frame[_y:_y + _h, _x:_x + _w]
    return cv2.resize(_frame, (width, height))


def get_cap():
    """
    Функция get_cap() возвращает объект capture для работы с камерой
    """
    try:
        capture = cv2.VideoCapture(0)
        _ret, _frame = capture.read()
        height, width, _ = _frame.shape
        if not capture.isOpened():  # Проверка открытия камеры
            raise RuntimeError("Camera not found. Check that your camera is connected.")
    except RuntimeError:  # Обработка ситуация отсутствия камеры
        sys.exit(1)
    return capture, height, width


def frame_preprocessing(_frame, height, width, resize_multiplier=1.5):
    """
        Функция frame_preprocessing обрабатывает изображение, делая его более пригодным для OCR
        :param _frame: кадр, который нужно обработать
        :param height: высота выходного кадра
        :param width: ширина выходного кадра
        :param resize_multiplier: коэффициент масштабирования
        :return: обработанный кадр
    """

    _frame = cv2.GaussianBlur(_frame, (7, 7), 0)
    gray_f = cv2.cvtColor(_frame, cv2.COLOR_BGR2GRAY)
    thresh_f = cv2.threshold(gray_f, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    preprocessed_frame = 255 - thresh_f
    if resize_multiplier != 1:
        return cv2.resize(crop_by_color(preprocessed_frame, height, width),
                          None,
                          fx=resize_multiplier,
                          fy=resize_multiplier,
                          interpolation=cv2.INTER_AREA)
    return crop_by_color(preprocessed_frame, height, width)


# Инициализация бинарника установленного Tesseract-OCR
pytesseract.pytesseract.tesseract_cmd = 'Tesseract-OCR/tesseract.exe'

# Список букв армянского алфавита для сравнения вывода
ALPHABET = "ԱԲԳԴԵԶԷԸԹԺԻԼԽԾԿՀՁՂՃՄՅՆՇՈՉՊՋՌՍՎՏՐՑՒՓՔ"

# Инициализация шрифта, поддерживающего Unicode (для вывода армянских букв)
UNICODE_FONT = ImageFont.truetype("FreeSerif.ttf", 200)

# Инициализация камеры
cap, cam_h, cam_w = get_cap()

# Цикл обработки кадров
while True:
    ret, frame = cap.read()  # Чтение кадра из потока камеры

    frame_p = frame_preprocessing(frame, cam_h, cam_w, 1.5)  # Обработка полученного кадра

    # Распознавание текста с помощью библиотеки Tesseract
    text_data = pytesseract.image_to_data(
                                        frame_p,
                                        output_type=pytesseract.Output.DICT,
                                        lang='arm',  # Базовый язык + Дообучение
                                        config='--psm 10')  # Ищет 1 символ

    for i in range(len(text_data['text'])):
        if len(text_data['text'][i]) == 1 and text_data['text'][i] in ALPHABET:
            (x, y, w, h, conf) = (text_data['left'][i],
                                  text_data['top'][i],
                                  text_data['width'][i],
                                  text_data['height'][i],
                                  text_data['conf'][i])

            # Отрисовка сектора с найденным текстом. Отключено, поскольку бесполезно для задачи,
            # поскольку Tesseract часто возвращает сектор по площади больший, чем сама буква в кадре
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

            # Отрисовка распознанной буквы в координатах, переданных из image_to_data()
            if conf > 80:  # Вероятность успеха больше 80%

                # Информация об успешных распознаваняих
                # print(f"Recognized {text_data['text'][i]} with {conf}% confidence")

                img_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(img_pil)
                draw.text((x - 80, y - 80),
                          text_data['text'][i],
                          font=UNICODE_FONT,
                          fill=(87, 87, 87, 0))
                frame = np.array(img_pil)
            # Информация об ошибочных распознаваниях
            # else:
            #     print(f"Tried to recognize {text_data['text'][i]} but {conf}")

    # Отображаются 2 потока: 1) с распознаванием; 2) без, но обработанный
    cv2.imshow('Video', frame)
    cv2.imshow('Preprocessed Video', frame_p)

    # Завершение работы при нажатии клавиши "q"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Завершение работы
cap.release()
cv2.destroyAllWindows()
