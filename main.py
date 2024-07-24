# pylint: disable=no-name-in-module
# pylint: disable=no-member
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals

"""
Подключение необходимых библиотек (pip install -r requirements.txt)
"""
import sys
import time
import math
import logger

try:
    import cv2
    import numpy as np
    import pytesseract
    from PIL import ImageFont, ImageDraw, Image
    import imutils
except ModuleNotFoundError:
    print("Please install required libraries: $ pip install -r requirements.txt")
    sys.exit(1)


# Инициализация логгера
logs = logger.Logger()
logs.init_log_file()


def calculate_lat_long(center_latitude,
                       center_longitude,
                       pixel_x, pixel_y,
                       image_width,
                       image_height,
                       altitude):
    """
    Вычисляет GPS-координаты точки на снимке, зная координаты центра снимка,
    пиксельные координаты точки, размер изображения и высоту съемки.

    Args:
    center_latitude: Широта центра снимка в градусах.
    center_longitude: Долгота центра снимка в градусах.
    pixel_x: Координата X точки на снимке в пикселях.
    pixel_y: Координата Y точки на снимке в пикселях.
    image_width: Ширина изображения в пикселях.
    image_height: Высота изображения в пикселях.
    altitude: Высота съемки в метрах.

    Returns:
    Кортеж с GPS-координатами точки (широта, долгота) в градусах.
    """

    # Вычисляем размер пикселя в метрах, используя высоту съемки и размер изображения
    pixel_size = (image_width * math.tan(math.radians(45))) / altitude

    # Вычисляем расстояние от центра снимка до точки в метрах
    x_offset = (pixel_x - image_width / 2) * pixel_size
    y_offset = (pixel_y - image_height / 2) * pixel_size

    # Переводим широту и долготу в радианы
    center_latitude_rad = math.radians(center_latitude)
    center_longitude_rad = math.radians(center_longitude)

    # Вычисляем новые координаты в метрах
    new_latitude_meters = y_offset + center_latitude_rad * 6371000  # 6371000 - средний радиус Земли
    new_longitude_meters = x_offset + center_longitude_rad * 6371000 * math.cos(center_latitude_rad)

    # Переводим новые координаты в радианы
    new_latitude_rad = new_latitude_meters / 6371000
    new_longitude_rad = new_longitude_meters / (6371000 * math.cos(center_latitude_rad))

    # Переводим координаты в градусы
    new_latitude = math.degrees(new_latitude_rad)
    new_longitude = math.degrees(new_longitude_rad)

    return new_latitude, new_longitude


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


def get_cap(cam_number=0):
    """
    Функция get_cap() возвращает объект capture для работы с камерой
    Необязательный аргумент cam_number позволяет указать желаемую камеру, если их несколько
    """
    try:
        capture = cv2.VideoCapture(cam_number)
        _ret, _frame = capture.read()
        height, width, _ = _frame.shape
        if not capture.isOpened():  # Проверка открытия камеры
            raise RuntimeError("Camera not found. Check that your camera is connected.")
    except RuntimeError:  # Обработка ситуация отсутствия камеры
        sys.exit(1)
    logs.log(f"Capture {cam_number} opened", "SUCCESS")
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
cap, cam_h, cam_w = get_cap(1)

# Цикл обработки кадров
while True:
    start_time = time.perf_counter()
    ret, frame = cap.read()  # Чтение кадра из потока камеры

    frame_p = frame_preprocessing(frame, cam_h, cam_w)  # Обработка полученного кадра

    # Распознавание текста с помощью библиотеки Tesseract
    text_data = pytesseract.image_to_data(
                                        frame_p,
                                        output_type=pytesseract.Output.DICT,
                                        lang='arm',  # Базовый язык + Дообучение
                                        config='--psm 10')  # psm 10 подходит для 1 симв.

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
                end_time = time.perf_counter()
                s_time = round(end_time - start_time, 3)
                # Информация об успешных распознаваняих
                logs.log(
                    f"{text_data['text'][i]} | {conf}% | {s_time}s",
                    "SUCCESS")

                img_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(img_pil)
                draw.text((x - 80, y - 80),
                          text_data['text'][i],
                          font=UNICODE_FONT,
                          fill=(87, 87, 87, 0))
                frame = np.array(img_pil)

    # Отображаются 2 потока: 1) с распознаванием; 2) без, но обработанный
    cv2.imshow('Video', frame)
    cv2.imshow('Preprocessed Video', frame_p)

    # Завершение работы при нажатии клавиши "q"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        logs.log("Initiated exiting by pressing 'Q'", "INFO")
        break

# Завершение работы
cap.release()
logs.log("Cap released", "INFO")
cv2.destroyAllWindows()
logs.log("Program finished", "INFO")
