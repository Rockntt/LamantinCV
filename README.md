# SKAT_2024

Репозиторий программы компьютерного зрения для Ku-7 в рамках конкурса СКАТ

Статус: Интеграция обученной модели YOLO9 в программу.

### Запуск программы:

```shell
git clone https://github.com/Rockntt/SKAT_2024.git
cd SKAT_2024
pip install -r requirements.txt
python main.py
```

Tesseract-OCR предустановлен в репозитории, поэтому указывать путь до его исполняемого файла не нужно, однако при необходимости воспользоваться Tesseract-ом вне кода (например, в целях переобучения/перенастройки) необходимо добавить директорию Tesseract-OCR в среду переменных Path вашей системы.

Инструкция для Windows:

1. Панель управления
2. Система (О системе)
3. Дополнительные параметры системы
4. Переменные среды
5. Выбрать и открыть переменную среды "Path"
6. Нажмите "Создать" и добавьте путь к директории Tesseract-OCR, например: 
```C:\ваша\рабочая\директория\SKAT_2024\Tesseract-OCR```.
7. Выполните в терминале команду `tesseract --version`, чтобы удостовериться, что вы сделали всё верно. В случае возникновения ошибки `"tesseract" не является внутренней или внешней
командой, исполняемой программой или пакетным файлом` попробуйте запустить терминал от имени администратора.

### Пример работы распознавания
![example.gif](readme_assets/example.gif)

### Пример работы функционала обрезки кадра по щиту с буквой
![example2.gif](readme_assets/example2.gif)

### Работы над Go-версией ведутся в процессе в ветке golang_test

#### Остались вопросы? Задайте их ниже

<a href="https://discordapp.com/users/1184134942326804595" target="_blank">
  <img src="https://img.shields.io/badge/Discord-%235865F2.svg?style=for-the-badge&logo=discord&logoColor=white" alt="Discord">
</a>
<a href="https://t.me/Rockntt" target="_blank">
  <img src="https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white" alt="Telegram">
</a>


