import datetime
from colorama import init, Fore
from colorama import Back
from colorama import Style

class Logger:
    def __init__(self):
        self.log_file = ''

    def init_log_file(self):
        now = datetime.datetime.now()
        formatted_date = now.strftime("%Y-%m-%d_%H-%M-%S")
        with open(f"logs/log-{formatted_date}.txt", "w") as file:
            self.log_file = f"logs/log-{formatted_date}.txt"

    def log(self, message='n/a', level='n/a'):
        with open(self.log_file, 'a', encoding='utf-8') as f:
            now = datetime.datetime.now()
            formatted_time = now.strftime("%H:%M:%S")
            f.write(f"[{formatted_time}] {level}: {message}\n")
            if level == 'ERROR':
                status_color = Fore.LIGHTRED_EX
            elif level == 'WARNING':
                status_color = Fore.LIGHTYELLOW_EX
            elif level == "SUCCESS":
                status_color = Fore.LIGHTGREEN_EX
            else:
                status_color = Fore.LIGHTWHITE_EX
            print(status_color + f"[{formatted_time}] {level}: {message}")
