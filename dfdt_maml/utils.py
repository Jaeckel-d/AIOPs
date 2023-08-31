import os
import datetime as dt


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


class Logger:
    def __init__(self, log_path, end="\n", stdout=True):
        self.log_path = log_path
        self.end = end
        self.stdout = stdout

    def log(self, message):
        time_str = dt.datetime.strftime(dt.datetime.now(), '%Y-%m-%d %H:%M:%S')
        log_str = "[{}] {}{}".format(time_str, message, self.end)
        with open(self.log_path, "a") as f:
            f.write(log_str)
        if self.stdout:
            print(log_str, end="")

