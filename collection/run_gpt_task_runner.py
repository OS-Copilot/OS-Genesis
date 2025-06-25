import os
import time


def run_gpt_task():
    os.system("python gpt_task_runner.py")

while True:
    run_gpt_task()
    time.sleep(20)