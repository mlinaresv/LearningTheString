import os
import time
import requests

def send_slack(message):
    webhook_url="https://hooks.slack.com/services/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"
    payload = {"text": message}
    requests.post(webhook_url, json=payload)


def get_folder_count(path):
    return len([
        f for f in os.listdir(path)
        if os.path.isdir(os.path.join(path, f))
    ])


def watch_folders(path, interval=10, notify=None):
    previous_count = 0# get_folder_count(path)

    while True:
        time.sleep(interval)

        current_count = get_folder_count(path)

        if current_count > previous_count:
            new_folders = current_count - previous_count

            if notify:
                send_slack(
                    message=f"{new_folders} new folder(s) detected in {path}",                    
                )

            previous_count = current_count


if __name__ == '__main__':

    watch_folders(
        path='/data2/anasynth_nonbp/linares/LearningTheString/runs/20260424_155645/',
        notify=True
    )