import os
import re
import numpy as np
from tqdm import tqdm
import imageio.v2 as imageio
from pdf2image import convert_from_path

def spectrograms_to_video(folder_path, output_path="output.mp4", fps=8):
    pattern = re.compile(r"predicted_spectrogram_step_(\d+)\.pdf")

    files = []
    for f in os.listdir(folder_path):
        match = pattern.match(f)
        if match:
            step = int(match.group(1))
            files.append((step, f))

    files.sort(key=lambda x: x[0])

    with imageio.get_writer(output_path, fps=fps) as writer:
        for step, filename in tqdm(files):
            filepath = os.path.join(folder_path, filename)

            images = convert_from_path(filepath, first_page=1, last_page=1)
            image = np.array(images[0])  # ✅ FIX

            writer.append_data(image)

    print(f"Video saved to {output_path}")


if __name__ == '__main__':

    folder_path = '/u/anasynth/linares/LearningTheString/sav_string/runs/20260412_020303'

    spectrograms_to_video(folder_path)