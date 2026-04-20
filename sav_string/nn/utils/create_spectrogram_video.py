import os
import re
import numpy as np
from tqdm import tqdm
import imageio.v2 as imageio
from pdf2image import convert_from_path

def spectrograms_to_video(folder_path, output_path="output.mp4", fps=8):
    pattern = re.compile(r"predicted_spectrogram_step_(\d+)\.pdf")

    # --- Load reference spectrogram once ---
    reference_path = os.path.join(folder_path, "reference_spec.pdf")
    ref_img = convert_from_path(reference_path, first_page=1, last_page=1)[0]
    ref_img = np.array(ref_img)

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

            pred_img = convert_from_path(filepath, first_page=1, last_page=1)[0]
            pred_img = np.array(pred_img)

            # --- Ensure same height ---
            ref_pil = Image.fromarray(ref_img)
            pred_pil = Image.fromarray(pred_img)

            # Resize reference to match predicted height (or vice versa)
            if ref_pil.height != pred_pil.height:
                ref_pil = ref_pil.resize(
                    (int(ref_pil.width * pred_pil.height / ref_pil.height),
                     pred_pil.height)
                )

            # --- Concatenate side-by-side ---
            combined = Image.new(
                "RGB",
                (ref_pil.width + pred_pil.width, pred_pil.height)
            )

            combined.paste(ref_pil, (0, 0))
            combined.paste(pred_pil, (ref_pil.width, 0))

            writer.append_data(np.array(combined))

    print(f"Video saved to {output_path}")


if __name__ == '__main__':

    folder_path = '/u/anasynth/linares/LearningTheString/sav_string/runs/20260412_020303'

    spectrograms_to_video(folder_path)