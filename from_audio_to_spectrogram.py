import numpy as np
from scipy.signal import spectrogram
from scipy.io.wavfile import read
import matplotlib.pyplot as plt

# Load the .wav file.
filename = 'E2_string_test'
sampling_rate, audio_data = read(f'{filename}.wav')

# Compute the spectrogram.
frequencies, times, Sxx = spectrogram(audio_data, fs=sampling_rate, window='hann', nperseg=4096 * 2, noverlap=512, scaling='spectrum')
# Add a small epsilon.
epsilon = 1e-10
Sxx = Sxx + epsilon

# Step 3: Plot the spectrogram
plt.figure(figsize=(10, 6))
plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
plt.title('Spectrogram')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar(label='Power/Frequency (dB/Hz)')
plt.ylim(0, 6000)
plt.legend()
plt.savefig(f'{filename}.pdf')