import numpy as np
import librosa as lr
from glob import glob

import analize_song as analize

data_dir = '../songs'
audio_files = glob(data_dir + '/*.wav')

header = False

for song in audio_files:
    audio, sfreq = lr.load(song, sr = 44100, mono = True)
    time = np.arange(0, len(audio)) / sfreq
    duration = lr.get_duration(audio, sr = 44100)

    # Obtenemos el nombre de la cancion
    song_name = song.replace(data_dir, '')
    song_name = song_name.replace('.wav', '')

    # Obtenemos los puntos criticos de la cancion
    song_landmarks = analize.getLandmarks(song_name, audio, duration)

    if header:    
        song_landmarks.to_csv('landmarks.csv', mode='a', index=False, header=None)
    else:
        song_landmarks.to_csv('landmarks.csv', mode='a', index=False)
        header = True
    print(song_landmarks)