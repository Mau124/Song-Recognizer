import librosa as lr
import numpy as np
from glob import glob
import sounddevice as sd

def read_song_from_file(song_dir):
    data_dir = './test_songs'
    audio_files = glob(data_dir + '/*.wav')

    audio, sfreq = lr.load(song_dir, sr = 44100, mono = True)
    time = np.arange(0, len(audio)) / sfreq
    duration = lr.get_duration(audio, sr = 44100)

    return audio, time, duration

def read_song_from_micro(duration):
    fs=44100
    audio = sd.rec(duration * fs, samplerate=fs, channels=1,dtype='float64')
    time = np.arange(0, len(audio)) / fs
    print("Recording Audio")
    sd.wait()
    print ("Play Audio Complete")

    return audio, time, duration