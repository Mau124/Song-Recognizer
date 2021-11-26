import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import librosa as lr
import analize_song as analize
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

# Read song
# data_dir = './test_songs'
# audio_files = glob(data_dir + '/*.wav')

# audio, sfreq = lr.load(audio_files[2], sr = 44100, mono = True)
# time = np.arange(0, len(audio)) / sfreq
# duration = lr.get_duration(audio, sr = 44100)

###############################################

fs=44100
duration = 15  # seconds
audio = sd.rec(duration * fs, samplerate=fs, channels=1,dtype='float64')
print("Recording Audio")
sd.wait()
print ("Play Audio Complete")

################################################

landmarks_test = analize.getLandmarks('Test 1', audio, duration)
#print(landmarks_test)

landmarks_db = pd.read_csv('landmarks.csv', header=0)
#print(landmarks_db)

landmarks_db['diff'] = 0

for index, row in landmarks_test.iterrows():
    #print(hash_test)
    #print(row)
    #print(row['time'])
    df = landmarks_db[landmarks_db['hash'] == row['hash']]
    
    for index2, row2 in df.iterrows():
        diff = abs(row2['time'] - row['time'])
        landmarks_db.loc[index2, 'diff'] = diff
    # print(df['time'] - row['time'])
    # diff = df['time'] - row['time']
    # landmarks_db['diff'] = landmarks_db['diff'].mask(landmarks_db[diff.loc == landmarks_db.loc], diff)
    
    

    #print(df)
landmarks_db.to_csv('data.csv')
#ans = result.groupby(['id']).count()
print(len(landmarks_db))

landmarks_db = landmarks_db[landmarks_db['diff'] != 0]
df = landmarks_db.groupby(['id', 'diff']).size().groupby('id').sum()

print(df)
df.to_csv('data2.csv')

print(type(df))
#total = df[df.loc[:, 0] == '\Adele - Rolling in the Deep'].sum()
#print(total)




#print(landmarks_db.hash == landmarks_test.hash)
# Plot song
# fig, ax = plt.subplots()
# ax.plot(time, audio)
# ax.set(xlabel = 'Time(s)', ylabel = 'Sound Amplitud')
# plt.show()
