import numpy as np
import pandas as pd

RANGE = [40, 80, 120, 180, 300]

def getIndex(freq):
    i = 0
    while RANGE[i] < freq:
        i += 1
    return i

FUZ_FACTOR = 2

def hash(p1, p2, p3, p4):
    return (p4 - (p4 % FUZ_FACTOR)) * 100000000 + (p3 - (p3 % FUZ_FACTOR)) * 100000 + (p2 - (p2 % FUZ_FACTOR)) * 100 + (p1 - (p1 % FUZ_FACTOR))

def getLandmarks(song_name, song, duration):
    CHUNK_SIZE = 1024

    chunks = int(duration*44100/CHUNK_SIZE)
    chunk_interval = CHUNK_SIZE/44100

    results = []
    intervals = []

    # Para cada chunk
    for times in range(chunks):
        x = np.zeros(CHUNK_SIZE, dtype=complex)
        for i in range(CHUNK_SIZE):
            # Coloca el dominio de amplitud en un numero complejo
            x[i] = complex(song[(times*CHUNK_SIZE) + i], 0)
        
        # Aplica FFT en el chunk
        intervals.append(times*chunk_interval)
        results.append(np.fft.fft(x))

    highscores = np.zeros((len(results), 5))
    points = np.zeros((len(results), 5))
    h = np.zeros(len(results))

    for t in range(len(results)):
        for freq in range(40, 300):
            mag = np.log(np.abs(results[t][freq]) + 1)

            index = getIndex(freq)

            if mag > highscores[t][index]:
                highscores[t][index] = mag
                points[t][index] = freq

        h[t] = hash(points[t][0], points[t][1], points[t][2], points[t][3])

    songId = [song_name] * len(results)

    intervals = np.array(intervals)
    songId = np.array(songId)

    df = pd.DataFrame({'id': songId, 'time': intervals, 'hash': h})
    df = df[df['hash'] != '0.0']

    return df

def match_songs(db_dir, audio, duration):
    landmarks_test = getLandmarks('Test 1', audio, duration)
    landmarks_db = pd.read_csv(db_dir, header=0)

    landmarks_db['diff'] = 0

    for index, row in landmarks_test.iterrows():
        df = landmarks_db[landmarks_db['hash'] == row['hash']]
        
        for index2, row2 in df.iterrows():
            diff = abs(row2['time'] - row['time'])
            landmarks_db.loc[index2, 'diff'] = diff

    landmarks_db = landmarks_db[landmarks_db['diff'] != 0]
    df = landmarks_db.groupby(['id', 'diff']).size().groupby('id').sum()

    return df