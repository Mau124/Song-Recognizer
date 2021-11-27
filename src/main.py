import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import read_song as rsong
import analize_song as analize

#############################################

# Read song from file
# audio, time, duration = rsong.read_song_from_file('./test_songs/test1.wav')

# Read song from micro
audio, time, duration = rsong.read_song_from_micro(15)

# Find coincidences between songs
df = analize.match_songs('../landmarks.csv', audio, duration)

# Get Fourier transform for graphing

fourier = np.abs(np.fft.fft(audio))
freq = np.abs(np.fft.fftfreq(time.shape[-1]))

# Graph everything

plt.style.use('seaborn-darkgrid')
fig, ax = plt.subplots(2, 2)

ax[0, 0].plot(time, audio, color='darkcyan')
ax[0, 0].set_title('Gráfica Amplitud - Tiempo')
ax[0, 0].set_xlabel('Tiempo (s)')
ax[0, 0].set_ylabel('Amplitud (dB)')

ax[0, 1].plot(freq, fourier, color='darkcyan')
ax[0, 1].set_title('Transformada de Fourier')
ax[0, 1].set_xlabel('Frecuencia (Hz)')
ax[0, 1].set_ylabel('F(w)')

ax[1, 0].bar(x = np.arange(len(df)), height = df, color='darkcyan')
ax[1, 0].set_title('Coincidencias')
ax[1, 0].set_xlabel('Índice de la canción')
ax[1, 0].set_ylabel('Número de coincidencias')

table_songs = pd.DataFrame(df.index)

print('\n############################################## \n')
print('Table de puntaje de canciones')

print(df)

print('\n############################################## \n')
print('Id de cada canción que se muestra en la gráfica de coincidencias')

print(table_songs)

print('\n############################################## \n')
print('\nLa canción con mayor relación a la muestra es')

print(df.idxmax())

print('\n############################################## \n')

plt.show()
