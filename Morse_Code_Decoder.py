from Morse_codes import Characters
import matplotlib.pyplot as plt
import librosa as lr
import numpy as np
import itertools
import re



plt.style.use('bmh')
audio , sfreq = lr.load('Path of your wav file')

dash = re.compile(r'(0{8,30}5{8,30}){125,135}')
dot = re.compile(r'0{400,460}5{1}.{1000,1400}5{1}0{400,460}')
alpha_space =  re.compile(r'5{1}0{4230,4630}5{1}')
space = re.compile(r'0{14500,15000}')

Morse_array = []
threshold = 0.4 # Threshold may be required to change accordingly 

audio = list(filter(lambda a : a > 0 , audio)) #negative chop
audio = list(map(lambda b : 5 if b > threshold else 0 if threshold > b > 0 else 0 , audio))

audio_string = ''.join([str(au) for au in audio])

for a in dash.finditer(audio_string) : Morse_array.append((a.span() , '-'))
for b in dot.finditer(audio_string)  : Morse_array.append((b.span() , '.'))
for c in alpha_space.finditer(audio_string)   : Morse_array.append((c.span() , ' '))
for d in space.finditer(audio_string)   : Morse_array.append((d.span() , '       '))

Morse_array = sorted(Morse_array , key = lambda m : m[0][0])

morse_code = ''.join([m[1] for m in Morse_array])

encrypted_array = morse_code.split('       ')

fin_string = ''
for e in encrypted_array : 
    temp = ''
    for q in e.split(' ') : temp += Characters.get(q , '')
    fin_string += temp + ' '

print('Morse code :' , morse_code)
print('Decoded message :' , fin_string)
   
time = np.arange(0 , len(audio)) / sfreq

plt.ylabel('Amplitude')
plt.xlabel('Time')
plt.plot(time , audio)
plt.show()
