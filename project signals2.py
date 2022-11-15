import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.fftpack import fft

t =   np.linspace(0 , 3 , 12 * 1024)
arrayf=[293.66,329.63,392,392,261.63,261.63,293.66,329.63,261.63,293.66,329.63,392,220,174.61,293]
x3=0
x4=0
for x in arrayf:
    x1 = (np.sin(2 * np.pi * 0 * t)+ np.sin( 2 * x *np.pi*t))*(t <= 0.2+x4)*(t >= x4)
    x4=x4+0.2
    x3=x3+x1
N = 3 * 1024

f = np. linspace(0 , 512 , int(N/2))

f1,f2= np.random.randint(0, 512, 2)

u1=np.zeros(np.shape(t))
u2=np.zeros(np.shape(t))
u1[t>=0]=1
u2[t>=3]=1
n1=(np.sin(2*f1*np.pi*t)+np.sin(2*f2*np.pi*t))*(t>=0)*(t<=3)
#fourier transfer original song
x_f1 = fft(x3)

x_f1 = (2/N) * np.abs(x_f1[0:int(N/2)])

nt=x3+n1

x_f2 = fft(nt)
x_f2 = (2/N) * np.abs(x_f2[0:int(N/2)])

Maxpeak=np.round(np.amax(x_f1))

array2=[]

for y in range(len(x_f2)):
    if x_f2[y]>(Maxpeak):
        array2.append(y)

f1_new =np.round(f[array2[0]])
f2_new =np.round(f[array2[1]])

x_filtered=nt-(np.sin( f1_new*2*np.pi*t)+ np.sin( f2_new*2*np.pi*t))
x_f3 = fft(x_filtered)
x_f3 = (2/N) * np.abs(x_f3[0:int(N/2)])

sd.play(x_filtered, N)
plt.subplot(3, 2, 1)
plt.plot(t,x3,'r')
plt.subplot(3, 2, 3)
plt.plot(t, nt,'b')
plt.subplot(3, 2, 5)
plt.plot(t, x_filtered,'g')
plt.subplot(3, 2, 2)
plt.plot(f,x_f1,'r')
plt.subplot(3, 2, 4)
plt.plot(f, x_f2,'b')
plt.subplot(3, 2, 6)
plt.plot(f, x_f3,'g')
plt.show()
