import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import signal

garm_number=1
# measurement time 
t_ism=0.1# ms
# number of signals
tds_number=5000
# sampling frequency
w_d=10000# kHz
# signal lengh
sig_len=int(w_d*t_ism)
# max noise rate
noise_rate_max=40

garm0=[0 for t in range(sig_len)]
f = open('TNT_dataset.txt', 'w')
list_noise=[]
garm=[]

for i in range(tds_number):
    point_norm=0
    point_norm=random.random()*100
    list_noise.append(point_norm)

for i in range(tds_number):
    garm=[]
    garm_count=random.randint(0,garm_number)
    sig=[]
    # frequency shift
    sdvig=random.random()
    # the shift factor is selected depending on the temperature range
    # frequency uncertainty range
    delta_w=[sdvig*12+867, sdvig*27+852, sdvig*20+843, 
                sdvig*21+837, sdvig*14+840, sdvig*14+833]
    # amplitude uncertainty
    delta_a=1
    ampl=[(random.random()*delta_a+1), (random.random()*delta_a+1), (random.random()*delta_a+1), (random.random()*delta_a+1), (random.random()*delta_a+1), (random.random()*delta_a+1)]
    # attenuation coefficient uncertainty
    gamma=0.0005
    delta_gamma=random.random()*gamma+0.000001
    # initial phase uncertainty
    # identical initial phases
    delta_fi=random.random()*2*np.pi
    # different initial phases
    #delta_fi = [random.random()*2*np.pi for j in range(len(delta_w))]

    # signal generator
    for t in range(sig_len):
        sg=0
        for gn in range(len(delta_w)):
            sg+=ampl[gn]*np.sin(np.pi*2*delta_w[5-gn]/w_d*t+delta_fi)*np.exp(-(delta_gamma)*t)
        garm.append(sg)

    #plt.plot([j for j in range(sig_len)],garm)
    #plt.show()
    #ft=abs(np.fft.fft(garm))
    #plt.plot([j for j in range(len(ft))],ft)
    #plt.show()
    # noise
    noise_rate=random.random()*noise_rate_max
    noise=[np.random.normal(0,noise_rate) for j in range(sig_len)]
    # signal with noise or noise without signal
    sig=list(map(lambda a, b: a + b, noise, garm)) if garm_count==1 else list(map(lambda a, b: a + b, noise, garm0))
    print(i)

    f.write(str(sig)+'\t'+str(garm_count)+'\n')
f.close()
print('end')
