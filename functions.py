import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import math
import time
from collections import Counter

class Profiler(object):
    def __enter__(self):
        self._startTime = time.time()
         
    def __exit__(self, type, value, traceback):
        print ("match many time: {:.3f} sec".format(time.time() - self._startTime))

def roc(mas_m,tr):
    ind_n0=[]
    ind_n1=[]
    print("show time")
    ind=0
    ctr=Counter(tr)
    c = Counter(mas_m)
    print(mas_m)
    print(c[0])
    for ok in range(c[0]):
        ind=mas_m.index(0,ind,len(mas_m))
        ind_n0.append(ind)
        ind=ind+1
    ind=0
    c = Counter(mas_m)
    print(mas_m)
    print(c[1])
    for ok in range(c[1]):
        ind=mas_m.index(1,ind,len(mas_m))
        ind_n1.append(ind)
        ind=ind+1
    print('yo:',ind_n1)
    print('yo:',ind_n0)
    x=0
    y=0
    xx=[]
    yy=[]
    j=0
    for i in range(len(ind_n1)):
        if tr[ind_n1[i]]==1:
            y=y+1/ctr[1]
        else:
            x=x+1/ctr[0]
        xx.append(x)
        yy.append(y)
    for i in range(len(ind_n0)):
        if tr[ind_n0[i]]==1:
            y=y+1/ctr[1]
        else:
            x=x+1/ctr[0]
        xx.append(x)
        yy.append(y)
    return xx,yy

def explose_det(for_match_many,ampl_avr,name,w_d,noise_rate,sig_len,sig_number,garm_number):
    xx=[]
    yy=[]
    garm0=[0 for t in range(sig_len)]
    p_m=0
    p_e=0
    tr=0
    p=0
    p_mmp=0
    match_many=len(for_match_many)
    print('mm:\t'+str(match_many))
    q=0
    count=0

    for i in range(sig_number):
        if count==i:
            print(count)
            count=count+100
        det=0
        det_m=0
        det_mmp=0
        det_e=0
        garm_count=random.randint(0,garm_number) if garm_number!=0 else 0
        sig=[]
        garm=[]
        sdvig=random.random()
        sd=[0.09, 0.19, 0.14, 0.15, 0.1, 0.1]
        wavr=[867, 852, 843, 837, 840, 833]

        #tnt
        garm_min=[]
        noise=[np.random.normal(0,noise_rate) for j in range(sig_len)]
        if (name=='tnt'):
            delta_w=[sdvig*2.4+867, sdvig*5.4+852, sdvig*4+843, 
                        sdvig*4.2+837, sdvig*2.8+840, sdvig*2.8+833]#+1
            delta_a=5
            ampl=[(random.random()*delta_a+1), (random.random()*delta_a+1), (random.random()*delta_a+1), (random.random()*delta_a+1), (random.random()*delta_a+1), (random.random()*delta_a+1)]
            gamma=0.0005#0#
            delta_gamma=random.random()*gamma+0.000001
            delta_fi=random.random()*2*np.pi
            for t in range(sig_len):
                sg=0
                for gn in range(len(delta_w)):
                    sg+=ampl[gn]*np.sin(np.pi*2*delta_w[5-gn]/w_d*t+delta_fi)*np.exp(-(delta_gamma)*t)
                garm.append(sg)
            for t in range(sig_len):
                sg=0
                for gn in range(len(delta_w)):
                    sg+=1*np.sin(np.pi*2*delta_w[5-gn]/w_d*t+delta_fi)*np.exp(-(4*gamma+delta_gamma)*t)
                garm_min.append(sg)

        #rdx
        if (name=='rdx'):
            delta_w=[sdvig*4+5235, sdvig*3+5190, sdvig*4.5+5030]
            #delta_w=[sdvig*200+5235]
            delta_a=random.random()*2
            ampl=[1+delta_a, 1+delta_a, 1+delta_a]
            gamma=0.0001
            delta_gamma=random.random()*gamma
            delta_fi=random.random()*2*np.pi
            for t in range(sig_len):
                sg=0
                for gn in range(len(delta_w)):
                    sg+=ampl[gn]*np.sin(np.pi*2*delta_w[gn]/w_d*t+delta_fi)*np.exp(-(4*gamma+delta_gamma)*t)
                garm.append(sg)
       

        #hmx
        if (name=='hmx'):
            delta_w=[sdvig*20+5290, sdvig*25+5050]
            delta_a=1#delta_a=random.random()*2#
            ampl=[1+delta_a, 1+delta_a]
            gamma=0.0001
            delta_gamma=random.random()*gamma
            delta_fi=random.random()*2*np.pi
            for t in range(sig_len):
                sg=0
                for gn in range(len(delta_w)):
                    sg+=ampl[gn]*np.sin(np.pi*2*delta_w[gn]/w_d*t+delta_fi)*np.exp(-(4*gamma+delta_gamma)*t)
                garm.append(sg)
        #noise=[0 for j in range(sig_len)]

        q+=sum(list(map(lambda a, b: a * b, garm, garm)))/(1+sum(list(map(lambda a, b: a * b, noise, noise))))
        sig=list(map(lambda a, b: a + b, noise, garm)) if garm_count==1 else list(map(lambda a, b: a + b, noise, garm0))
        xx.append(sig)
        ylist=[0 for cn in range(2)]
        ylist[int(garm_count)]=1
        yy.append(ylist)

        #match
        match=signal.correlate(garm, sig, 'same')
        sig_sig=signal.correlate(garm, garm, 'same')
        if (match[int(len(match)/2)]>1/2*sig_sig[int(len(sig_sig)/2)]):
            det=1
        if (det==garm_count):
            p=p+1

        #many match 
        cnt_mf=0
        for j in range(match_many):
                cnt_mf=cnt_mf+1         
                match_m=signal.correlate(for_match_many[j], sig, 'same')
                sig_sig_m=signal.correlate(for_match_many[j], for_match_many[j], 'same')
                if (match_m[int(len(match_m)/2)]>1/2*sig_sig_m[int(len(sig_sig_m)/2)]):
                    det_m=1
                    if (det_m==garm_count):
                        p_m=p_m+1
                        break
        if (det_m==0 and garm_count==0):
            p_m=p_m+1

        # maximum likelihood method
        '''A=np.zeros((9))
        def F(A):
            d=0.0
            t=0
            Fs=[]
            gamma=0.0001
            for f in sig:
                s_mmp=0
                for s in range(len(delta_w)):
                    s_mmp+=A[s+3]*np.sin(np.pi*2*(wavr[s]+sd[s]*A[0])/w_d*t+A[1])*np.exp(-(4*gamma+A[2])*t)
                t=t+1
                d+=(s_mmp-f)**2     
            return(d)
        A = [0,0,0, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5]
        bounds = [[0,1], [0,6.29], [0,0.00001],[1,6], [1,6], [1,6], [1,6], [1,6], [1,6],]
        res= differential_evolution(F,bounds,disp = True)
        sig_mmp=[]
        t=0
        for t in range(sig_len):
            s_mmp=0
            for s in 6:
                s_mmp+=res.x[s+3]*np.sin(np.pi*2*(wavr[s]+sd[s]*res.x[0])/w_d*t+res.x[1])*np.exp(-(4*gamma+res.x[2])*t)
            sig_mmp.append(s_mmp)    
          
        match_mmp=signal.correlate(sig_mmp, sig, 'same')
        sig_sig_mmp=signal.correlate(sig_mmp, sig_mmp, 'same') 
        if (match_mmp[int(len(match_mmp)/2)]>1/2*sig_sig_mmp[int(len(sig_sig_mmp)/2)]):
            det_mmp=1
        if (det_mmp==garm_count):
            p_mmp=p_mmp+1
        #A = [res_g.x[0],res_g.x[1],res_g.x[2]]
        #bnds = ((0.5, 1.5), (0.2, 0.5),(0,7),(0.004, 0.005))
        #res=minimize(F,A,method='powell',options={'xtol': 1e-8, 'disp': False})'''



        '''bounds_1 = [[A[0]-0.01 ,A[0]+0.01],[A[1]-0.2,A[1]+0.2],[0.004,0.005]]
        res= differential_evolution(F,bounds_1,disp = False)'''


        '''print('\n')
        print(res.x)
        print(delta_w,delta_fi,delta_g+0.005)'''

        '''sig_mmp=[np.sin(res.x[0]*t+res.x[1])*np.exp(-res.x[2]*t) for t in range(sig_len)]       
        match_mmp=signal.correlate(sig_mmp, sig, 'same')
        sig_sig_mmp=signal.correlate(sig_mmp, sig_mmp, 'same') 
        if (match_mmp[int(len(match_mmp)/2)]>1/2*sig_sig_mmp[int(len(sig_sig_mmp)/2)]):
            det_mmp=1
        if (det_mmp==garm_count):
            p_mmp=p_mmp+1'''

        # ED
        noise_e=[np.random.normal(0,noise_rate) for j in range(sig_len)]

        b, a = signal.butter(3, [2*830/w_d, 2*900/w_d],btype='bandpass')
        sig_f = signal.filtfilt(b, a, sig)
        noise_e_f=signal.filtfilt(b, a, noise_e)
        garm_min_f=signal.filtfilt(b, a, garm_min)
     
        E_n=sum(list(map(lambda a, b: a * b, noise_e_f, noise_e_f)))
        E_s=sum(list(map(lambda a, b: a * b, garm_min_f, garm_min_f)))
        if (sum(list(map(lambda a, b: a * b, sig_f, sig_f)))>(E_n+E_s/2)):#(E_n+E_s/2)
            det_e=1
        if (det_e==garm_count):
            p_e=p_e+1
 
    tr=garm_count
    p_mmp=p_mmp/sig_number
    p_m=p_m/sig_number
    p_e=p_e/sig_number
    p=p/sig_number
    q=q/sig_number
    #q=np.log10(q)*10
    return xx,yy,p_m,p,q,p_e



def match_discr_2_3(noise_rate,sig_len,sig_number,garm_number):
    xx=[]
    yy=[]
    garm0=[0 for t in range(sig_len)]
    p_match=0
    p_match_avr=0
    p_e=0
    p_m=0
    p_mmp=0
    p_fft=0
    match_many=500
    for_match_many=[]
    q=0
    count=0
    for i in range(sig_number):
        if count==i:
            print(count)
            count=count+100
        det=0
        det_m=0
        det_e=0
        det_mmp=0
        det_avr=0
        det_fft=1
        garm_count=random.randint(0,garm_number)
        sig=[]

        delta_w=random.random()*0.2+0.3
        delta_fi=random.random()*7
        delta_g=random.random()*0.001#

        delta_w2=delta_w+random.random()*0.05+0.05
        delta_fi2=random.random()*7
        delta_g2=random.random()*0.001#

        delta_w3=delta_w2+random.random()*0.05+0.05
        delta_fi3=random.random()*7
        delta_g3=random.random()*0.001#

        noise=[np.random.normal(0,noise_rate) for j in range(sig_len)]
        #noise=[0 for j in range(sig_len)]
        #noise_e=[np.random.normal(0,noise_rate) for j in range(sig_len)]

        garm0=[np.sin(delta_w*t+delta_fi)*np.exp(-(0.005-delta_g)*t) for t in range(sig_len)]
        garm1=[np.sin(delta_w*t+delta_fi)*np.exp(-(0.005-delta_g)*t)+np.sin(delta_w2*t+delta_fi2)*np.exp(-(0.005-delta_g2)*t) for t in range(sig_len)]
        garm2=[np.sin(delta_w*t+delta_fi)*np.exp(-(0.005-delta_g)*t)+np.sin(delta_w2*t+delta_fi2)*np.exp(-(0.005-delta_g2)*t)+np.sin(delta_w3*t+delta_fi3)*np.exp(-(0.005-delta_g3)*t) for t in range(sig_len)]
        #normirovka
        E1=sum(list(map(lambda a, b: a * b, garm0, garm0)))
        E2=sum(list(map(lambda a, b: a * b, garm1, garm1)))
        E3=sum(list(map(lambda a, b: a * b, garm2, garm2)))
        A=math.sqrt(E1/E2)
        B=math.sqrt(E1/E3)
        for g in range(len(garm1)):
            garm1[g]=A*garm1[g]
        for g in range(len(garm2)):
            garm2[g]=B*garm2[g]

        sig=list(map(lambda a, b: a + b, noise, garm1)) if garm_count==1 else list(map(lambda a, b: a + b, noise, garm2))
        xx.append(sig)
        q+=sum(list(map(lambda a, b: a * b, garm1, garm1)))/sum(list(map(lambda a, b: a * b, noise, noise)))
        ylist=[0 for cn in range(garm_number+1)]
        ylist[int(garm_count)]=1
        yy.append(ylist)
        #match
        match1=signal.correlate(garm1, sig, 'same')
        match2=signal.correlate(garm2, sig, 'same')
        if (match2[int(len(match2)/2)]<match1[int(len(match1)/2)]):
            det=1
        if (det==garm_count):
            p_match=p_match+1
        #spectrum
        spectrum=abs(np.fft.fft(sig))
        spec=[spectrum[m] for m in range(len(spectrum))]

        maxi1=max(spec[0:int(len(spec)/2)])
        ind=spec.index(maxi1)
        spec1=spec[0:(ind-5)]
        spec2=spec[(ind+5):int(len(spec)/2)]
        spec1.extend(spec2)

        maxi2=max(spec1[0:int(len(spec1))])
        ind1=spec1.index(maxi2)
        spec3=spec1[0:(ind1-5)]
        spec4=spec1[(ind1+5):int(len(spec1))]
        spec3.extend(spec4)

        if max(spec3)>1/2*maxi1:
            det_fft=0
        if (det_fft==garm_count):
            p_fft=p_fft+1
    p_match=p_match/sig_number
    p_match_avr=p_match_avr/sig_number
    p_m=p_m/sig_number
    p_e=p_e/sig_number
    p_mmp=p_mmp/sig_number
    p_fft=p_fft/sig_number
    q=q/sig_number
    return xx,yy,p_match,p_fft,q


def gauss_noise(noise_rate,sig_len):
    return [sum([random.random() for i in range(100)])/100*noise_rate-noise_rate/2 for j in range(sig_len)]

def get_batch(file_name,garm_cnt):
    fi_num=0
    with open(file_name,'r') as file:
        for line_for_cnt in file:
            fi_num=fi_num+1
            if(fi_num==1):
                line_for_cnt=line_for_cnt.replace('[','')
                line_for_cnt=line_for_cnt.replace(']','')
                line_for_cnt=line_for_cnt.replace('\t',', ')
                line_for_cnt=line_for_cnt.replace('\n','')
                line2=line_for_cnt.split(', ')
                sig_len=len(line2)-1
    p=0
    yy=[]
    ylist=[]
    with open(file_name,'r') as file:
        xx=np.zeros(([fi_num,sig_len]))
        for line in file:
            line=line.replace('[','')
            line=line.replace(']','')
            line=line.replace('\t',', ')
            line=line.replace('\n','')
            line2=line.split(', ')
            for i in range(len(line2)-2):
                xx[p][i]=float(line2[i])
            y=float(line2[len(line2)-1])
            ylist=[0 for cn in range(garm_cnt+1)]
            ylist[int(y)]=1
            yy.append(ylist)
            p=p+1
    return xx,yy,sig_len,fi_num

def get_batch_w(file_name,garm_cnt):
    fi_num=0
    with open(file_name,'r') as file:
        for line_for_cnt in file:
            fi_num=fi_num+1
            if(fi_num==1):
                line_for_cnt=line_for_cnt.replace('[','')
                line_for_cnt=line_for_cnt.replace(']','')
                line_for_cnt=line_for_cnt.replace('\t',', ')
                line_for_cnt=line_for_cnt.replace('\n','')
                line2=line_for_cnt.split(', ')
                sig_len=len(line2)-1
    p=0
    yy=[]
    ylist=[]
    with open(file_name,'r') as file:
        xx=np.zeros(([fi_num,sig_len]))
        for line in file:
            line=line.replace('[','')
            line=line.replace(']','')
            line=line.replace('\t',', ')
            line=line.replace('\n','')
            line2=line.split(', ')
            for i in range(len(line2)-2):
                xx[p][i]=float(line2[i])
            y=float(line2[len(line2)-1])
            ylist=[0 for cn in range(garm_cnt)]
            ylist[0]=y
            yy.append(ylist)
            p=p+1
    return xx,yy,sig_len,fi_num

def let_2_mass(xx,yy):
    l=int(len(yy)/2)
    return xx[0:l],yy[0:l],xx[l:2*l],yy[l:2*l]

def prorej(xx,q,win_len):
    c=0
    xx0=np.zeros(([len(xx),len(xx[0])+q*win_len]))
    for j in range(len(xx)):
        for i in range(len(xx0[0])):
            if (i<q*win_len/2):
                xx0[j][i]=0            
            elif (i>sig_len+q*win_len/2-1):
                xx0[j][i]=0
            else:
                xx0[j][i]=xx[j][i-int(q*win_len/2)]
    xxx=np.zeros(([len(xx0),sig_len*win_len]))
    for x in xx0:
        for i in range(len(xx[0])):
            for j in range(win_len):
                xxx[c][j+i*win_len]=x[i:(i+win_len*q)][::q][j]

        c=c+1
    return xxx

