
import scipy.signal as signal
import numpy as np
import scipy
import matplotlib.pyplot as plt
import random
import warnings
warnings.filterwarnings('ignore')

def artifial_ecg(rr,noise,num,a):
        fs = 250.0 #  rate
        pqrst = scipy.signal.wavelets.daub(num) 
        
        ecg = scipy.concatenate([scipy.signal.resample(pqrst, int(r*fs*a)) for r in rr])
        noise = np.random.normal(0, noise, len(ecg))
        ecg_res=ecg+noise
        return ecg_res  


class artificial_data():
    def __init__(self,n,arr,noise,arryhtmia=False):
        self.n=n
        self.arryhtmia=arryhtmia
        print('enter number of peaks, noise level, arrhythmia type: False, True(fo random) or A_Flu,A_Fib,Sup_Tach,PAC,VR')
        
    def create_rr(self,n,arr,noise,arryhtmia):
        #n-number of PQRST peaks
        #arr-number of arrhythmia peaks for rndom
        #noise-noise level
        #aaryhtmmia-False:normal beat, True:random,input:types
        rr=[0.5]*len(np.arange(n))
        if arryhtmia==True:
            rr_a=[]
            for i in range (len(np.arange(arr))):
                rr_a.append(np.random.random())
            rr=np.concatenate([rr,rr_a,rr])
        ecg_res=artifial_ecg(rr,noise,9,1)
                
        if arryhtmia=='A_Flu':
            rr_a=[0.5,0.4,0.5]*arr
            normal=artifial_ecg(rr,noise,9,1)
            arrhythmia=artifial_ecg(rr_a,0.05,5,1)
            ecg_res=np.concatenate([normal,arrhythmia,normal,arrhythmia])
        
        if arryhtmia=='A_Fib':
            rr_a=[0.9,0.4,0.9,0.5,0.2]*arr
            normal=artifial_ecg(rr,noise,9,1)
            arrhythmia=artifial_ecg(rr_a,0.05,5,1)
            ecg_res=np.concatenate([normal,arrhythmia,normal,arrhythmia])
            
        if arryhtmia=='Sup_Tach':
            rr_a=[0.5,0.5,0.5,0.5,0.2]*arr
            normal=artifial_ecg(rr,noise,9,1)
            arrhythmia=artifial_ecg(rr_a,0.03,4,1)
            ecg_res=np.concatenate([normal,arrhythmia,normal,arrhythmia])
        if arryhtmia=='PAC':
            rr_a=[0.2,0.3,0.5,0.3]*arr
            normal=artifial_ecg(rr,noise,9,1)
            arrhythmia=artifial_ecg(rr_a,noise,9,1)
            ecg_res=np.concatenate([normal,arrhythmia,normal,arrhythmia])
        if arryhtmia=='VR':
            rr_a=[1]*arr
            normal=artifial_ecg(rr,noise,9,1)
            arrhythmia=artifial_ecg(rr_a,0.03,2,1)
            ecg_res=np.concatenate([normal,arrhythmia,normal,arrhythmia,normal,arrhythmia])
                       
   
        return ecg_res


