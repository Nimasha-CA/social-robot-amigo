import pyaudio
import wave
import numpy as np
import time
import matplotlib.pyplot as plt
import speech_recognition as sr
from scipy import signal
import math
import threading
import multiprocessing as ms
import os
import cv2

def get_audio_device():
    audio = pyaudio.PyAudio() # start pyaudio
    for ii in range(0,audio.get_device_count()):
        # print out device info
        d = audio.get_device_info_by_index(ii)
        if "snd_rpi_i2s_card:" in str(d['name']):
            return int(d['index'])
    return None

def is_speaking(data, THRESH):
    data = np.array(data)
    data = np.frombuffer(np.array(data), np.int16) / 100

    data_rms = np.sqrt(np.mean(np.square(data)))

    if data_rms > THRESH:
        print("Speaking..")
        return 1
    else:
        return 0

def save_audio(arr,audio,WAVE_OUTPUT_FILENAME,CHANNELS,FORMAT,RATE):
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(arr))
    waveFile.close()

def recognize(sr,r,WAVE_OUTPUT_FILENAME):
    with sr.AudioFile(WAVE_OUTPUT_FILENAME) as source:
        # listen for the data (load audio to memory)
        audio_data = r.record(source)
        # recognize (convert from speech to text)
        
        try:
            text = r.recognize_google(audio_data)
        except:
            return "?"
    return text


def lag_finder(y1, y2, sr):
    corr = signal.correlate(y2, y1, mode='full', method="direct")
    lags = signal.correlation_lags(len(y2), len(y1))
    corr /= np.max(corr)
    """
    fig, (ax_orig, ax_noise, ax_corr) = plt.subplots(3, 1, figsize=(4.8, 4.8))
    ax_orig.plot(y2)
    ax_corr.plot(lags, corr)
    ax_noise.plot(y1)
    plt.show()
    """
    lag = lags[np.argmax(corr)]

    return lag, lags, corr


def rootmean(m1, m2):
    mean1 = np.mean(m1)
    mean2 = np.mean(m2)
    #print(mean1 / mean2)
    
def plot_out(m1, m2, lags, corr):
    fig, axs = plt.subplots(3)
    axs[0].plot(m1)
    axs[1].plot(m2)
    axs[2].plot(lags, corr)
    plt.show()

def get_corresponding_mic_data(frames, start_index, stop_index):
    result = np.frombuffer(np.array(frames[start_index:stop_index]), np.int16)
    result = np.reshape(result, (int(result.shape[0] / 2), 2))
    mic_l_new, mic_r_new = result[:, 0], result[:, 1]

    size = len(mic_l_new)
    #print(size)
    start = int(np.ceil(size/2)-6000)
    stop = int(np.ceil(size/2)+6000)

    mic_l_new = mic_l_new[start:stop]
    mic_r_new = mic_r_new[start:stop]

    return mic_l_new / np.max(mic_l_new), mic_r_new / np.max(mic_r_new)

def find_angle(delay,mic_D,velocity):
    try:
        angle = math.acos(velocity*delay/mic_D)
    except:
        if delay<0:
            angle=-70
        if delay>0:
            angle=70
        return angle
    angle = angle * 180 / math.pi
    angle = (90-(1*angle))
    if delay<0:
        angle = math.acos(velocity * -delay / mic_D)
        angle = angle * 180 / math.pi
        angle = -(90-(1*angle))
    #print(angle)
    return angle

def pid_cal(cx,cy,fx,fy,dt,Ix_old,errorx_old,Iy_old,errory_old, P_GAIN, I_GAIN, D_GAIN):
    ex = cx-fx
    Px= ex*P_GAIN
    Ix = Ix_old+ (ex*I_GAIN*dt)
    Dx = (D_GAIN*(errorx_old-ex))/dt
    pidx = Px+Ix+Dx
    
    ey = cy-fy
    Py= ey*P_GAIN
    Iy = Iy_old+ (ey*I_GAIN*dt)
    Dy = (D_GAIN*(errory_old-ey))/dt
    pidy = Py+Iy+Dy
    
    Ix_old = Ix
    Iy_old =Iy
    errory_old=ey
    errorx_old = ex
    
    return pidx, pidy,errorx_old,Ix_old,errory_old,Iy_old


def find_correct_face(faces):
    x1, y1, x2, y2 =None, None, None, None
    areas =[]
    Face_cores=[]
    
    for face in faces:
        
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        areas.append((x2-x1)*(y2-y1))
        Face_cores.append([x1,y1,x2,y2])
    try:
        face = Face_cores[np.argmax(np.array(areas))]
    except:
        face = [None,None,None,None]
    
    return face


def detect_face(detector, frame):
                
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    #landmarks = predictor(gray, face)
    #x = landmarks.part(27).x 
    #y = landmarks.part(27).y
    [x1, y1, x2, y2] = find_correct_face(faces)
                
    return x1, y1, x2, y2


        
    