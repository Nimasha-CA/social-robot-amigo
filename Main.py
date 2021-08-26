from help_modules import *
from motor_control import *
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
from nanpy import (ArduinoApi, SerialManager)
import dlib
import pygame

###configure tts####
def speak(audio_file_name):

    pygame.mixer.init(16500)
    pygame.mixer.music.load(audio_file_name)
    pygame.mixer.music.play()
    #while pygame.mixer.music.get_busy() == True:
        #continue


P_GAIN, I_GAIN, D_GAIN = 0.025,0.00000001,0.001

###Configuring face detector###
detector = dlib.get_frontal_face_detector()
###Configuring Video ###
vs = cv2.VideoCapture(0)
w,h=260,195
### Configuring Audio###

r = sr.Recognizer()
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 2048
SPEAKING_THRESH = 0.8
WAVE_OUTPUT_FILENAME = "file.wav"
DEVICE_INDEX = get_audio_device()
frame=np.zeros((480,360))
frames = [0] * 5000
frames_l = [0] * 5000
frames_r = [0] * 5000
times = [0] * 5000
data=0

audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    input_device_index=DEVICE_INDEX,
                    frames_per_buffer=CHUNK)

def record_audio():
    global CHUNK
    global data
    global times
    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        frames.pop(0)
        times.append(time.time())
        times.pop(0)

audio_capture = threading.Thread(target=record_audio)
audio_capture.start()
time.sleep(2)

###data,frame are always available for any calculation###
    
det=""
make_contact=0

def speech_rec():
    global det
    global data
    global SPEAKING_THRESH
    global make_contact
    t1 = time.time()
    start_time = t1
    stopped_time = t1
    while True:
        if is_speaking(data, SPEAKING_THRESH):

            if (time.time() - t1) > 1:
                start_time = time.time() - 1
            t1 = time.time()

        else:
            t2 = time.time()
            if (t2 - t1) > 1 and t1 > stopped_time:
                stopped_time = t2 + 0.5

                start_index = (np.abs(np.array(times) - start_time)).argmin()
                stop_index = (np.abs(np.array(times) - stopped_time)).argmin()
                mic_l, mic_r = get_corresponding_mic_data(frames, start_index, stop_index)
                save_audio(frames[start_index:stop_index],audio,WAVE_OUTPUT_FILENAME,CHANNELS,FORMAT,RATE)
                det = recognize(sr,r,WAVE_OUTPUT_FILENAME)
                print(det)
                if "hello Amigo" in det:
                    lag, lags, corr = lag_finder(mic_l, mic_r, 44100)
                    lag = lag * 1000000 / RATE#microseconds
                    angle = find_angle(lag/1000000, 9, 36750)
                    print("angle: ",angle)
                    move_neck(angle)
                    make_contact=1
                    speak("Audio/hello.wav")
               
                if "bye" in det:
                    speak("Audio/bye.wav")
                    make_contact=0
                    reset_motors()


speech_reco = threading.Thread(target=speech_rec)
speech_reco.start()


def get_video_info():
    _,frame = vs.read()
    frame = cv2.resize(frame, (w,h))
    x1, y1, x2, y2 = detect_face(detector, frame)
    try:
        frame = cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 1)
    except:
        frame=frame
    return frame,x1,y1,x2,y2



while True:
        not_detected = 0
        frame,x1,y1,x2,y2= get_video_info()
        if "hello Amigo" in det:
            t0=time.time()
            Ix_old,errorx_old,Iy_old,errory_old = 0,0,0,0
            while not_detected<100:
                frame,x1,y1,x2,y2= get_video_info()
                #cv2.imshow("vision",frame)
                key = cv2.waitKey(1)& 0xFF
                if key == ord("q"):
                        vs.release()
                        cv2.destroyAllWindows()
                        sys.exit()
                        break
                    
                    
                if not(x1==None) and make_contact==1:
                    fx=x1+(x2-x1)/2
                    fy=y1+(y2-y1)/2
                    t1=time.time()
                    dt=t1-t0
                    pidx, pidy,errorx_old,Ix_old,errory_old,Iy_old = pid_cal(w/2,h/2,fx,fy,dt,Ix_old,errorx_old,Iy_old,errory_old, P_GAIN, I_GAIN, D_GAIN)
                    change_servox(pidx)
                    change_servoy(-pidy)
                    t0=t1
                    not_detected=0
                    
                    if "bye" in det:
                        speak("Audio/bye.wav")
                        det=""
                        make_contact = 0
                        reset_motors()
                    if "company close" in det or "closing time" in det:
                        speak("Audio/close_time.wav")
                        det=""
                else:
                    not_detected=not_detected+1
            print("Face not detected..")
            make_contact=0
            reset_motors()
            det =""
        
        
        #cv2.imshow("vision",frame)
        key = cv2.waitKey(1)& 0xFF
        if key == ord("q"):
                vs.release()
                cv2.destroyAllWindows()
                sys.exit()
                break

    
    
    

    