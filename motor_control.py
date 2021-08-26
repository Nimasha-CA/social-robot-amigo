from nanpy import (ArduinoApi, SerialManager)
from nanpy import Servo
import time

c = SerialManager()
a=ArduinoApi(connection=c)

servo_xpin = 10
servox = Servo(servo_xpin)
xmin, xmax, xmid = 60, 125, 90
a.pinMode(servo_xpin,a.OUTPUT)

servo_ypin = 9
servoy = Servo(servo_ypin)
ymin, ymax, ymid = 65, 125, 90
a.pinMode(servo_ypin,a.OUTPUT)

servoy.write(ymid)
servox.write(xmid)

x_current = xmid
y_current = ymid



StepPins = [7,6,5,4]
nbStepsPerRev=2048
# Set all pins as output
for pin in StepPins:
        a.pinMode(pin,a.OUTPUT)
# Define some settings
WaitTime = 0.000000
# Define simple sequence
StepCount1 = 4
Seq1 = []
Seq1 = [i for i in range(0, StepCount1)]
Seq1[0] = [1,0,0,0]
Seq1[1] = [0,1,0,0]
Seq1[2] = [0,0,1,0]
Seq1[3] = [0,0,0,1]

Seq = Seq1
StepCount = StepCount1

def change_servox(ang):
    global x_current,y_current,ymin, ymax, ymid,xmin, xmax, xmid,a,servoy,servox
    new_ang = x_current+ang
    if new_ang>xmax:
        new_ang = xmax
    if new_ang<xmin:
        new_ang = xmin
    x_current = new_ang
    servox.write(new_ang)
    
def change_servoy(ang):
    global x_current,y_current,ymin, ymax, ymid,xmin, xmax, xmid,a,servoy,servox
    new_ang = y_current+ang
    if new_ang>ymax:
        new_ang = ymax
    if new_ang<ymin:
        new_ang = ymin
    y_current = new_ang
    servoy.write(new_ang)
    
cum_ang=0
def move_neck(angle):
    global cum_ang
    cum_ang = cum_ang +angle
    nb = int(angle*nbStepsPerRev/360)
    StepCounter = 0
    if nb<0: sign=-1
    else: sign=1
    nb=sign*nb*1 
    print("nbsteps {} and sign {}".format(nb,sign))
    for i in range(nb):
            for pin in range(4):
                    xpin = StepPins[pin]
                    if Seq[StepCounter][pin]!=0:
                            a.digitalWrite(xpin, a.HIGH)
                    else:
                            a.digitalWrite(xpin,a.LOW)
            StepCounter += sign
    # If we reach the end of the sequence
    # start again
            if (StepCounter==StepCount):
                    StepCounter = 0
            if (StepCounter<0):
                    StepCounter = StepCount-1
            # Wait before moving on
            #time.sleep(WaitTime)

def reset_motors():
    move_neck(-cum_ang)
    servoy.write(ymid)
    servox.write(xmid)
    
