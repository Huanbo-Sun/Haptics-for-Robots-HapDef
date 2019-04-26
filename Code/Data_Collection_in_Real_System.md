# Data Collection in Real System

``` jupyter notebook
# Packages:
import numpy as np
import matplotlib.pyplot as plt
import pypot.dynamixel
import serial
import makerbot_driver
import threading
import time

from drawnow import *
import pyqtgraph as pg

%matplotlib notebook

# Data:
## Force Positions
### raw force position in Euclidian space
raw_fp = np.loadtxt("../01_Force_Positions/01_Force_poistion_with_5mm_to_edges.txt",dtype=float) 
### force position transformation in testbed and Dynamxiel coordinates
position = np.zeros((raw_fp.shape)) 
position[:,0] = raw_fp[:,0]
position[:,1] = np.round((raw_fp[:,2]-0.15992)/0.106 * 10000.0) # x
position[:,2] = np.round((np.sqrt(raw_fp[:,1]**2+raw_fp[:,3]**2)- 0.033806)/0.005*2000.0)+1 # z
position[:,3] = 180.0 * np.sign(raw_fp[:,3]) - np.arctan2(raw_fp[:,3],raw_fp[:,1])/np.pi*180  # sita

# Serials Connection and Initialization
## Find all ports connected
port = pypot.dynamixel.get_available_ports()
print('ports found',port)
## Facilities connection
### motor connection
motor = pypot.dynamixel.DxlIO(port[2],57600,use_sync_read=False)
### testbed connection
testbed = makerbot_driver.s3g()
file = serial.Serial(port[1], 115200, timeout=1)
testbed.writer = makerbot_driver.Writer.StreamWriter(file,threading.Condition())
### arduinoboard connection
SGData = serial.Serial(port[0], 115200)
## Position Initialization
###motor
motor.set_goal_position({1: 0})
###testbed: Direction:(x+:right)(y+:away)(z-:plate up) 
testbed.queue_extended_point([0, 0, 0, 0, 0], 800, 0, 0)

# Data Collection
# set to start position
motor.set_goal_position({1: 0})
testbed.queue_extended_point([-470, 0, 2000, 0, 0], 800, 0, 0)
time.sleep(1)

for i in range(3000):
    # go to goal position
    motor.set_goal_position({1: position[i,3]})
    time.sleep(0.5)
    testbed.queue_extended_point([position[i,1], 0, 2000, 0, 0], 800, 0, 0)
    time.sleep(0.5)
    testbed.queue_extended_point([position[i,1], 0, position[i,2], 0, 0], 800, 0, 0)
    
    # get sensor data
    SG_P= np.zeros((1,16))
    SG_Value = np.zeros((11))
    for j in range(20):
        
        testbed.queue_extended_point([position[i,1], 0, position[i,2]-60*j, 0, 0], 800, 0, 0)
        
        SGData.flushInput()
        SGData.reset_input_buffer()
        SG = str(SGData.readline()).encode("utf-8")
        dataArray = SG.split(',')
        SG_Value = [ float(x) for x in dataArray]
        
        Te_Value = testbed.get_extended_position()[0][:3]
        Po = praw[i,0:4]
        
        SG_p = np.append(Po,np.append(Te_Value,SG_Value))
        SG_P = np.vstack((SG_P, SG_p))
        time.sleep(0.5)
        
    np.savetxt("../03_Collected_Data/Calibration/0101_Drifteffectmore/"+str(i)+".txt", SG_P, delimiter=' ')
                    
    # set to start position w.r.t. z
    testbed.queue_extended_point([position[i,1], 0, 2000, 0, 0], 800, 0, 0)
    time.sleep(2)
```
