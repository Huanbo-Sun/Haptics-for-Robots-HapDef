# Data Collection in Real System
## Data Collection
``` jupyter notebook
# Packages:
import numpy as np
import matplotlib.pyplot as plt
import pypot.dynamixel
import serial
import makerbot_driver
import threading
import time
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
## Data in Real Time visualization
``` jupyter notebook
SG1,SG2,SG3,SG4,SG6,SG7,SG8,SG9,SG10,FT = [],[],[],[],[],[],[],[],[],[]
cnt = 0
timepoints = []
view_time = 40
duration = 240000

fig = plt.figure(figsize=(6,5))
fig.suptitle('Live data', fontsize='18', fontweight='bold')
plt.xlabel('Time[s]', fontsize='14', fontstyle='italic')
plt.ylabel('Signal[V]', fontsize='14', fontstyle='italic')
plt.axes().grid(True)
line1, = plt.plot(SG1,marker='',label='SG1')
line2, = plt.plot(SG2,marker='',label='SG2')
line3, = plt.plot(SG3,marker='',label='SG3')
line4, = plt.plot(SG4,marker='',label='SG4')
line6, = plt.plot(SG6,marker='',label='SG6')
line7, = plt.plot(SG7,marker='',label='SG7')
line8, = plt.plot(SG8,marker='',label='SG8')
line9, = plt.plot(SG9,marker='',label='SG9')
line10, = plt.plot(SG10,marker='',label='SG10')
lineFT, = plt.plot(FT,marker='',label='FT')
plt.xlim([0,view_time])
plt.ylim([-0.1,5.1])
plt.show()

# arduinoData = serial.Serial('/dev/ttyACM1', 115200)
plt.ion()
start_time = time()
SGData.flushInput()
run = True
while run:
    SGData.reset_input_buffer()
    arduinoString = str(SGData.readline()).encode("utf-8")
    dataArray = arduinoString.split(',')
    try:
        SG1.append(float(dataArray[0])*5.0/4096.0)
        SG2.append(float(dataArray[1])*5.0/4096.0)
        SG3.append(float(dataArray[2])*5.0/4096.0)
        SG4.append(float(dataArray[3])*5.0/4096.0)
        SG6.append(float(dataArray[4])*5.0/4096.0)
        SG7.append(float(dataArray[5])*5.0/4096.0)
        SG8.append(float(dataArray[6])*5.0/4096.0)
        SG9.append(float(dataArray[7])*5.0/4096.0)
        SG10.append(float(dataArray[8])*5.0/4096.0)
        FT.append((float(dataArray[9]))*10.0/4096.0)
        timepoints.append(time()-start_time)
        current_time = timepoints[-1]
        line1.set_xdata(timepoints)
        line1.set_ydata(SG1)
        line2.set_xdata(timepoints)
        line2.set_ydata(SG2)
        line3.set_xdata(timepoints)
        line3.set_ydata(SG3)
        line4.set_xdata(timepoints)
        line4.set_ydata(SG4)
        line6.set_xdata(timepoints)
        line6.set_ydata(SG6)
        line7.set_xdata(timepoints)
        line7.set_ydata(SG7)
        line8.set_xdata(timepoints)
        line8.set_ydata(SG8)
        line9.set_xdata(timepoints)
        line9.set_ydata(SG9)
        line10.set_xdata(timepoints)
        line10.set_ydata(SG10)
        lineFT.set_xdata(timepoints)
        lineFT.set_ydata(FT)
        
        if current_time > view_time:
            plt.xlim([current_time-view_time,current_time])
        if timepoints[-1] > duration: run=False
    
    except: pass
    
    fig.canvas.draw()
arduinoData.close()
```
