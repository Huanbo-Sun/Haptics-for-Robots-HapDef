# Single-Contact System Integration
- Package
``` Jupyter Notebook(Python2.7)
import numpy as np
import seaborn as sns
import tensorflow as tf
from collections import defaultdict
import os
import pickle
import time

import serial
import makerbot_driver
import threading
import pypot.dynamixel
import pyqtgraph as pg

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor

import matplotlib as mpl
from matplotlib import ticker, cm, gridspec
import matplotlib.pyplot as plt
# from matplotlib2tikz import save as tikz_save
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

%matplotlib notebook
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

def Data(sensorvalue,basis):
    Force = sensorvalue[10]
    SG = sensorvalue[:10] - basis[:10] +1700.0
    SG_std = (SG - x_mean)/(x_std+1e-16)
    knn_prediction = neigh.predict(SG_std[None,:])*y_std[1:]+y_mean[1:]
    svr_prediction = svr.predict(SG_std[None,:])*y_std[1:]+y_mean[1:]
    mlp_prediction = Single_Direct_sess.run(T_pred,feed_dict={T_in: SG_std[None,:]})*y_std[1:]+y_mean[1:]
    return(knn_prediction,svr_prediction,mlp_prediction)

def three_twos(positions):
    dp2 = np.zeros((positions.shape[0],2))
    dp2[:,1] = positions[:,1]
    R = np.sqrt(positions[:,0]**2 + positions[:,2]**2)[:,None]
    z = np.multiply(R,np.arctan2(positions[:,2],positions[:,0])[:,None])[:,None]
    dp2[:,0] = z.flatten()
    plt.scatter(dp2[:,0],dp2[:,1],s=18,color='red')
    return dp2

def three_twos_value(positions):
    dp2 = np.zeros((positions.shape[0],2))
    dp2[:,1] = positions[:,1]
    R = np.sqrt(positions[:,0]**2 + positions[:,2]**2)[:,None]
    z = np.multiply(R,np.arctan2(positions[:,2],positions[:,0])[:,None])[:,None]
    dp2[:,0] = z.flatten()
    return dp2

def three_twof(positions,marker,label,color,fig=plt):
    dp2 = np.zeros((positions.shape[0],2))
    dp2[:,1] = positions[:,1]
    R = (np.sqrt(positions[:,0]**2 + positions[:,2]**2)-2.0)[:,None]
    z = np.multiply(R,np.arctan2(positions[:,2],positions[:,0])[:,None])[:,None]
    dp2[:,0] = z.flatten()
    if marker=='^':
        fig.scatter(dp2[:,0],dp2[:,1],s=100,marker=marker,color=color,label=label)
    else:
        fig.scatter(dp2[:,0],dp2[:,1],s=100,color=color,label=label)
    return dp2

def three_twof_value(positions):
    dp2 = np.zeros((positions.shape[0],2))
    dp2[:,1] = positions[:,1]
    R = (np.sqrt(positions[:,0]**2 + positions[:,2]**2)-2.0)[:,None]
    z = np.multiply(R,np.arctan2(positions[:,2],positions[:,0])[:,None])[:,None]
    dp2[:,0] = z.flatten()
    return dp2

def sensormap(positions,si,col,fig=plt):
    dp2 = np.zeros((positions.shape[0],2))
    dp2[:,1] = positions[:,1]
    R = np.sqrt(positions[:,0]**2 + positions[:,2]**2)[:,None]
    z = np.multiply(R,np.arctan2(positions[:,2],positions[:,0])[:,None])[:,None]
    dp2[:,0] = z.flatten()
    fig.scatter(dp2[:,0],dp2[:,1],s=si,c=col)

def deformation_scatter(deformation, force_position,title,which):
    sensor_position = np.loadtxt("../Data/Bottleneck/Default/00_real_sensor_positions.txt",dtype=float)[:,1:]*1000.0
    sensor_map_nodal = np.loadtxt("../Data/Bottleneck/Default/00_sensor_map_nodal.txt",dtype=float)[:,1:]*1000.0
    t1 = three_twos(sensor_position)
    t2 = three_twos_value(sensor_map_nodal)
    if force_position.shape[1]<5:
        t3 = three_twof(force_position[0,1:][None,:])
        plt.scatter(t2[np.argmax(deformation),0],t2[np.argmax(deformation),1],color='orange')
    elif force_position.shape[1]<10:
        t3 = three_twof(force_position[0,1:4][None,:])
        t3 = three_twof(force_position[0,5:8][None,:])
    else:        
        t3 = three_twof(force_position[0,1:4][None,:])
        t3 = three_twof(force_position[0,5:8][None,:])
        t3 = three_twof(force_position[0,9:12][None,:])
    sensormap(sensor_map_nodal,1,deformation)
    plt.title(title,fontsize=13)
    plt.tick_params(direction='in')
    plt.yticks(rotation=90,fontsize=13)
    plt.xlabel("Arc length[mm]",fontsize=13)
    plt.ylabel("y[mm]",fontsize=13)
    

    if which == "a":
        cbaxes = fig.add_axes([0.426, 0.125, 0.01, 0.2]) 
        plt.colorbar(cax=cbaxes,ticks=[np.around(np.min(deformation),decimals=1)+0.1,(np.around(np.max(deformation),decimals=1)-np.around(np.min(deformation),decimals=1))/2.0,np.around(np.max(deformation),decimals=1)])
    else:
        cbaxes = fig.add_axes([0.849, 0.125, 0.01, 0.2])
        plt.colorbar(cax=cbaxes,ticks=[np.around(np.min(deformation),decimals=1)+0.1,(np.around(np.max(deformation),decimals=1)-np.around(np.min(deformation),decimals=1))/2.0,np.around(np.max(deformation),decimals=1)])
        
def sensormap_normalize_value(positions):
    dp2 = np.zeros((positions.shape[0],2))
    dp2[:,1] = (positions[:,1]-min(positions[:,1]))/(max(positions[:,1])-min(positions[:,1]))
    R = np.sqrt(positions[:,0]**2 + positions[:,2]**2)[:,None]
    z = np.multiply(R,np.arctan2(positions[:,2],positions[:,0])[:,None])[:,None]
    dp2[:,0] = z.flatten()
    dp2[:,0] = (dp2[:,0]-min(dp2[:,0]))/(max(dp2[:,0])-min(dp2[:,0]))
    return dp2

def heatmap(sensor_map_value,size,precision,basis):
    pixel_position = np.zeros((2*size**2,2))
    for i in range(size):
        for j in range(2*size):
            pixel_position[2*size*i+j,0] = 0.5/size*j
            pixel_position[2*size*i+j,1] = 1.0/size*i
    sensor_map_nodal = np.loadtxt("../Data/Bottleneck/Default/00_sensor_map_nodal.txt",dtype=float)
    sensor_map_nodal_normalize_positions = sensormap_normalize_value(sensor_map_nodal[:,1:]*1000)
    
    map_index = []
    pixel_index = []
    interpolated_map = np.zeros((pixel_position.shape[0],1))
    for i in range(pixel_position.shape[0]):
        a = np.linalg.norm(pixel_position[i,:]-sensor_map_nodal_normalize_positions,axis=1)
        if min(a)<precision:
            interpolated_map[i] = sensor_map_value[np.argmin(a)]
            map_index.append(np.argmin(a))
            pixel_index.append(i)
    interpolated_map = np.zeros((2*size**2))
    interpolated_map[pixel_index] = sensor_map_value[map_index]
#     plt.imshow(interpolated_map.reshape(size,2*size),cmap='gray',vmin=0,vmax=0.5)
    plt.imshow(interpolated_map.reshape(size,2*size),vmin=0,vmax=basis)
    plt.xlim([0,2*size-1])
    plt.ylim([0,size-1])
    plt.axis('off')
    
common_path = "../../../Data/Bottleneck/Raw_Data/"

def get_dataset(single_or_double, real_or_simu, data_type, train_val_test):
    template = '00_{}_{}_{}_{}.txt'
    filename = template.format(single_or_double, real_or_simu, data_type, train_val_test)
    return np.loadtxt(os.path.join(common_path, filename))

def nested_ddict():
    return defaultdict(nested_ddict)

def data_input(combinations):
    data = nested_ddict()
    for comb in combinations:
        for d_type in ('train', 'valid', 'test'):
            dataset = get_dataset(*comb, train_val_test=d_type)
            data[comb[0]][comb[1]][comb[2]][d_type] = dataset
    return data

def standardization(arr1,arr2):
    return (arr1 - np.mean(arr2,axis=0)) / (np.std(arr2,axis=0)+1e-16)

def get_batch(arr_in,arr_out,size):
    indices = np.random.randint(arr_in.shape[0],size = size)
    return arr_in[indices], arr_out[indices]
```
- Hardware Connection
``` Jupyter Notebook(Python2.7)
port = pypot.dynamixel.get_available_ports()
print('ports found',port)

motor = pypot.dynamixel.DxlIO(port[2],57600,use_sync_read=False)

testbed = makerbot_driver.s3g()
file = serial.Serial(port[1], 115200, timeout=1)
testbed.writer = makerbot_driver.Writer.StreamWriter(file,threading.Condition())

SGData = serial.Serial(port[0], 115200)
pypot.dynamixel.AbstractDxlIO??

#motor
motor.set_goal_position({1: 0})
#testbed: Direction:(x+:right)(y+:away)(z-:plate up) 
testbed.queue_extended_point([0, 0, 0, 0, 0], 800, 0, 0)
time.sleep(1)
testbed.queue_extended_point([10, 10, 10, 0, 0], 800, 0, 0)
time.sleep(2)
testbed.queue_extended_point([0, 0, 2000, 0, 0], 800, 0, 0)
time.sleep(2)
```
- Prediction Model Load
``` Jupyter Notebook(Python2.7)
filename = 'Model/knn_single_touch_model.sav'
neigh = pickle.load(open(filename, 'rb'))

svr = pickle.load(open("Model/svr_single_touch_model.sav",'rb'))

tf.reset_default_graph()
with tf.variable_scope("Single_Direct"):
    T_in = tf.placeholder(tf.float32,[None,10],name="Single_Direct_input")
    T_out = tf.placeholder(tf.float32,[None,5],name="Single_Direct_output")
    T_l1 = tf.layers.dense(T_in,500,activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.001))
    T_l2 = tf.layers.dense(T_l1,500,activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.001))
    T_l3 = tf.layers.dense(T_l2,500,activation=tf.nn.relu,kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.001))
    T_pred = tf.layers.dense(T_l3,5,activation=None,kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.001))
    
T_loss = tf.sqrt(tf.reduce_mean(tf.square(T_out-T_pred)))
T_training_summary = tf.summary.scalar("Single_Direct_training_error",T_loss)
T_validation_summary = tf.summary.scalar("Single_Direct_validation_error",T_loss)

T_train = tf.train.AdamOptimizer(5e-5).minimize(T_loss)

Single_Direct_sess = tf.Session()
Single_Direct_merged = tf.summary.merge_all()
Single_Direct_writer = tf.summary.FileWriter("Model/Single_Direct_training",Single_Direct_sess.graph)
Single_Direct_sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(Single_Direct_sess,"Model/Single_Direct.ckpt")
```
