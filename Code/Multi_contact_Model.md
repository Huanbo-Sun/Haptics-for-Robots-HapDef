# Multi Contact Model
## Package
### Import
``` Jupyter Notebook(Python2.7)
import numpy as np
import tensorflow as tf
import os
from sklearn.neighbors import NearestNeighbors
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import seaborn as sns
%matplotlib notebook

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
```
### DIY
``` Jupyter Notebook(Python2.7)
class ImportGraph():
    def __init__(self,loc):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            saver = tf.train.import_meta_graph(loc +'.meta',clear_devices=True)
            saver.restore(self.sess,loc)
            self.output = tf.get_collection('Output')
    def run(self, data):
        return self.sess.run(self.output,feed_dict={"Input:0":data})
```

## Data
``` Jupyter Notebook(Python2.7)
x_mean_transfernet = np.loadtxt("Multi_Contact_Demo/00_x_mean_transfernet",dtype=float)
x_std_transfernet = np.loadtxt("Multi_Contact_Demo/00_x_std_transfernet",dtype=float)
y_mean_transfernet = np.loadtxt("Multi_Contact_Demo/00_y_mean_transfernet",dtype=float)
y_std_transfernet = np.loadtxt("Multi_Contact_Demo/00_y_std_transfernet",dtype=float)

x_mean_reconstructnet = np.loadtxt("Multi_Contact_Demo/00_x_mean_reconstructnet",dtype=float)
x_std_reconstructnet = np.loadtxt("Multi_Contact_Demo/00_x_std_reconstructnet",dtype=float)
y_mean_reconstructnet = np.loadtxt("Multi_Contact_Demo/00_y_mean_reconstructnet",dtype=float)
y_std_reconstructnet = np.loadtxt("Multi_Contact_Demo/00_y_std_reconstructnet",dtype=float)

x_mean_sensitivitynet = np.loadtxt("Multi_Contact_Demo/00_x_mean_sensitivitynet",dtype=float)
x_std_sensitivitynet = np.loadtxt("Multi_Contact_Demo/00_x_std_sensitivitynet",dtype=float)
y_mean_sensitivitynet = np.loadtxt("Multi_Contact_Demo/00_y_mean_sensitivitynet",dtype=float)
y_std_sensitivitynet = np.loadtxt("Multi_Contact_Demo/00_y_std_sensitivitynet",dtype=float)
```

## Model
``` Jupyter Notebook(Python2.7)
tf.reset_default_graph()
TransferNet = ImportGraph('Multi_Contact_Demo/Model/Single_10240.ckpt')
ReconstructNet = ImportGraph('Multi_Contact_Demo/Model/Single_double_m2403211_MLP.ckpt')
SensitivityNet = ImportGraph('Multi_Contact_Demo/Model/Single_FDA.ckpt')
```
### Transfernet
``` Jupyter Notebook(Python2.7)
tf.reset_default_graph()
with tf.variable_scope("Single_10240_MLP"):
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)
    T_in = tf.placeholder(tf.float32,[None,10],name="Single_10240_MLP_input")
    T_out = tf.placeholder(tf.float32,[None,240],name="Single_10240_MLP_output")
    T_l1 = tf.layers.dense(T_in,250,activation=tf.nn.tanh,kernel_regularizer=regularizer)
    T_l2 = tf.layers.dense(T_l1,250,activation=tf.nn.tanh,kernel_regularizer=regularizer)
    T_l3 = tf.layers.dense(T_l2,250,activation=tf.nn.tanh,kernel_regularizer=regularizer)
    T_l4 = tf.layers.dense(T_l3,250,activation=tf.nn.tanh,kernel_regularizer=regularizer)
    T_l5 = tf.layers.dense(T_l4,250,activation=tf.nn.tanh,kernel_regularizer=regularizer)
    T_l6 = tf.layers.dense(T_l5,250,activation=tf.nn.tanh,kernel_regularizer=regularizer)
    T_pred = tf.layers.dense(T_l6,240,activation=None,kernel_regularizer=regularizer)

reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)

T_loss = tf.sqrt(tf.reduce_mean(tf.square(T_out-T_pred)))
T_loss_reg = T_loss+reg_term
T_training_summary = tf.summary.scalar("Single_10240_MLP_training_error",T_loss)

T_train = tf.train.AdamOptimizer(0.0001).minimize(T_loss_reg)

Single_Direct_sess = tf.Session()
Single_Direct_merged = tf.summary.merge_all()
Single_Direct_writer = tf.summary.FileWriter("Model/Single_10240_training",Single_Direct_sess.graph)
Single_Direct_sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(Single_Direct_sess,"Model/Single_10240.ckpt")
```

### Reconstructnet
``` Jupyter Notebook(Python2.7)
tf.reset_default_graph()
with tf.variable_scope("Single_double_m2403211_MLP"):
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
    T_in = tf.placeholder(tf.float32,[None,240],name="Single_double_m2403211_MLP_input")
    T_out = tf.placeholder(tf.float32,[None,3211],name="Single_double_m2403211_MLP_output")
    T_l1 = tf.layers.dense(T_in,600,activation=tf.nn.relu,kernel_regularizer=regularizer)
    T_l2 = tf.layers.dense(T_l1,1200,activation=tf.nn.relu,kernel_regularizer=regularizer)
    T_l3 = tf.layers.dense(T_l2,1800,activation=tf.nn.relu,kernel_regularizer=regularizer)
    T_pred = tf.layers.dense(T_l3,3211,activation=None,kernel_regularizer=regularizer)

reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)

T_loss = tf.sqrt(tf.reduce_mean(tf.square(T_out-T_pred)))
T_loss_reg = T_loss+reg_term
T_training_summary = tf.summary.scalar("Single_double_m2403211_MLP_training_error",T_loss)
T_validation_summary = tf.summary.scalar("Single_double_m2403211_MLP_validation_error",T_loss)

T_train = tf.train.AdamOptimizer(0.0001).minimize(T_loss_reg)

Single_Direct_sess = tf.Session()
Single_Direct_merged = tf.summary.merge_all()
Single_Direct_writer = tf.summary.FileWriter("Model/Single_double_m2403211_MLP_training",Single_Direct_sess.graph)
Single_Direct_sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.restore(Single_Direct_sess,"Model/Single_double_m2403211_MLP.ckpt")
```

### Sensitivitynet
``` Jupyter Notebook(Python2.7)
tf.reset_default_graph()
with tf.variable_scope("Single_FD_A"):
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.0006)
    T_in = tf.placeholder(tf.float32,[None,4],name="Single_Direct_FD_A_input")
    T_out = tf.placeholder(tf.float32,[None,1],name="Single_Direct_FD_A_output")
    T_l1 = tf.layers.dense(T_in,250,activation=tf.nn.relu,kernel_regularizer=regularizer)
    T_l2 = tf.layers.dense(T_l1,250,activation=tf.nn.relu,kernel_regularizer=regularizer)
    T_l3 = tf.layers.dense(T_l2,250,activation=tf.nn.relu,kernel_regularizer=regularizer)
    T_l4 = tf.layers.dense(T_l3,250,activation=tf.nn.relu,kernel_regularizer=regularizer)
    T_pred = tf.layers.dense(T_l4,1,activation=None,kernel_regularizer=regularizer)

reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
    
T_loss = tf.sqrt(tf.reduce_mean(tf.square(T_out-T_pred)))
T_loss_reg = T_loss+reg_term
T_training_summary = tf.summary.scalar("Single_FD_A_training_error",T_loss)
T_validation_summary = tf.summary.scalar("Single_FD_A_validation_error",T_loss)

T_train = tf.train.AdamOptimizer(0.0001).minimize(T_loss_reg)

Single_Direct_sess = tf.Session()
Single_Direct_merged = tf.summary.merge_all()
Single_Direct_writer = tf.summary.FileWriter("Model/Single_FD_A_training",Single_Direct_sess.graph)
Single_Direct_sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(Single_Direct_sess,"Model/Single_FDA.ckpt")
```
##
