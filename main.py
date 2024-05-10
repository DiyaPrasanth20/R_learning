import os  #makes it easier to traverse through diff file systems 
import cv2 #imports opencv to preprocess and load videos 
import tensorflow as tf #also used to use tensorflow data input pipeline (allows for preprocessing data)
#Data pipelines capture and deliver the information that's being used in a machine learning model
import numpy as np 
from typing import List 
from matplotlib import pyplot as plt #allows rendering of postprocessed data function
import imageio #can turn numpy array to gif 
import gdown 

'''
Building Data Loading Funtions: 
1) load videos 
2) preprocess annotations 
'''

url = 'https://drive.google.com/uc?id=1YlvpDLix3S-U8fd-gqRwPcWXAXm8JwjL'
output = 'data.zip'
gdown.download(url, output, quiet = False)
gdown.extractall('data.zip')


def load_video(path:str) -> List[float]: 

    cap = cv2.VideoCapture(path)  #creates a cv2 instance
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        #ret - boolean to see if frame read properly 
        ret,frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame) #transalate to grayscale so less data to preprocess 
        frames.append(frame[190:236, 80:220, :]) #statically isolates mouth 
    cap.release()

    #standardizing data 
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce.std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std




