import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import numpy as np
from load_data import *
# import cv2
import time
import os

# current = os.getcwd()
# # import wandb
# # wandb.init(project="IRL-Final", group = "random heatmaps" )
# # tar_f = "/data/atari-head/breakout/highscore/527_RZ_4153166_Jul-26-10-00-12.tar.bz2"
# # txt_f = "/data/atari-head/breakout/highscore/527_RZ_4153166_Jul-26-10-00-12.txt"

# tar_f = "/data/atari-head/venture/100_RZ_3592991_Aug-24-11-44-38.tar.bz2"
# txt_f = "/data/atari-head/venture/100_RZ_3592991_Aug-24-11-44-38.txt"

# # tar_f = "/data/atari-head/space_invaders/177_RZ_9004819_Jun-14-14-38-52.tar.bz2"
# # txt_f = "/data/atari-head/space_invaders/177_RZ_9004819_Jun-14-14-38-52.txt"
# d = Dataset(current+tar_f, current+txt_f)


# dfile = "/IL-CGL/BC-CGL/predicted_gaze_heatmaps/human_gaze_venture.npz"
# # image = "/data/atari-head/space_invaders/gaze_data_tmp/177_RZ_9004819_Jun-14-14-38-52/RZ_9004819_"

# print(d.train_lbl)
# d.load_predicted_gaze_heatmap(current+dfile)
# train = d.train_imgs
# print(train.shape)
# # images = mpimg.imread()
# print(d.train_GHmap.shape)
# print(d.train_imgs.shape)
# # plt.ion
# # plt.figure
# np.seed = 2023
imag = np.random.randint(0,5000,17)
# # print(d.train_GHmap[0])
# imag = range(0,17)
# print(d.train_GHmap[imag])
# print(imag)

# # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# for i,j in enumerate(imag):
#     if i == 0:
#         continue
#     # print(d.train_lbl[imag])
    
#     plt.subplot(4,4,i)
#     plt.axis('off')
#     plt.imshow(d.train_GHmap[j])
#     # plt.imshow(train[j],alpha = 0.5, cmap='Greys_r')
#     plt.tight_layout()
# plt.show()
#     # cv2.waitKey(200)
#############

import h5py
cwd = os.getcwd()

filename = cwd+'/ahead/data/processed/breakout.hdf5'

with h5py.File(filename,'r') as f:
    for i,j in enumerate(imag):
        if i == 0:
            continue
        # print(d.train_lbl[imag])
        
        plt.subplot(4,4,i)
        plt.axis('off')
       
        plt.tight_layout()
        plt.imshow(f['112_RZ_3866968_Aug-27-15-50-16']['gazes'][j,:,:])
        plt.imshow(f['112_RZ_3866968_Aug-27-15-50-16']['images'][j,0,:,:],alpha =0.5,cmap='Greys_r')
        
    plt.show()