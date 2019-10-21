# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 11:15:42 2019

@author: Nadia
"""

# c'est un site ou on peut dessiner mais s'il me demande de dessiner par exemple un arbre et je dessine une autre forme prés
#d'une Ballon il la destingue et me dit c'est un ballon fiih dataset huge el des images que les gens on la dessiner.
#QuickDraw CNN with TensorFlow 2.0
#To start this noteboook you need to download the following files
#full_numpy_bitmap_airplane.npy
#full_numpy_bitmap_book.npy
#et on la sauvgarde dans un dossier qui 'est nommée Quick_draw_dataset
#Dependencies
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import os
#LOad the dataset
dataset_dir="quick_draw_dataset"
files=os.listdir(dataset_dir)
max_size_per_cl=1500
draw_class=[]
#evalute the size of the dataset
size=0
for name in files:
    draws=np.load(os.path.join(dataset_dir,name))
    draws=draws[:max_size_per_cl]#take only 10 000 draw
    size +=draws.shape[0]
images=np.zeros((size,28,28))
tragets=np.zeros((size,))
it=0
t=0
for name in files:
    #open each dataset and add the nex class
    draw_class.append(name.replace("full_numpy_bitmap_","").replace(".npy"))
    draws=np.load(os.path.join(dataset_dir,name))
    draws=draws[max_size_per_cl]#take only 10 000 draw
    #add images to the buffer
images[it:it+draws.shape[0]]=np.invert(draws.reshape(-1,28,28))
tragets[it:it+draws.shape[0]]=t
#iter
it+=draws.shape[0]
t+=1
#shuffle dataset
indexes=np.arange(size)
np.random.shuffle(indexes)
images=images[indexes]
tragets=targets[indexes]
images,images_valid,targets,targets_valid=train_test_split(images,targets,test_size=0.33)
print("image.shape",images.shape)
print("targets.shape",images.shape)
print("images_valid.shape",images_valid.shape)
print("targets_valid.shape",targets_valid.shape)
print(draw_class)    
#plot exemples of images:
w=10
h=10
fig=plt.figure(figsize=(8,8))
columns=4
rows=5
for i in range(1,columns*rows+1):
    index=np.random.randint(len(images))
    img=images[index]
    fig.add_subplot(rows,columns,i)
    plt.title(draw_class[int(targets[index])])
    plt.imshow(img,cmap="gray")
plt.show()
#Normalisation
print("Mean and std of images",images.mean(),images.std())
scaler=StandardScaler()
scaled_images=scaler_transform(images.reshape(-1,28,28))
print("Mean and std of scaled images",scaled_images.mean(),scaled_images.std())
scaled_images_scaled_images.reshape(-1,28,28,1)
scaled_images_valid=scaled_images_valid.reshape(-1,28,28,1)
#former notre datset pour entainer notre modéle
#create the datta set
#print(scaled_images.shape)
#print(scaled_images_valide.shape)
#hedhaa el parourir el normale el 3adiiyaaa benessba lel haajtoi biih en utilasnt le tf.dataset
train_dataset=tf.data.Dataset.from_tensor_slices(scaled_images)
valid_dataset=tf.data.Dataset.from_tensor_slices(scaled_images_valid)
#recuperation des données
#inter in dataset
for item in train_dataset:
    print(item.shape)
    break
#pour minimiser l'erreur on utilise des bash ce sont des blocs ou on traitent les données d'une façon
#pour arriver  a recuperer et rainer la totalités du dataset
    #nombre d'epoches iter in the dataset with a number of epoch and baatch size
print(scaled_images.shape)
epoch=1
batch_size=32
for batch_taining in train_dataset.repeat(epoch).batch(32):
    print(batch_training.shape)
    break
#create the training dataset
#iter in the dataset with a number of epoch and batch size
    epoch=1
    batch_size=32
for images_batch,target_batch in train_dataset.repeat(epoch).batch(batch_size):
#print(images_batch.shape,targets_batch.shape)
    print(images_batch[6].shape,target_batch[6])
#mazeel na9ess el resau d convolution fel video 17 
