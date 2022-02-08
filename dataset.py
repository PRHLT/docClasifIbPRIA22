from __future__ import print_function
from __future__ import division

import os, glob

import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
import random
# from torchvision import transforms
#import cv2
import logging
from sklearn.preprocessing import normalize


class myDataset(Dataset):

    def __init__(self, data, logger=None, transform=None, sequence=True):
        """
        data es el array "data" que proviene de la funcion load que te copio aqui abajo.
        Para cada fila la mete en otro array self.data
        """
        self.logger = logger or logging.getLogger(__name__)
        self.transform = transform
        self.data = []
        self.ids = [] #Nombre del fichero o documento
        for i in range(len(data)):
            self.ids.append(data[i][-1])
            self.data.append(
                (data[i][:-2], data[i][-2]) #Introducimos, las caracteristicas de un documento, y la clase
                                            #([array de caract, clase])   
                #(data[i][:-1], data[i][-1])
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #En el main se llamara para ir pillando 'items' según el bachsize
        data_row = self.data[idx]
        info, labels = data_row #info = features
        info = torch.tensor(info, dtype=torch.float) #pasamos a tipo tensor
        labels = torch.tensor(int(labels))
        sample = {
            "row": info, #caracteristicas
            "label": labels, #clase
            "id": self.ids[idx], #ID del documento
        }
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load(path, num_feats):
        #print("Loading {}".format(path))
        f = open(path, "r")
        lines = f.readlines()[1:] #QUITAMOS PRIMERA FILA
        f.close()
        data, labels = [], []
        #count_mal = 0
        fnames = []

        for line in lines:
            line = line.strip() #Quitar espacios blancos inecesarios
            
            fname = line.split()[0] #Split por espacios en blanco
            fnames.append(fname) #Cogemos el documento o fila
            feats = line.split()[1:] #Cogemos todo menos el nombre, el resto de la fila
            
            label = int(feats[-1]) #Etiqueta de clase, que esta en la utima columna
            labels.append(label)
            feats = feats[:-1]#Todo menos la clase
            feats = [float(f) for f in feats[:num_feats]] #num_feats -> numero de caracteristicas
            
            f = np.array(feats)
            f = f / f.sum() #Normalizamos entre la suma de todos los items
            f = list(f)
            data.append(feats)
        classes = set() #Que clases hay
        for i in range(len(labels)):
            data[i].append(labels[i])
            data[i].append(fnames[i])
            classes.add(labels[i])
        len_feats = len(data[0]) - 2
        
        return data, len_feats, len(classes)