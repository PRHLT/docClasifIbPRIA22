from __future__ import print_function
from __future__ import division

import os, glob
import pathlib
import math
import operator
import sys
import random
import logging

if __name__ == "__main__":

    #CÁLCULO TF TRAIN:
    print('CALCULANDO EL VECTOR TF.IDF DE TRAIN')
    carpeta ='/Users/Juanjo Flores/OneDrive/Desktop/Clasificación de imágenes con RNN/PRHLT/code/paper_final/JMBD_4950_5clases_loo'
    directorio  = os.listdir(carpeta)
    m = [] #TODOS LOS DOCUMENTOS
    prob = float(sys.argv[1]) #Probabilidad por la que filtramos 
    fD = {} #Diccionario, key -> Doc, valor -> Numero de palabras en X.
    f_vD = {} #Diccionario key -> (doc,word) valor -> estimación del numero de veces que aparece una palabra v en un doc.
    tf_train = {} #Diccionario key -> (Doc, palabra), valor-> Tf
    f_tv = {} #(El numero de documentos que contiene la palabra), Diccionario key -> palabra, valor -> suma de sus prob max para cada doc
    lw_all = [] #Set con todas las palabras
    s = {} #Diccionario key -> doc, palabra, valor -> prob max de esa palabra en ese doc
    for doc in directorio:
        ## ESCOGER PALABRAS CON PROBABILIDAD DE 0.5 PARA ARRIBA
        #print("Loading {}".format(path))
        path = carpeta + '/' + doc
        f = open(path, "r")
        lines = f.readlines() 
        f.close()
        acum = 0 #Acumulador de las probabilidades de las palabras del doc
        lw = [] #set con todas las palabras del doc
        t_doc =  doc.split('.')[0][0]
        if t_doc == 'p':   
            for line in lines:
                line = line.strip() #Quitar espacios blancos inecesarios
                word = line.split()[0] #Split por espacios en blanco
                prob_word = line.split()[1:] #Cogemos la probabilidad
                if(float(prob_word[0]) > prob):
                    #Calculo de fD -> Numero total de palabras en X
                    acum += float(prob_word[0])
                    if f_vD.get((doc, word), 0) == 0:
                        f_vD[doc,word] = float(prob_word[0])
                    else:
                        f_vD[doc,word] += float(prob_word[0])
                    lw_all.append(word)
                    #Calculo de f(tv)
                    if s.get((doc,word), 0) == 0: #Devuelve el valor de doc,word que es una prob si ya esta y si no un 0.
                        s[doc,word] = float(prob_word[0])
                    else:
                        s[doc,word] = max(s[doc,word], float(prob_word[0]))
            fD[doc] = acum
            m.append(doc)
            
    lw_all = set(lw_all)
    for word in lw_all:
        for doc in m:
            if (doc,word) in s:
                if f_tv.get(word,0) == 0:
                    f_tv[word] = float(s[doc,word])
                else:
                    f_tv[word] += float(s[doc,word])

    #Tf= f(v,D)/f(D)
    for word in lw_all:
        for doc in m:
            if (doc, word) in f_vD:
                tf_train[doc, word] = f_vD[doc, word]/fD[doc]
    
    #Calculo de Idf:
    idf = {} #Diccionario key -> palabra, valor -> idf
    for word in lw_all:
        idf[word] = math.log(len(m)/f_tv[word])

    #Calculo de Tf_train*Idf:
    tf_train_Idf = {} #Diccionario key -> tupla(doc, palabra), valor -> tf*idf
    for doc in m:
        for word in lw_all:
            if (doc,word) in tf_train:
                tf_train_Idf[doc,word] = tf_train[doc,word]*idf[word]
            else:
                tf_train_Idf[doc,word] = 0.0
    
    words = []
    with open('resultadosIG_4950_loo.txt', 'r') as f:
        lines = f.readlines()
        f.close()
        for line in lines:
            line = line.strip() #Quitar espacios blancos inecesarios
            word = line.split()[0] #Split por espacios en blanco
            words.append(word)
    
    
    with open('vector_tf_idf_train_4950_loo', 'w') as f:
        f.write('LegajoID ')
        for word in words:
            s = word + ' '
            f.write(s)
        f.write('clase')
        f.write('\n')
        for doc in m:
            d = doc + ' '
            f.write(d)
            n_clase = doc.split('.')[0].split('_')[-1]
            for pal in words:
                n = str(tf_train_Idf[doc,pal]) + ' '
                f.write(n)
            if(n_clase == 'P'):
                f.write('0')
            elif(n_clase == 'CP'):
                f.write('1')
            elif(n_clase == 'O'):
                f.write('2')
            elif(n_clase == 'A'):
                f.write('3')
            elif(n_clase =='T'):
                f.write('4')
            f.write('\n')
    
    #CÁLCULO TF TEST:
    print('CALCULANDO EL VECTOR TF.IDF DE TEST')
    carpeta ='/Users/Juanjo Flores/OneDrive/Desktop/Clasificación de imágenes con RNN/code/leave_one_out/all_samples_4950'
    directorio  = os.listdir(carpeta)
    m = [] #TODOS LOS DOCUMENTOS
    fD = {} #Diccionario, key -> Doc, valor -> Numero de palabras en X.
    f_vD = {} #Diccionario key -> (doc,word) valor -> estimación del numero de veces que aparece una palabra v en un doc.
    tf_test = {} #Diccionario key -> (Doc, palabra), valor-> Tf
    f_tv = {} #(El numero de documentos que contiene la palabra), Diccionario key -> (doc,palabra), valor -> prob max
    s = {} #Diccionario key -> doc, palabra, valor -> prob max de esa palabra en ese doc
    for doc in directorio:
        ## ESCOGER PALABRAS CON PROBABILIDAD DE 0.5 PARA ARRIBA
        #print("Loading {}".format(path))
        path = carpeta + '/' + doc
        f = open(path, "r")
        lines = f.readlines() 
        f.close()
        acum = 0 #Acumulador de las probabilidades de las palabras del doc 
        t_doc =  doc.split('.')[0][0]
        if t_doc == 'p':  
            for line in lines:
                line = line.strip() #Quitar espacios blancos inecesarios
                word = line.split()[0] #Split por espacios en blanco
                prob_word = line.split()[1:] #Cogemos la probabilidad
                if(float(prob_word[0]) > prob):
                    #Calculo de fD -> Numero total de palabras en X
                    acum += float(prob_word[0])
                    if f_vD.get((doc, word), 0) == 0:
                        f_vD[doc,word] = float(prob_word[0])
                    else:
                        f_vD[doc,word] += float(prob_word[0])              
            fD[doc] = acum
            m.append(doc)
    
    
    #Tf= f(v,D)/f(D)
    for word in lw_all:
        for doc in m:
            if (doc, word) in f_vD:
                tf_test[doc, word] = f_vD[doc, word]/fD[doc]
                
    #Calculo de Tf_test*Idf:
    tf_test_Idf = {} #Diccionario key -> tupla(doc, palabra), valor -> tf*idf
    for doc in m:
        for word in lw_all:
            if (doc,word) in tf_test and word in idf:
                tf_test_Idf[doc,word] = tf_test[doc,word]*idf[word]
            else:
                tf_test_Idf[doc,word] = 0.0
                
    words = []
    with open('resultadosIG_4950.txt', 'r') as f:
        lines = f.readlines()
        f.close()
        for line in lines:
            line = line.strip() #Quitar espacios blancos inecesarios
            word = line.split()[0] #Split por espacios en blanco
            words.append(word)
    

    with open('vector_tf_idf_test_4950', 'w') as f:
        f.write('LegajoID ')
        for word in words:
            s = word + ' '
            f.write(s)
        f.write('clase')
        f.write('\n')
        for doc in m:
            d = doc + ' '
            f.write(d)
            n_clase = doc.split('.')[0].split('_')[-1]
            for pal in words:
                if (doc,pal) in tf_test_Idf:
                    n = str(tf_test_Idf[doc,pal]) + ' '
                    f.write(n)
            if(n_clase == 'P'):
                f.write('0')
            elif(n_clase == 'CP'):
                f.write('1')
            elif(n_clase == 'O'):
                f.write('2')
            elif(n_clase == 'A'):
                f.write('3')
            elif(n_clase =='T'):
                f.write('4')
            elif(n_clase =='S'):
                f.write('5')
            f.write('\n')