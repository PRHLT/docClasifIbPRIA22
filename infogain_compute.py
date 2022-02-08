from __future__ import print_function
from __future__ import division

import os, glob
import pathlib
import math
import operator
import random
import logging
import sys
import argparse
import numpy as np


if __name__ == "__main__":
    
    if(len(sys.argv)<2):
        print("Error, number of iterations or system all information print argument, have not been passed!")
    else:
        prob = float(sys.argv[1]) #Filtering probability
        clases = sys.argv[2][1:len(sys.argv[2])-1] 
        clases = clases.split(',')#Class list
        print("CARGANDO TODOS LOS ARCHIVOS Y SUS PALABRAS")
        #carpeta = location of JMBD files
        carpeta ='/Users/Juanjo Flores/OneDrive/Desktop/Clasificación de imágenes con RNN/PRHLT/code/paper_final/JMBD_4949_5clases_loo'
        directorio  = os.listdir(carpeta)
        m = [] #All docs list
        s = {} #Dictionary key -> doc, word, valor -> prob max
        lw_all_total = [] #Set of words
        for doc in directorio:
            path = carpeta + '/' + doc
            f = open(path, "r")
            lines = f.readlines() 
            f.close()  
            t_doc =  doc.split('.')[0][0]
            if t_doc == 'p':
                m.append(doc)
                for line in lines:
                    line = line.strip() #Quitar espacios blancos inecesarios
                    word = line.split()[0] #Split por espacios en blanco
                    if len(word) < 3: continue
                    prob_word = line.split()[1:] #Cogemos la probabilidad
                    if(float(prob_word[0]) > prob):#Posibilidad de filtrado
                        lw_all_total.append(word)
                        #Calculo de f(tv)
                        if s.get((doc,word), 0) == 0: #Devuelve el valor de doc,word que es una prob si ya esta y si no un 0.
                            s[doc,word] = float(prob_word[0])
                        else:
                            s[doc,word] = max(s[doc,word], float(prob_word[0]))
        lw_all_total = set(lw_all_total)
        f_tv = {} #Diccionario key -> word, valor -> estimación del numero de veces que aparece una palabra v en un doc.
        for word in lw_all_total:
            for doc in m:
                if (doc,word) in s:
                    if f_tv.get(word,0) == 0:
                        f_tv[word] = float(s[doc,word])
                    else:
                        f_tv[word] += float(s[doc,word])
                        
        ###Finalizamos cálculo de P(tv) y P(no_tv):
        p_tv = {} #Diccionario key -> palabra, valor -> prob que algun doc contenga v
        p_notv = {}#Diccionario key -> palabra, valor -> prob que algun doc NO contenga v
        for word in f_tv:
            p_tv[word] = f_tv[word] / len(m)
            p_notv[word] = 1 - p_tv[word]

        #1º.Para cada clase que tengamos en la lista, calculamos f(c,tv)
        f_c_tv = {} #Diccionario key -> [clase,palabra], valor ->  valor -> nº de docs de la clase
        p_c_tv = {} #Diccionario key ->(clase,palabra), valor->prob. de que algun doc sea de la clase c, estando v
        p_c_notv = {} #Diccionario key ->(clase,palabra), valor->prob. de que algun doc NO sea de la clase c, estando v
        s_c = {} #Diccionario key -> doc, palabra, valor -> prob max de esa palabra en ese doc
        v = [] #Set de palabras
        m_c = {} #Diccionario con el número de docs por clase
        
        for c in clases:
            r = 0           
            for doc in m:
                clas_doc =  doc.split('.')[0].split('_')[-1]
                if clas_doc == c:
                    path = carpeta + '/' + doc
                    f = open(path, "r")
                    lines = f.readlines() 
                    f.close()
                    r += 1 
                    for line in lines:
                        line = line.strip() #Quitar espacios blancos inecesarios
                        word = line.split()[0] #Split por espacios en blanco
                        if len(word) < 3: continue
                        prob_word = line.split()[1:] #Cogemos la probabilidad
                        if(float(prob_word[0]) > prob):
                            v.append(word)
                            if s_c.get((c,doc,word), 0) == 0: #Devuelve el valor de doc,word que es una prob si ya esta y si no un 0.
                                s_c[c,doc,word] = float(prob_word[0])
                            else:
                                s_c[c,doc,word] = max(s_c[c,doc,word], float(prob_word[0]))
            m_c[c] = r           
        v = set(v)
        for c in clases:
            for doc in m:
                for word in v:
                    if (c,doc,word) in s_c:   
                        if f_c_tv.get((c,word),0) == 0:
                            f_c_tv[c,word] = float(s_c[c,doc,word])
                        else:
                            f_c_tv[c,word] += float(s_c[c,doc,word])
            
        #2º.Para cada clase calculamos P(c|tv) y P(c_not|tv):
        p_c = {}
        for c in clases: 
            p_c[c] = m_c[c]/(len(m)-1) 
            for word in v:
                if (c,word) in f_c_tv: 
                    p_c_tv[c,word] = f_c_tv[c,word] / f_tv[word]

                    if len(m) != f_tv[word]:   
                        p_c_notv[c,word] = (m_c[c] - f_c_tv[c,word]) / (len(m) - f_tv[word])
                    else:
                        p_c_notv[c,word] = m_c[c]/(len(m))
                else:
                    p_c_notv[c,word] = m_c[c]/(len(m)-f_tv[word])
        
        print('CALCULANDO EL INFORMATION GAIN DE CADA PALABRA')
        ig = {} #IG de un palabra
        for word in lw_all_total:
            ig[word] = 0
            r1 = 0
            r2 = 0
            r3 = 0
            for c in clases:
                r1 += p_c[c]*math.log(p_c[c])
                if (c,word) in p_c_tv and p_c_tv[c,word] != 0.0:
                    r2 += p_c_tv[c,word]*math.log(p_c_tv[c,word])
                else:
                    r2 += 0.0
                if (c,word) in p_c_notv and p_c_notv[c,word] != 0.0:
                    r3 += p_c_notv[c,word]*math.log(p_c_notv[c,word])
                else:
                    r3 += 0.0
            ig[word] += -r1 + (p_tv[word] * r2) + (p_notv[word]*r3)
        
        print("ORDENANDO EL DICCIONARIO")
        #Ordenamos el IG de mayor a menor:
        infGain_sort = sorted(ig.items(), key = operator.itemgetter(1), reverse=True)
        i = 0
        print("SACANDO LOS RESULTADOS AL FICHERO EXTERNO")
        with open('resultadosIG_4950.txt', 'w') as f:
            for word in infGain_sort:
                if i <= 32768:
                    w0 = str(word[0])
                    w1 = str(word[1])
                    s = str(w0 + ' ' + w1 + '\n')
                    f.write(s)
                else: break
                i += 1
        #python .\infogain_compute.py 0 [P,CP,O,A,T]
