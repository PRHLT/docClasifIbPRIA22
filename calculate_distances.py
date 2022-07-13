from __future__ import print_function
from __future__ import division

import os, glob
import math
import sys
import argparse
import textdistance
import tqdm

def _str_to_bool(data):
    """
    Nice way to handle bool flags:
    from: https://stackoverflow.com/a/43357954
    """
    if data.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif data.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

parser = argparse.ArgumentParser(description='Create the spans')
parser.add_argument('--prob', type=float, help='Filtering probability')
parser.add_argument('--C_max', type=float, help='Filtering probability')
parser.add_argument('--Nwords', type=int, help='Filtering probability')
parser.add_argument('--classes', type=str, help='List of separated value of classes')
parser.add_argument('--data_path', type=str, help='Data path')
parser.add_argument('--path_res_train', type=str, help='Data path')
parser.add_argument('--path_res_c', type=str, help='Data path')
parser.add_argument('--IG_file', type=str, help='Data path')
parser.add_argument('--path_res_classes', type=str, help='Data path')
parser.add_argument('--all_files', type=_str_to_bool, help='Data path')
parser.add_argument('--prod', type=_str_to_bool, help='Data path')
args = parser.parse_args()


def error(refs, hyps):
    errors = [textdistance.levenshtein.distance(r, h) for r, h in zip(refs, hyps)]
    num_errors = sum(errors)
    ref_length = sum(len(r) for r in refs)
    cer = float(num_errors) / float(ref_length)
    return num_errors, cer

if __name__ == "__main__":
    words = set()
    with open(args.IG_file, 'r') as f:
        lines = f.readlines()[:args.Nwords]
        f.close()
        for line in lines:
            line = line.strip() #Quitar espacios blancos inecesarios
            if len(line.split()) < 2:
                continue
            word = line.split()[0] #Split por espacios en blanco
            words.add(word)
    #CÁLCULO TF TRAIN:
    print('CALCULANDO EL VECTOR TF.IDF DE TRAIN')
    carpeta = args.data_path
    # directorio = os.listdir(carpeta)
    directorio = glob.glob(os.path.join(carpeta, "*idx"))
    clases = args.classes
    clases = clases.split(',')#Class list
    clases = [c.lower() for c in clases]
    if args.all_files:
        clases.append("other")
    clases_dict = {c:i for i,c in enumerate(clases)}
    m = [] #TODOS LOS DOCUMENTOS
    prob = args.prob #Probabilidad por la que filtramos 
    fD = {} #Diccionario, key -> Doc, valor -> Numero de palabras en X.
    f_vD = {} #Diccionario key -> (doc,word) valor -> estimación del numero de veces que aparece una palabra v en un doc.
    f_g = {} # Diccionario key,key -> distance con distancias precalculadas allvsall. Cuadratico
    C_max =args.C_max
    pair_compared = set()
    tf_train = {} #Diccionario key -> (Doc, palabra), valor-> Tf
    f_tv = {} #(El numero de documentos que contiene la palabra), Diccionario key -> palabra, valor -> suma de sus prob max para cada doc
    lw_all = [] #Set con todas las palabras
    s = {} #Diccionario key -> doc, palabra, valor -> prob max de esa palabra en ese doc
    directorio = tqdm.tqdm(directorio)
    for path in directorio:
        ## ESCOGER PALABRAS CON PROBABILIDAD DE 0.5 PARA ARRIBA
        #print("Loading {}".format(path))
        words_doc = set()
        words_doc_IG = set()
        doc = path.split("/")[-1]
        f = open(path, "r")
        lines = f.readlines() 
        f.close()
        acum = 0 #Acumulador de las probabilidades de las palabras del doc
        lw = [] #set con todas las palabras del doc
        t_doc =  doc.split('_')[-1].split(".")[0].lower()
        if t_doc not in clases and not args.all_files: 
            continue
        for line in lines:
            line = line.strip() #Quitar espacios blancos inecesarios
            try:
                word, prob_word = line.split() #Split por espacios en blanco
            except:
                continue
            prob_word = float(prob_word) #Cogemos la probabilidad
            if(prob_word > prob):
                words_doc.add(word)
                if word in words:
                    words_doc_IG.add(word)
                #Calculo de fD -> Numero total de palabras en X
                acum += prob_word
                if f_vD.get((doc, word), 0) == 0:
                    f_vD[doc,word] = prob_word
                else:
                    f_vD[doc,word] += prob_word
                lw_all.append(word)
                #Calculo de f(tv)
                if s.get((doc,word), 0) == 0: #Devuelve el valor de doc,word que es una prob si ya esta y si no un 0.
                    s[doc,word] = prob_word
                else:
                    s[doc,word] = max(s[doc,word], prob_word)
        fD[doc] = acum
        m.append(doc)
        ## DISTANCE
        
        words_doc = list(words_doc)
        # print(f"Distance - {len(words_doc_IG)} x {len(words_doc)} words in {doc}")
        for word1 in words_doc_IG:
            for word2 in words_doc:
                if (word1, word2) not in pair_compared:
                    errors, cer = error([word1], [word2])
                    if cer < C_max and cer != 0:
                        l = f_g.get(word1, [])
                        l.append(word2)
                        f_g[word1] = l
                pair_compared.add((word1, word2))
        # print(f"Compared with {len(pair_compared)} pairs - {len(f_g)} pairs")

                
        # exit()
    print("Saving F_g")
    f = open(args.path_res_c, "w")
    for word, l in f_g.items():
        l = " ".join(l)
        f.write(f"{word} | {l}\n")
    f.close()