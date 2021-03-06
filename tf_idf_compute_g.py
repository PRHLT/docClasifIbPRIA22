from __future__ import print_function
from __future__ import division

import os, glob
import math
import sys
import argparse
import textdistance

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
parser.add_argument('--alpha', type=float, help='Filtering probability')
parser.add_argument('--classes', type=str, help='List of separated value of classes')
parser.add_argument('--data_path', type=str, help='Data path')
parser.add_argument('--data_path_te', type=str, help='Data path')
parser.add_argument('--data_path_prod', type=str, help='Data path')
parser.add_argument('--path_res_test', type=str, help='Data path', default="")
parser.add_argument('--path_res_prod', type=str, help='Data path', default="")
parser.add_argument('--path_res_train', type=str, help='Data path')
parser.add_argument('--IG_file', type=str, help='Data path')
parser.add_argument('--path_res_classes', type=str, help='Data path')
parser.add_argument('--all_files', type=_str_to_bool, help='Data path')
parser.add_argument('--prod', type=_str_to_bool, help='Data path')
parser.add_argument('--path_distances', type=str, help='Data path')
args = parser.parse_args()


def error(refs, hyps):
    errors = [textdistance.levenshtein.distance(r, h) for r, h in zip(refs, hyps)]
    num_errors = sum(errors)
    ref_length = sum(len(r) for r in refs)
    cer = float(num_errors) / float(ref_length)
    return num_errors, cer

def load_path_distances(p:str) -> dict:
    f = open(p, "r")
    lines = f.readlines()
    f.close()
    res = {}
    for line in lines:
        line = line.strip()
        word, words = line.split("|")
        word = word.strip()
        words = words.strip()
        words = words.split(" ")
        res[word] = words
    return res

def add_prob_pseudowords(f_vD:dict, f_g:dict, args:object ):
    f_vD_aux = f_vD.copy()
    for (doc,word), prob_f_vD in f_vD_aux.items():
        list_pseudo_words = f_g.get(word, [])
        p_accum = 0.0
        for p in list_pseudo_words: # TODO revisar
            p_accum += f_vD_aux.get((doc,p), 0)
        p_accum = args.alpha * p_accum
        f_vD[(doc, word)] += p_accum
        # if p_accum > 0:
        #     print(f"{(doc,word)} [{prob_f_vD}]  +  {p_accum} ")
    return f_vD

def add_prob_pseudowords2(f_vD:dict, f_g:dict, args:object, ig_list:list):
    f_vD_aux = f_vD.copy()
    docs = list(set([doc for (doc,word), prob_f_vD in f_vD_aux.items()]))
    for doc in docs:
        for word in ig_list:
            list_pseudo_words = f_g.get(word, [])
            p_accum = 0.0
            for p in list_pseudo_words:
                p_accum += f_vD_aux.get((doc,p), 0.0)
            p_accum = args.alpha * p_accum
            a = f_vD.get((doc, word), 0.0)
            f_vD[(doc, word)] = p_accum + a
            if a == 0.0 and p_accum > 0 and word == "FACULTAD":
                print(f"{(doc, word)} = 0 -> {p_accum}")
            # if p_accum > 0:
            #     print(f"{(doc,word)} [{prob_f_vD}]  +  {p_accum} ")
    exit()
    return f_vD

if __name__ == "__main__":

    #C??LCULO TF TRAIN:
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
    f_vD = {} #Diccionario key -> (doc,word) valor -> estimaci??n del numero de veces que aparece una palabra v en un doc.
    f_g = {} # Diccionario key,key -> distance con distancias precalculadas allvsall. Cuadratico
    f_g = load_path_distances(args.path_distances)
    tf_train = {} #Diccionario key -> (Doc, palabra), valor-> Tf
    f_tv = {} #(El numero de documentos que contiene la palabra), Diccionario key -> palabra, valor -> suma de sus prob max para cada doc
    lw_all = [] #Set con todas las palabras
    s = {} #Diccionario key -> doc, palabra, valor -> prob max de esa palabra en ese doc

    words = []
    with open(args.IG_file, 'r') as f:
        lines = f.readlines()
        f.close()
        for line in lines:
            line = line.strip() #Quitar espacios blancos inecesarios
            if len(line.split()) < 2:
                continue
            word = line.split()[0] #Split por espacios en blanco
            words.append(word)

    for path in directorio:
        ## ESCOGER PALABRAS CON PROBABILIDAD DE 0.5 PARA ARRIBA
        #print("Loading {}".format(path))
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
       
    # f_vD_aux = f_vD.copy()
    # for (doc,word), prob_f_vD in f_vD_aux.items():
    #     list_pseudo_words = f_g.get(word, [])
    #     p_accum = 0.0
    #     for p in list_pseudo_words:
    #         p_accum += f_vD_aux.get((doc,word), 0)
    #     p_accum = args.alpha * p_accum
    #     f_vD[(doc, word)] += p_accum
    #     if p_accum > 0:
    #         print(f"{(doc,word)} [{prob_f_vD}]  +  {p_accum} ")
    f_vD = add_prob_pseudowords2(f_vD, f_g, args, words )
    # exit()

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
    
    
    
    
    with open(args.path_res_train, 'w') as f:
        f.write('LegajoID ')
        for word in words:
            s = word + ' '
            f.write(s)
        f.write('clase')
        f.write('\n')
        for doc in m:
            d = doc + ' '
            f.write(d)
            n_clase = doc.split('.')[0].split('_')[-1].lower()
            # print(tf_train_Idf)
            for pal in words:
                n = str(tf_train_Idf.get((doc,pal), 0.0)) + ' '
                f.write(n)
            if n_clase not in clases:
                if not args.all_files:
                    raise Exception(f'Class {n_clase} not found')
                else:
                    n_clase = "other"
            f.write(f'{clases_dict.get(n_clase)}\n')
    
    with open(args.path_res_classes, "w") as f:
        for c, n_c in clases_dict.items():
            f.write(f'{c} {n_c}\n')
    
    if args.path_res_test != "":
        #C??LCULO TF TEST:
        print('CALCULANDO EL VECTOR TF.IDF DE TEST')
        # carpeta = '/Users/Juanjo Flores/OneDrive/Desktop/Clasificaci??n de im??genes con RNN/code/leave_one_out/all_samples_4950'
        carpeta = args.data_path_te
        # directorio  = os.listdir(carpeta)
        directorio = glob.glob(os.path.join(carpeta, "*idx"))
        m = [] #TODOS LOS DOCUMENTOS
        fD = {} #Diccionario, key -> Doc, valor -> Numero de palabras en X.
        f_vD = {} #Diccionario key -> (doc,word) valor -> estimaci??n del numero de veces que aparece una palabra v en un doc.
        tf_test = {} #Diccionario key -> (Doc, palabra), valor-> Tf
        f_tv = {} #(El numero de documentos que contiene la palabra), Diccionario key -> (doc,palabra), valor -> prob max
        s = {} #Diccionario key -> doc, palabra, valor -> prob max de esa palabra en ese doc
        for path in directorio:
            ## ESCOGER PALABRAS CON PROBABILIDAD DE 0.5 PARA ARRIBA
            #print("Loading {}".format(path))
            doc = path.split("/")[-1]
            f = open(path, "r")
            lines = f.readlines() 
            f.close()
            acum = 0 #Acumulador de las probabilidades de las palabras del doc 
            t_doc =  doc.split('_')[-1].split(".")[0].lower()
            if t_doc not in clases and not args.all_files: 
                continue
            for line in lines:
                line = line.strip() #Quitar espacios blancos inecesarios
                try:
                    word, prob_word = line.split() #Split por espacios en blanco
                except:
                    continue
                prob_word = float(prob_word)
                if(prob_word > prob):
                    #Calculo de fD -> Numero total de palabras en X
                    acum += prob_word
                    if f_vD.get((doc, word), 0) == 0:
                        f_vD[doc,word] = prob_word
                    else:
                        f_vD[doc,word] += prob_word              
            fD[doc] = acum
            m.append(doc)
        
        f_vD = add_prob_pseudowords(f_vD, f_g, args )
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

        with open(args.path_res_test, 'w') as f:
            f.write('LegajoID ')
            for word in words:
                s = word + ' '
                f.write(s)
            f.write('clase')
            f.write('\n')
            for doc in m:
                d = doc + ' '
                f.write(d)
                n_clase = doc.split('.')[0].split('_')[-1].lower()
                for pal in words:
                    if (doc,pal) in tf_test_Idf:
                        n = str(tf_test_Idf[doc,pal]) + ' '
                        f.write(n)
                if n_clase not in clases:
                    if not args.all_files:
                        raise Exception(f'Class {n_clase} not found')
                    else:
                        n_clase = "other"
                f.write(f'{clases_dict.get(n_clase)}\n')
    
    if args.path_res_prod != "":
        #C??LCULO TF TEST:
        print('CALCULANDO EL VECTOR TF.IDF DE PROD')
        # carpeta = '/Users/Juanjo Flores/OneDrive/Desktop/Clasificaci??n de im??genes con RNN/code/leave_one_out/all_samples_4950'
        carpeta = args.data_path_prod
        # directorio  = os.listdir(carpeta)
        directorio = glob.glob(os.path.join(carpeta, "*idx"))
        m = [] #TODOS LOS DOCUMENTOS
        fD = {} #Diccionario, key -> Doc, valor -> Numero de palabras en X.
        f_vD = {} #Diccionario key -> (doc,word) valor -> estimaci??n del numero de veces que aparece una palabra v en un doc.
        tf_test = {} #Diccionario key -> (Doc, palabra), valor-> Tf
        f_tv = {} #(El numero de documentos que contiene la palabra), Diccionario key -> (doc,palabra), valor -> prob max
        s = {} #Diccionario key -> doc, palabra, valor -> prob max de esa palabra en ese doc
        for path in directorio:
            ## ESCOGER PALABRAS CON PROBABILIDAD DE 0.5 PARA ARRIBA
            #print("Loading {}".format(path))
            doc = path.split("/")[-1]
            f = open(path, "r")
            lines = f.readlines() 
            f.close()
            acum = 0 #Acumulador de las probabilidades de las palabras del doc 
            t_doc =  doc.split('_')[-1].split(".")[0].lower()
            # if t_doc not in clases and not args.all_files: 
            #     continue
            for line in lines:
                line = line.strip() #Quitar espacios blancos inecesarios
                try:
                    word, prob_word = line.split() #Split por espacios en blanco
                except:
                    continue
                prob_word = float(prob_word)
                if(prob_word > prob):
                    #Calculo de fD -> Numero total de palabras en X
                    acum += prob_word
                    if f_vD.get((doc, word), 0) == 0:
                        f_vD[doc,word] = prob_word
                    else:
                        f_vD[doc,word] += prob_word              
            fD[doc] = acum
            m.append(doc)
        
        f_vD = add_prob_pseudowords(f_vD, f_g, args )
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

        with open(args.path_res_prod, 'w') as f:
            f.write('LegajoID ')
            for word in words:
                s = word + ' '
                f.write(s)
            f.write('clase')
            f.write('\n')
            for doc in m:
                d = doc + ' '
                f.write(d)
                n_clase = doc.split('.')[0].split('_')[-1].lower()
                for pal in words:
                    if (doc,pal) in tf_test_Idf:
                        n = str(tf_test_Idf[doc,pal]) + ' '
                        f.write(n)
                # if n_clase not in clases:
                #     if not args.all_files:
                #         raise Exception(f'Class {n_clase} not found')
                #     else:
                #         n_clase = "other"
                f.write(f'{clases_dict.get(n_clase, -1)}\n')