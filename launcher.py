import sys
import argparse
#IMPORTAMOS MAIN PARA HACER -> LEAVE ONE OUT
from main_loo import main
import numpy as np

if __name__ == "__main__":
    if(len(sys.argv)<4):
        print("Error, number of iterations or system all information print argument, have not been passed!")
    else:
        iter = int(sys.argv[1])#Number of iterations
        all = int(sys.argv[2]) #If 1, system print all information
        max_epoch = int(sys.argv[3]) #Number of max_epochs
        lr = float(sys.argv[4]) #Learning Rate
        n = len(sys.argv[5]) #Num feats
        num_feats = sys.argv[5][1:n-1]
        num_feats = num_feats.split(',')
        
        wn = 99999 #Worst accuracy
        bnf = 0 #Best num_feats
        wnf = 99999 #Worst num_feats
        best_intervalRadius = 0
        
        for j in range(len(num_feats)): #Para cada Num_feats
            total = 0 #Suma de accuracys
            ir = 0 #suma de interval radius
            bn = 0 #Best accuracy
            confusion_matrix = np.zeros((9,9))
            for i in range(iter): #Realizamos i iteraciones
                #print('-'*50)
                #print("Iteration:",i+1)
                #print('Number of feats:',num_feats[j])  
                n,r,lf= main.trainEvaluation(all,lr,max_epoch,int(num_feats[j]),i) #n -> Total accuracy, r -> interval radius, cm -> confusion matrix
                if(bn < n):
                    bn = n
                    bnf = float(num_feats[j])
                    best_intervalRadius = r
                if(wn > n):
                    wn = n
                    wnf = float(num_feats[j])
                i += 1
                total += n
                ir += r
                # print("Muestras mal clasificadas y su correspondiente matriz de confusión:")
                # for x in range(0,len(le)):
                #     print('Muestra mal clasificada:',le[x])
                #     print(l_cm[x])
            
            media_nf = total/i
            print('*'*50)
            print('Para',lf,'Num feats, y despues de',iter,'inicializaciones, tenemos:')
            media_error = media_nf
            print('Media de error:',media_error)
            media_ir = ir/i
            print('Media de interval radius',media_ir)
            upInterval =  media_ir + media_nf
            downInterval = media_nf - media_ir
            print("Interval of Confidence: [%2d%%,%2d%%]" % (downInterval, upInterval))
            print('-'*50)
            j += 1      




        # print('*'*50)
        # print("Best Number of Features:",bnf)
        # print("Best accuracy: %2d%%" % bn)
        # upInterval = bn + best_intervalRadius
        # downInterval = bn - best_intervalRadius
        # print("Interval of Confidence: [%2d%%,%2d%%]" % (downInterval, upInterval))
        # print('*'*50)
        # print("Worst Number of Features:",wnf)
        # print("Worst accuracy: %2d%%" % wn)


        #AÑADIR COLUMNA CON CLASS ERROR A CONFUSION MATRIX (SUMAR LA FILA Y DIVIDIR ENTRE EL NUM DE MUESTRAS)
        #CALCULAR MLP-0
        #MAX MIN (0,100) INTERVALO CONFIANZA
        #python launcher.py 100 0 100 0.01 [8,16,32,64,128,256,512,1024,2048,4096,8192,16384] > resultsNF_MLP3.txt