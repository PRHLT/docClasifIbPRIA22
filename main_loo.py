import torch
import os
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import sklearn 
from math import sqrt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from pytorch_lightning.metrics import ConfusionMatrix
from dataset import myDataset
from model import Net


class main():
    def trainEvaluation(all,learning_rate,me,nf,iter):
        
        np.random.seed(iter)
        torch.manual_seed(1)        
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        max_epochs = me
        num_feats = nf #1024
        batch_size = 3
        n_classes = 5
        layers = [128,128,128]
       
        gpu = 0
        use_gpu = True
        device = torch.device('cuda:{}'.format(gpu) if use_gpu else 'cpu')
        #Data Loading
        data_total, len_feats, n_classes = myDataset.load('./vector_tf_idf_4950_loo', num_feats)
        dataset_total = myDataset(data_total)
        #data_total = torch.utils.data.DataLoader(dataset_total, batch_size=1,shuffle=True, num_workers=0)           
        #data_tr = torch.utils.data.DataLoader(dataset_total, batch_size=1,shuffle=True, num_workers=0)

        #Entrenamiento y evaluación con Leave-One-Out:
        
        count = 0
        errores = 0
        list_errores = []
        list_error_confmats = []
        #print(len(dataset_total))
        while count < len(dataset_total):
            model = Net(len_feats, layers, n_classes = n_classes) #CREAMOS EL MODELO
            
            if(all==2):
                print(model)
                print(f"Número de parámetros:{model.num_params}")
                
            model.to(device)
            criterion = torch.nn.NLLLoss() #Loss function Cross Entropy 
            optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate) #Optimizer -> Stochastic Gradient Descendent with learning rate = 0.01
            best_train_loss = 99999
            best_epoch = 0
            best_model = None
            #print(dataset_total[count]['id'])
            data_te = []     
            i = 0
            data_tr = []
            for sample_tr in dataset_total:
                if i != count:
                    data_tr.append(dataset_total[i])
                else:
                    data_te.append(dataset_total[i])
                i+=1
            # print(len(data_tr))
            # print(len(data_te))
            # for sample in data_tr:
            #     if sample['id'] == data_te[0]['id']:
            #         print('ERROR')
            data_te = torch.utils.data.DataLoader(data_te, batch_size=batch_size,shuffle=False, num_workers=0)
            data_tr = torch.utils.data.DataLoader(data_tr, batch_size=batch_size,shuffle=True, num_workers=0)
            #print(len(dataset_total))
            for epoch in range(max_epochs):
                train_loss = 0.0
                contador = 1 
                model.train()#Model in train mode
                for batch,sample in enumerate(data_tr):
                        
                        c = sample['row'].to(device) #Obtenemos las caracteristicas y las pasamos a device
                        
                        l = sample['label'].to(device) #Obtenemos la etiqueta de clase y la pasamos a device
                        
                        
                        optimizer.zero_grad()
                        
                        outputs = model(c) #SALIDA DE FORWARD -> USAR LUEGO EN EVALUACIÓN

                        #Compute Loss

                        loss = criterion(outputs, l)
                        
                        loss.backward() #Aplicamos el algoritmo de backpropagation
                        optimizer.step()
                        
                        train_loss += loss.item() / l.data.size()[0]
                        contador += 1
                    
                train_loss = train_loss / contador #Normalizamos
                # if(all==1):
                #     print('Epoch {}: train loss: {}'.format(epoch, train_loss))

                if train_loss <= best_train_loss: 
                    best_train_loss = train_loss
                    best_epoch = epoch
                    best_model = model.state_dict()

            # if(all==1):
            #     print('Best Epoch {}: Best train loss: {}'.format(best_epoch, best_train_loss))    
                   
            
            #Cargamos best_model
            #print(f'Loading Best model, from epoch:{best_epoch}')
            model.load_state_dict(best_model)
        
            # #EVALUATION
            model.eval() #Model in evaluation mode
            test_loss = 0.0
            class_correct = list(0. for i in range(n_classes)) #Clases que el sistema ha acertado
            class_total = list(0. for i in range(n_classes)) #Todas las clases
            correct_array = np.zeros((n_classes,n_classes))
            hipotesis_array = np.zeros((n_classes,n_classes))
            contador = 0
            confmat = np.zeros((n_classes,n_classes))
            for batch, sample in enumerate(data_te):

                c = sample['row'].to(device)
                l = sample['label'].to(device)

                output = model(c)
                maximo = -999999
                #Printeo de probabilidad de la clase en la que clasifica:
                # print(output[0])
                # for i in range(0,len(output[0])):
                #     if maximo < output[0][i]:
                #         maximo = output[0][i]
                # print(maximo)
                loss = criterion(output,l)

                test_loss += loss.item()*c.size(0)

                _, pred = torch.max(output, 1)
                # print(pred)
                # print('-'*50) 
                correct = pred.eq(l) #Comprobamos si el objeto pertence a la clase o no.
                
                correct_array =  l
                hipotesis_array = pred

                for i in range(c.size(0)):  
                    label = l.data[i]
                    class_correct[label] += correct[i].item()
                    class_total[label] += 1
                contador += 1
                #Compute of Confusion Matrix
                cm = confusion_matrix(correct_array.cpu(),hipotesis_array.cpu(),labels=range(n_classes))
                confmat = confmat + cm
            
            #resSK = accuracy_score(correct_array,hipotesis_array)
            #print(resSK)
            test_loss = test_loss / contador #Normalizamos
            #if(all==1):
                #print('Test loss: {}'.format(test_loss))

            #Accuracy for each class
            #if(all==1):
            for i in range(n_classes):
                if class_total[i] > 0:
                    if class_correct[i] == 0: 
                        errores += 1 
                        #list_errores.append(dataset_total[count]['id'])
                        if(all == 1):
                            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (str(i+1), 100 * class_correct[i] / class_total[i], np.sum(class_correct[i]), np.sum(class_total[i])))
                            print(dataset_total[count]['id'])
                            print(confmat)
                            print('*'*25) 
                        #list_error_confmats.append(confmat)
                # else:
                #     print('Test Accuracy of %5s: N/A (no training examples)' % (class_total[i]))
            
            count+=1
        #Total accuracy
        res = (100. * errores / len(dataset_total))#,np.sum(class_correct), np.sum(class_total))
        #print('\nTest Accuracy (Overall): %2d%%' % res)

        #Confusion Matrix 
        #if(all==1):
            #print("Confusion Matrix (Overall):",confmat)

        # Confidence interval:
        interval = (1.96 * sqrt((res/100*(1-res/100))/len(dataset_total)))*100
        #if(all==1):
            #print(f'Confidence Interval Radius:{interval}')
        print('ERRORES TOTALES:',errores)
        #PATH = '/Users/Juanjo Flores/OneDrive/Desktop/Clasificación de imágenes con RNN/code/leave_one_out'
        #torch.save(model, os.path.join(PATH,'model.pth'))
        return res,interval,len_feats
