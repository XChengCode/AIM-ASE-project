#from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
import torch
import numpy as np

def draw_confusion_matrix(model, data_loader, device):
 
    model.eval()
    
    y_true = []
    y_pred= []

    with torch.inference_mode():
        for X, y in data_loader:
            y=y.tolist()
            for i in y:
                y_true.append(i)
      
            test_pred = model(X.to(device))
            test_pred = test_pred.tolist()
        
            for i in test_pred:
                single_pred=np.argmax(i)
                y_pred.append(single_pred)
               
    C = confusion_matrix(y_true, y_pred)
    #print(classification_report(y_true, y_pred))
    plt.matshow(A=C, cmap=plt.cm.Reds) 


    for i in range(len(C)):
        for j in range(len(C)):
            plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    plt.xticks(range(0,8), labels=['neutral','calm','happy','sad','angry','fear','disgust','surprise']) 
    plt.yticks(range(0,8), labels=['neutral','calm','happy','sad','angry','fear','disgust','surprise'])