import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import os
import util
from sklearn.model_selection import train_test_split
 

if __name__ == "__main__":
    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')
        
    os.system("kaggle competitions download -c journey-springfield --unzip --force")

    train_val_files = sorted(list(util.TRAIN_DIR.rglob('*.jpg')))
    test_files = sorted(list(util.TEST_DIR.rglob('*.jpg')))
    
    train_val_labels = [path.parent.name for path in train_val_files]
    train_files, val_files = train_test_split(train_val_files, test_size=0.25, \
                                            stratify=train_val_labels)
    
    train_dataset = util.SimpsonsDataset(train_files, mode='train')
    val_dataset = util.SimpsonsDataset(val_files, mode='val')
    
    n_classes = len(np.unique(train_val_labels))
    simple_cnn = util.SimpleCnn(n_classes).to(util.DEVICE)
    print("we will classify :{}".format(n_classes))
    print(simple_cnn)
    
    weights = ResNet50_Weights.DEFAULT # предобученные веса
    resnet = resnet50(weights=weights)

    for param in resnet.parameters():
        param.requires_grad = False # please, don't learn....

    layers = list(resnet.children()) 
    for layer in layers[-3:]: # 3 последних слоя будут обучаться, чтобы подстроиться под наши данные
        for param in layer.parameters():
            param.requires_grad = True
            
    in_features = 2048
    out_features = 42
    resnet.fc = nn.Linear(in_features, out_features)
    resnet = resnet.to(util.DEVICE)

