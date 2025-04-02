import os
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split
import config


class ResidualBloodSetDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.transform = transform
        self.images = images
        self.labels = labels
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self,idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def dataPath_stratified_spilt(data_dir, class_to_idx, split_ratio = [0.6, 0.2, 0.2]):
    images=[]
    labels=[]
    for img_name in os.listdir(data_dir):
        img_path = os.path.join(data_dir, img_name)
        images.append(img_path)
        labels.append(class_to_idx[img_name.split('_')[1]])
    
    test_size = 1-split_ratio[0]
    xtr, xte, ytr, yte = train_test_split(images, labels, test_size=test_size, stratify=labels)
    test_size = split_ratio[2]/(split_ratio[1]+split_ratio[2])
    xval, xte, yval, yte = train_test_split(xte, yte, test_size=test_size, stratify=yte) 

    return xtr, xval, xte, ytr, yval, yte
    


def train(model, training_set, valid_set, loss_fn, optimizer, epoch, model_fname, with_mixup=False):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if with_mixup==True:
        mixup = v2.MixUp(num_classes=config.dataSetting['classSize'])

    # best_loss = np.inf
    best_acc = 0
    for i in range(0,epoch):
        epoch_loss=0
        for xi,yi in training_set:
            if with_mixup==True:
                xi,yi = mixup(xi,yi)
            model.train()
            yhat = model(xi.to(device))
            loss = loss_fn(yhat, yi.to(device))
            epoch_loss=epoch_loss+loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if (i%1==0):
            with torch.no_grad():
                model.eval()
                correct=0
                validation_loss = 0
                for xi,yi in valid_set:
                    pred = model(xi.to(device))
                    validation_loss = validation_loss+loss_fn(pred, yi.to(device))
                    correct = correct+(pred.to("cpu").argmax(1) == yi.to("cpu")).type(torch.float).sum().item()
            training_loss = epoch_loss.item()/len(training_set.dataset)
            validation_loss = validation_loss.item()/len(valid_set.dataset)
            validation_acc = correct/len(valid_set.dataset)
            print("-"*20)
            print(f'[{i+1:>6d} epochs]')
            print(f'Training loss: {training_loss:>10.6f}')
            print(f'Validation loss: {validation_loss:>10.6f}')
            print(f'Validation Acc.: {validation_acc:>10.6f}')
            print("-"*20)
            # log metrics to wandb
            # if validation_loss<best_loss:
            if validation_acc>best_acc:
                torch.save(model, f'{model_fname}.pth')
                # best_loss = validation_loss
                best_acc = validation_acc
                print('current best model!')
    return model

if __name__ == '__main__':
    pass
    # data_dir = '/workspace/Dialysis_Form'
    # xtr, xval, xte, ytr, yval, yte = dataPath_stratified_spilt(data_dir=data_dir, class_to_idx={'10':0,'30':1})

    # stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # transform = v2.Compose([
    #     v2.Resize((224,224)),
    #     v2.ToTensor(),
    #     tv2.Normalize(*stats)])

    # train_set = ResidualBloodSetDataset(xtr, ytr, transform=transform)
    # for xi,yi in train_set:
    #     print(xi.shape)
    #     print(yi.shape)
    #     break

    