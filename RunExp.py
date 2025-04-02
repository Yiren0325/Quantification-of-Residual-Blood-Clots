import config
from residualBloodUtils import ResidualBloodSetDataset
from residualBloodUtils import dataPath_stratified_spilt
from residualBloodUtils import train
from models import CustomConvNet

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import logging
from datetime import datetime


if __name__ == '__main__':
    FORMAT = '%(asctime)s %(levelname)s: %(message)s'
    logging.basicConfig(level=logging.INFO,filename='exp_log.log',filemode='a', format=FORMAT)
    
    logging.info(f'------ Experiment_{datetime.today().date()} ------')
    logging.info(f'{config.trainSetting}')

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device...")

    input_size = config.dataSetting['inputSize']
    class_size = config.dataSetting['classSize']
    class_to_idx = config.dataSetting['class_to_idx']
    batch_size = config.trainSetting['batchSize']
    lr = config.trainSetting['lr']
    model_fname = config.trainSetting['modelName']
    runs = config.expSetting['runs']
    

    stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    transform_set = [
        v2.ColorJitter(brightness=.1, contrast=.1, saturation=.1, hue=.1),
        v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))
    ]
    transform_train = v2.Compose([
        v2.PILToTensor(),
        v2.Resize((input_size,input_size)),
        v2.RandomHorizontalFlip(p=0.5),
        # v2.RandomChoice(transform_set),
        v2.ColorJitter(brightness=.1, contrast=.1, saturation=.1, hue=.1),
        # v2.AugMix(),
        v2.ToDtype(torch.float32),
        v2.Normalize(*stats)])


    transform = v2.Compose([
        v2.PILToTensor(),
        v2.Resize((input_size,input_size)),
        v2.ToDtype(torch.float32),
        v2.Normalize(*stats)])

    # data_dir = '/workspace/raw_combined_images'
    data_dir = '/workspace/Dialysis_Form'

    
    testing_acc_no_pretrained=[]
    testing_acc = []
    testing_acc_aug = []

    for run in range(runs):
        xtr, xval, xte, ytr, yval, yte = dataPath_stratified_spilt(
            data_dir=data_dir,
            class_to_idx=class_to_idx
            )

        train_set = ResidualBloodSetDataset(
            images=xtr, 
            labels=ytr, 
            transform=transform
            )

        train_set_aug = ResidualBloodSetDataset(
            images=xtr, 
            labels=ytr, 
            transform=transform_train
            )

        valid_set = ResidualBloodSetDataset(
            images=xval, 
            labels=yval, 
            transform=transform
            )

        test_set = ResidualBloodSetDataset(
            images=xte, 
            labels=yte, 
            transform=transform
            )

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        train_loader_aug = DataLoader(train_set_aug, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


        '''
        training without pretrained model
        '''
        model = CustomConvNet(class_size,pretrained=False)
        # model = CustomEfficientNet(class_size)
        # model = CustomResNet(class_size)
        model.to(device)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(),lr=lr)

        model = train(
            model=model,
            training_set=train_loader,
            valid_set=valid_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epoch=config.trainSetting['epochs'],
            model_fname=model_fname
            )

        model = torch.load(f'{model_fname}.pth')
        model.eval()
        with torch.no_grad():    
            correct=0
            for xi,yi in test_loader:
                pred = model(xi.to(device))
                correct = correct+(pred.to("cpu").argmax(1) == yi.to("cpu")).type(torch.float).sum().item()
        acc = correct/len(test_loader.dataset)
        testing_acc_no_pretrained.append(acc)
        print('Testing Acc:',acc)
        logging.info(f'Run {run}(w/o aug and pretrained): {acc}')


        '''
        training without horizontal flip augmentation
        '''
        model = CustomConvNet(class_size)
        # model = CustomEfficientNet(class_size)
        # model = CustomResNet(class_size)
        model.to(device)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(),lr=lr)

        model = train(
            model=model,
            training_set=train_loader,
            valid_set=valid_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epoch=config.trainSetting['epochs'],
            model_fname=model_fname
            )

        model = torch.load(f'{model_fname}.pth')
        model.eval()
        with torch.no_grad():    
            correct=0
            for xi,yi in test_loader:
                pred = model(xi.to(device))
                correct = correct+(pred.to("cpu").argmax(1) == yi.to("cpu")).type(torch.float).sum().item()
        acc = correct/len(test_loader.dataset)
        testing_acc.append(acc)
        print('Testing Acc:',acc)
        logging.info(f'Run {run}(w/o aug): {acc}')


        '''
        training with horizontal flip augmentation
        '''
        # model_aug = CustomEfficientNet(class_size)
        model_aug = CustomConvNet(class_size)
        # model_aug = CustomResNet(class_size)
        
        model_aug.to(device)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model_aug.parameters(),lr=lr)

        model_aug = train(
            model=model_aug,
            training_set=train_loader_aug,
            valid_set=valid_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epoch=config.trainSetting['epochs'],
            model_fname=model_fname,
            # with_mixup=True
            )

        model_aug = torch.load(f'{model_fname}.pth')
        model_aug.eval()
        with torch.no_grad():    
            correct=0
            for xi,yi in test_loader:
                pred = model_aug(xi.to(device))
                correct = correct+(pred.to("cpu").argmax(1) == yi.to("cpu")).type(torch.float).sum().item()
        acc = correct/len(test_loader.dataset)
        testing_acc_aug.append(acc)
        print('Testing Acc:',acc)
        logging.info(f'Run {run}(w/ aug): {acc}')
    
    print('w/o Aug and Pretrained: ',np.mean(testing_acc_no_pretrained), np.std(testing_acc_no_pretrained))
    logging.info(f'Avg. Acc(w/o aug and Pretrained:): {np.mean(testing_acc_no_pretrained)}')
    logging.info(f'std(w/o aug and Pretrained:): {np.std(testing_acc_no_pretrained)}')

    print('w/o Aug: ',np.mean(testing_acc), np.std(testing_acc))
    logging.info(f'Avg. Acc(w/o aug): {np.mean(testing_acc)}')
    logging.info(f'std(w/o aug): {np.std(testing_acc)}')
    
    print('w/ Aug: ',np.mean(testing_acc_aug), np.std(testing_acc_aug))
    logging.info(f'Avg. Acc(w/ aug): {np.mean(testing_acc_aug)}')
    logging.info(f'std(w/o aug): {np.std(testing_acc_aug)}')

            
            