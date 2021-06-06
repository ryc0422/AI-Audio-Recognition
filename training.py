import os, sys, time, random, pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as Data
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from simple_cnn import CNN
from googlenet import GoogLeNet
from preprocess_data_feature import AudioDataset, load_split_file



def train(train_loader, model, criterion, optimizer):
    model.train()
    running_loss = 0.0
    running_correct = 0

    for i, data in enumerate(tqdm(train_loader)):
        
        data, target = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        #output = model(data)
        
        (output,aux2,aux) = model(data) #for googlenet
        loss = criterion(output, target) 

        running_loss += loss.item() * len(target)

        #preds = torch.max(output.data)
        preds = torch.argmax(output.data, dim=1)

        correct = (preds == target).sum()

        running_correct += correct.item() 
        loss.backward()
        optimizer.step()  

    
    train_loss = running_loss/len(train_loader.dataset)
    train_accuracy = 100 * (running_correct / len(train_loader.dataset))
    
    
    return train_loss, train_accuracy


def validate(test_loader, model,criterion):
    model.eval()
    running_loss = 0.0
    running_correct = 0

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            data, target = data[0].to(device), data[1].to(device)
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item() *len(target)
            preds = torch.argmax(output.data, dim=1)
            correct = (preds == target).sum()
            running_correct += correct.item() 
    eval_loss = running_loss / len(test_loader.dataset)
    eval_accuracy = 100 * running_correct / len(test_loader.dataset)
    
    return eval_loss, eval_accuracy


if __name__ == "__main__":
    
    """ Hyper Parameters """
    N_CLASS = 6
    TRAIN_BATCH_SIZE = 8
    VALID_BATCH_SIZE = 8
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 5    

    """ GPU or CPU """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """ Load model """
    model_name = 'googlenet_feature'
    model = GoogLeNet(num_classes=N_CLASS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()


    """ Generate Data """
    path = './training'
    ratio = 0.8
    train_x, train_y,test_x,test_y = load_split_file(path, ratio) # x:record; y: label   
    trainset = AudioDataset(train_x, train_y)
    testset = AudioDataset(test_x, test_y)
    train_loader = DataLoader(dataset = trainset, batch_size = TRAIN_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset = testset, batch_size=VALID_BATCH_SIZE, shuffle=False)

    """ Start training """
    Best_acc = 0
    recorder = []
    print('Start training')
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        Train_loss, Train_acc = train(train_loader, model, criterion, optimizer)
        Test_loss, Test_acc = validate(test_loader, model, criterion)
        epoch_time = time.time()-start_time
        recorder.append([epoch, Train_loss, Train_acc, Test_loss, Test_acc])
        print(f'Epoch{epoch}: Train_loss={Train_loss:.6f}, Train_acc={Train_acc:.6f}%; Test_loss={Test_loss:.6f}, Test_acc={Test_acc:.6f}%')
        print(epoch_time)

        """ Save model """
        if Train_acc > Best_acc:
            Best_acc = Train_acc
            torch.save(model.state_dict(), f'Model/{model_name}.pt')

        with open('Model/cnn.record', 'wb') as f:
            pickle.dump(np.array(recorder), f)
