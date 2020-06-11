import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import data
import resnet18
import os
import numpy as np
import cls_hrnet
import hrnet
## Note that: here we provide a basic solution for training and validation.
## You can directly change it if you find something wrong or not good enough.

_trainloss=[]
_trainacc=[]
_validloss=[]
_validacc=[]

def train_model(model,train_loader, valid_loader, criterion, optimizer,scheduler, num_epochs=20):

    def train(model, train_loader,optimizer,criterion):
        model.train(True)
        total_loss = 0.0
        total_correct = 0

        for i,(inputs, labels) in enumerate(train_loader):
            labels = labels.cuda(non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predictions = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            total_correct += torch.sum(predictions == labels.data)

        epoch_loss = total_loss / len(train_loader.dataset)
        epoch_acc = total_correct.double() / len(train_loader.dataset)
        return epoch_loss, epoch_acc.item()

    def valid(model, valid_loader,criterion):
        model.train(False)
        total_loss = 0.0
        total_correct = 0
        for inputs, labels in valid_loader:
            labels = labels.cuda(non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predictions = torch.max(outputs, 1)
            total_loss += loss.item() * inputs.size(0)
            total_correct += torch.sum(predictions == labels.data)
        epoch_loss = total_loss / len(valid_loader.dataset)
        epoch_acc = total_correct.double() / len(valid_loader.dataset)
        return epoch_loss, epoch_acc.item()

    best_acc = 0.0
    for epoch in range(num_epochs):
        
        print('epoch:{:d}/{:d}'.format(epoch, num_epochs))
        print('*' * 100)
        train_loss, train_acc = train(model, train_loader,optimizer,criterion)
        print("training: {:.4f}, {:.4f}".format(train_loss, train_acc))
        _trainloss.append(train_loss)
        _trainacc.append(train_acc)
        valid_loss, valid_acc = valid(model, valid_loader,criterion)
        print("validation: {:.4f}, {:.4f}".format(valid_loss, valid_acc))
        _validloss.append(valid_loss)
        _validacc.append(valid_acc)
        scheduler.step()

        if valid_acc > best_acc:
            best_acc = valid_acc
            best_model = model
            torch.save(best_model, 'best_model.pt')

def test(model, valid_loader,prediction,turelabel):
    model.train(False)
    for inputs, labels in valid_loader:
        #inputs = inputs.to(device)
        labels = labels.cuda(non_blocking=True)
        outputs = model(inputs)
        predictions = torch.max(outputs, 1)
        prediction.extend(predictions.cpu().detach().numpy().tolist())
        turelabel.extend(labels.data.cpu().detach().numpy().tolist())

    return predictions,turelabel 

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    ## about model
    num_classes = 10

    ## about data
    data_dir = "../data/"
    inupt_size = 224
    batch_size = 32

    ## about training
    num_epochs =100
    lr = 0.001
    cfg={'STAGE1':{
        'BLOCK': 'BOTTLENECK',
        'FUSE_METHOD': 'SUM',
        'NUM_BLOCKS': [1],
        'NUM_CHANNELS': [32],
        'NUM_MODULES': 1,
        'NUM_RANCHES': 1},
        'STAGE2':
        {'BLOCK': 'BASIC',
         'FUSE_METHOD': 'SUM',
         'NUM_BLOCKS': [2,2],
         'NUM_BRANCHES': 2,
         'NUM_CHANNELS': [16, 32],
         'NUM_MODULES': 1},
        'STAGE3':
        {'BLOCK': 'BASIC',
         'FUSE_METHOD': 'SUM',
         'NUM_BLOCKS': [2, 2, 2],
         'NUM_BRANCHES': 3,
         'NUM_CHANNELS': [16, 32, 64],
         'NUM_MODULES': 1},
        'STAGE4':
        {'BLOCK': 'BASIC',
         'FUSE_METHOD': 'SUM',
         'NUM_BLOCKS': [2, 2, 2, 2],
         'NUM_BRANCHES': 4,
         'NUM_CHANNELS': [16, 32, 64, 128],
         'NUM_MODULES': 1
        }}
    ############### model initialization##################
    #model = resnet18.model_B(num_classes=num_classes)
    
    model=hrnet.get_cls_net(cfg)
    #model=cls_hrnet.get_cls_net(cfg)
    cudnn.benchmark=True
    torch.backends.cudnn.deterministic =False
    torch.backends.cudnn.enabled=True
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   # if torch.cuda.is_available():
       # print(" gpu is ok")
    #model = model.to(device)

    ## data preparation
    train_loader, valid_loader = data.load_data(data_dir=data_dir,input_size=inupt_size, batch_size=batch_size)

    ## optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs+10, eta_min=0, last_epoch=-1)
    ## loss function
    criterion = nn.CrossEntropyLoss().cuda()
    train_model(model,train_loader, valid_loader, criterion, optimizer,scheduler, num_epochs=num_epochs)

    ## figure out curves
    c=[_trainloss,_validloss,_trainacc,_validacc]
    np.savetxt('resultC.txt',c)

 
    


