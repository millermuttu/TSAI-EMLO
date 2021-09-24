
import csv
import torchvision
import torch.nn as nn
import torch
from torchvision import transforms,models,datasets
from torch import optim
import os
from collections import OrderedDict
import sys


pathname = os.path.dirname(sys.argv[0])
path = os.path.abspath(pathname)
train_data_dir = os.path.join(path,'data', 'train')
validation_data_dir = os.path.join(path,'data', 'validation')
batch_size = 8

train_transform = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])

train_dataset = torchvision.datasets.ImageFolder(train_data_dir, transform= train_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size ,shuffle=True)

test_transform = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])

test_dataset = torchvision.datasets.ImageFolder(train_data_dir, transform= test_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size ,shuffle=True)



model = models.densenet121(pretrained = True)
for params in model.parameters():
    params.requires_grad = False


classifier = nn.Sequential(OrderedDict([
    ('fc1',nn.Linear(1024,500)),
    ('relu',nn.ReLU()),
    ('fc2',nn.Linear(500,2)),
    ('Output',nn.LogSoftmax(dim=1))
]))

model.classifier = classifier
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = optim.Adam(model.classifier.parameters())
criterian = nn.NLLLoss()
list_train_loss = []
list_test_loss = []
f = open('metrics.csv', 'w')
file = csv.writer(f)
file.writerow(['Epoch', 'Train loss', 'Test loss', 'Accuracy'])
epochs = 10
for epoch in range(epochs):
    train_loss = 0
    test_loss = 0
    for bat, (img, label) in enumerate(train_loader):
        # moving batch and lables to gpu
        img = img.to(device)
        label = label.to(device)

        model.train()
        optimizer.zero_grad()

        output = model(img)
        loss = criterian(output, label)
        loss.backward()
        optimizer.step()
        train_loss = train_loss + loss.item()
        # print(bat)

    accuracy = 0
    for bat, (img, label) in enumerate(test_loader):
        img = img.to(device)
        label = label.to(device)

        model.eval()
        logps = model(img)
        loss = criterian(logps, label)

        test_loss += loss.item()
        ps = torch.exp(logps)
        top_ps, top_class = ps.topk(1, dim=1)
        equality = top_class == label.view(*top_class.shape)
        accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

    list_train_loss.append(train_loss / 20)
    list_test_loss.append(test_loss / 20)
    print('epoch: ', epoch, '    train_loss:  ', train_loss / 20, '   test_loss:    ', test_loss / 20,
          '    accuracy:  ', accuracy / len(test_loader))
    file.writerow([epoch,train_loss / 20, test_loss / 20,accuracy / len(test_loader)])


# torch.save(model.state_dict(), 'model.pth')
torch.save(model, "model_pytorch.h5")
f.close()

