from DataLoad import ASLTrainDataset, ASLTestDataset
from torch.utils.data import DataLoader
from Hyperparameters import Hyperparams
from cnnmodel import CNNModel
from torch.autograd import Variable
import torch
import torch.nn as nn


def save_model():
    torch.save(model, 'ASL.pt')


Hyperparams = Hyperparams(batch_size=100, num_workers=4, learning_rate=0.01,
                          epoch=100, n_iters=3000)
train_loader = ASLTrainDataset('/home/madhavkhosla/Documents/ASL Project/Dataset')
dataloader_train = DataLoader(train_loader, batch_size=Hyperparams.batch_size, shuffle=True,
                              num_workers=Hyperparams.num_workers)

test_loader = ASLTestDataset('/home/madhavkhosla/Documents/ASL Project/Dataset')
dataloader_test = DataLoader(test_loader, batch_size=Hyperparams.batch_size, shuffle=True,
                             num_workers=Hyperparams.num_workers)


num_epoch = 6
model = CNNModel()
if torch.cuda.is_available():
    model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=Hyperparams.learning_rate)

iter = 0
for epoch in range(num_epoch):
    for image, label in enumerate(dataloader_train):
        if torch.cuda.is_available():
            images = Variable(image.cuda())
            labels = Variable(label.cuda())
        else:
            image = Variable(image)
            label = Variable(label)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
save_model()
