import torch as t
import matplotlib.pyplot as plt
from torch import nn,optim
import torch.nn.functional as F
from torchvision import transforms, datasets, models
import helper
import numpy as np

#datasets can be downloaded from kaggle
#execute the below command in terminal
#please read kaggle documentation on how to download datasets over terminal
#kaggle datasets download -d nafisur/dogs-vs-cats

#!pip install pillow==4.1.1
#%reload_ext autoreload
#%autoreload

train_transform = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])

test_transform = transforms.Compose([transforms.Resize(225),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor()])

training_data = datasets.ImageFolder('dataset/training_set', transform = train_transform)
test_data = datasets.ImageFolder('dataset/test_set', transform = test_transform)

train_loader = t.utils.data.DataLoader(training_data, batch_size = 64, shuffle = True)
test_loader = t.utils.data.DataLoader(test_data, batch_size=64, shuffle = True)

# Use GPU if it's available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#download model "densenet121"
model = models.densenet121(pretrained=True)

#print(model.state_dict().keys())

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False
    
model.classifier = nn.Sequential(nn.Linear(1024, 256),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(256, 2),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)

model.to(device);

import time

#for device in ['cpu','cuda']:
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = 0.001)
model.to(device)

epochs = 1
steps = 0
running_loss = 0
print_every = 5
for epoch in range(epochs):
    for inputs, labels in train_loader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(test_loader):.3f}.. "
                  f"Test accuracy: {accuracy/len(test_loader):.3f}")
            running_loss = 0
            model.train()

#save the model
t.save(model.state_dict(), 'model.pth')

#load the model
state_dict = t.load('model.pth', map_location = 'cpu')

#updating model's weight, biases etc
model.load_state_dict(state_dict)

#testing of  model
for test_image, test_label in test_loader:
    test_image += test_image
    
img = test_image[0].view(1,3,224,224)
helper.imshow(test_image[0])

with t.no_grad():
    logits = model.forward(img)
    
ps = t.exp(logits)
print(ps)

def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax

#testing of model's accuracy
for test_img2, test_lbl2 in test_loader:
    test_img2 += test_img2
    
#print(len(image))
acc_test_img2 = test_img2.view(test_img2.shape[0], 3, 224, 224)
print(acc_test_img2.shape)
with t.no_grad():
    logits2 = model.forward(acc_test_img2)

ps2 = F.softmax(logits2, dim = 1)
print(ps2.shape)
#measuring accuracy
top_p, top_class = ps2.topk(1, dim = 1)

equals = top_class == test_lbl2.view(*top_class.shape)

print('test lebel = ', test_lbl2.view(top_class.shape))
print('top class = ', top_class)
print('equal = ', equals)

#accuracy
accuracy = t.mean(equals.type(t.FloatTensor))
print(f'accuracy: {accuracy.item()*100}%')

def result(prediction): 
    
    if(prediction == 0):
        pred = 'Cat'
    else:
        pred = 'Dog'
        
    return(pred)
    
imshow(test_img2[3])   
result(top_class[3])

'''
#print('training loss',training_loss)
test_loss = 0
accuracy = 0

# Turn off gradients for validation, saves memory and computations
with torch.no_grad():
    # set to evaluation mode
    model.eval()
    for images, labels in test_loader:
        log_ps = model(images)
        test_loss += criterion(log_ps, labels)
                
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor))
training_loss.append(running_loss/len(train_loader))
test_losses.append(test_loss/len(test_loader))

print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(running_loss/len(train_loader)),
              "Test Loss: {:.3f}.. ".format(test_loss/len(test_loader)),
              "Test Accuracy: {:.3f}".format(accuracy/len(test_loader)))

#setting model back to training mode
model.train()
#plotting losses
plt.plot(training_losses, label = 'Traning loss')
plt.plot(test_losses, label = 'Test loss')
plt.legend(frameon = False)
#t.device('cpu')

#saving and loading of models
torch.save(model.state_dict(), 'model.pth')
'''
'''
loaded_model = t.load('model.pth',map_location='cpu')
#print(loaded_model.keys())

state_dict = torch.load('model.pth')
model.load_state_dict(state_dict)

for test_img, test_lbl in test_loader:
    test_img, test_lbl = test_img.to('cuda'), test_lbl.to('cuda')
    test_img += test_img
img = test_img[0].view(test_img.shape[0],-1)
with t.no_grad():
    logits = model.forward(img)

ps = F.softmax(logits, dim = 1)

helper.view_classify(img.view(1,224,224), ps, version = 'Fashion')
'''
