import torch
import torchvision.transforms as tr
from vit import MLP_Attention
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from rich.progress import track
import torch.nn as nn
from rich import pretty
########Transforms#######

transforms = tr.Compose([tr.ToTensor(),tr.Lambda(lambda x:torch.cat([x,x,x],0))])

########Hyperparameters and optimizer#########

batch_size = 32
num_heads = 8
embed_dim = 64
epochs = 100
lr = 0.0001
patch_size = 16
img_size = 28
model = MLP_Attention(n_embed=embed_dim,num_classes=10,patch_size=14)
optimizer = torch.optim.Adam(model.parameters(),lr=lr)


#######Dataset#######

train_dataset = MNIST(root='/home/chinu_tensor/basic_vision_transformer/mnist',download=True,transform=transforms)
test_dataset = MNIST(root='/home/chinu_tensor/basic_vision_transformer/mnist',download=True,train=False,transform=transforms)
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True) 
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True) 

######## Training ########

for epoch in track(range(epochs),description='Training'):
        epoch_loss = 0 
        for inps, target in train_loader:
 
            logits = model(inps)
            logits = logits.mean(dim=1)
            batch_loss = nn.functional.cross_entropy(logits, target)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            epoch_loss += batch_loss.item()

        epoch_loss /= len(train_loader)
        print(f"Loss:{epoch_loss}")

###### Evaluation #########

def evaluate_model(model,loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            logits = model(inputs)
            logits = logits.mean(dim=1)  # If using patch embeddings
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
    return 100 * correct / total

pretty.pprint(f"Test accuracy:{evaluate_model(model,test_loader)}")

######## Save Model dict #######

torch.save(model.state_dict(),'./checkpoints/vit.pt')

pretty.pprint(f'model saved in the checkpoints directory')




