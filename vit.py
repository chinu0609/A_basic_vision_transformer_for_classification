import torch
import torch.nn as nn

#Assuming data dim is [B,C,H,W] 



class Patch_Gen(nn.Module):
    

    def __init__(self,embed_dim:int=64,patch_size:int=8):
        super().__init__()
        self.patch_mapper = nn.Conv2d(in_channels=3,out_channels=embed_dim,kernel_size=patch_size,stride=patch_size)
        self.linear_1 = nn.Linear(embed_dim,embed_dim)
    

    def forward(self,x):
        x = self.patch_mapper(x)
        x = x.flatten(2) 
        x = x.transpose(1,2)
       
        x = self.linear_1(x)
        return x
      


class Attention_Part(nn.Module):
    def __init__(self,num_heads:int=8,embed_dim:int=64,patch_size:int=8):
        super().__init__()
        self.patch_embed = Patch_Gen(embed_dim=embed_dim,patch_size=patch_size)
        self.sa_head = nn.MultiheadAttention(embed_dim=embed_dim,num_heads=num_heads,batch_first=True)
    def forward(self,x):
        x = self.patch_embed(x)
        x,_ = self.sa_head(x,x,x) 
        return x


class MLP_Attention(nn.Module):
    def __init__(self,n_embed:int,num_classes:int,patch_size:int=8):
        super().__init__()
        self.main = Attention_Part(patch_size=patch_size,embed_dim=n_embed)
        self.layering = nn.Sequential(
            nn.Linear(n_embed,100),
            nn.ReLU(),
            nn.Linear(100,100),
            nn.ReLU(),
            nn.Linear(100,100),
            nn.ReLU(),
            nn.Linear(100,num_classes)
            )
    def forward(self,x):
        logits = self.main(x)
        logits = self.layering(logits)
        return logits 




