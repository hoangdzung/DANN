import torch 
from torch.nn import functional as F 

def rmse(y_pred,y_true):
    return torch.sqrt(torch.mean((y_pred-y_true)**2))

def appro_loss(y_pred, min_y, max_y):
    return torch.sum(F.relu(min_y-y_pred)+F.relu(y_pred-max_y))
    
def mono_loss(y1, y2):
    return torch.sum((y1>y2).detach().float()*F.relu(y1-y2))