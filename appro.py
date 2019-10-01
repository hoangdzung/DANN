import torch 
from torch import nn
from torch.nn import functional as F 

import numpy as np 
import argparse
from tqdm import tqdm 

from model import Model
from dataset import get_appro_syn_data
from utils import train_val_test_split, corrupt_X, np2tensor
from losses import rmse, appro_loss

def args_parser():
    parser = argparse.ArgumentParser()
    ### Model arguments
    parser.add_argument('--hid1', type=int, default=64)
    parser.add_argument('--hid2', type=int, default=128)

    ### Data arguments
    parser.add_argument('--n_samples', type=int, default=500)
    parser.add_argument('--p', type=float, default=0.1)
    parser.add_argument('--val_size', type=float, default=0.2)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--noise_free',action="store_true")

    ### Loss arguments
    parser.add_argument('--adapt',action="store_true")
    parser.add_argument('--lam', type=float, default=1.0)

    ### Training arguments
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wdc', type=float, default=1e-5)

    ### Additional arguments
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--debug',action="store_true")

    args = parser.parse_args()

    return args 

def main(args):
    X, y = get_appro_syn_data(args.n_samples)
    mean, std = np.mean(y), np.std(y)
    min_y = mean-std
    max_y = mean+std 

    X_train, X_val, X_test, y_train, y_val, y_test = \
        train_val_test_split(X, y, val_size=args.val_size, test_size=args.test_size, random_state=args.seed)

    X_train_noise = corrupt_X(X_train, args.p)
    X_val_noise = corrupt_X(X_val, args.p)

    X_train = np2tensor(X_train)
    X_train_noise = np2tensor(X_train_noise)
    X_val = np2tensor(X_val)
    X_val_noise = np2tensor(X_val_noise)
    X_test = np2tensor(X_test)
    y_train = np2tensor(y_train)
    y_val = np2tensor(y_val)
    y_test = np2tensor(y_test)

    model = Model(args.hid1, args.hid2)

    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.wdc) 
    
    best_val = 1e20 
    result = None 

    for epoch in tqdm(range(args.n_epochs)):
        #TODO: Mini-batch training
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train_noise)

        loss = rmse(y_pred,y_train)

        if args.adapt:
            adapt_loss =  appro_loss(y_pred, min_y, max_y)
        else:
            adapt_loss = 0

        total_loss = loss + args.lam * adapt_loss

        total_loss.backward()
        optimizer.step()

        model.eval()
        
        if args.noise_free:
            y_pred = model(X_val)
        else:
            y_pred = model(X_val_noise)

        val_loss = rmse(y_pred,y_val).item()
        if val_loss < best_val:
            best_val = val_loss
            y_pred = model(X_test)
            result = rmse(y_pred,y_test).item()

        if args.debug:
            print("Epoch ", epoch, total_loss.item(), val_loss, result)

    print(result)


args = args_parser()

### Set seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

### Training
main(args)