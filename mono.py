import torch 
from torch import nn

import numpy as np 
import argparse
from tqdm import tqdm 

from model import Model
from dataset import get_mono_syn_data
from utils import train_val_test_split, corrupt_Y, np2tensor
from losses import rmse, mono_loss

def args_parser():
    parser = argparse.ArgumentParser()
    ### Model arguments
    parser.add_argument('--hid1', type=int, default=512)
    parser.add_argument('--hid2', type=int, default=256)

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
    (X1, y1), (X2, y2), (X3, y3) = get_mono_syn_data(args.n_samples)
    
    ### Split train val test set 
    idx_train, idx_val, idx_test, _, _, _ = \
        train_val_test_split(np.arange(args.n_samples), np.arange(args.n_samples), val_size=args.val_size, test_size=args.test_size, random_state=args.seed)

    X1_train = X1[idx_train]
    X2_train = X2[idx_train]
    X3_train = X3[idx_train]
    X_train = np.vstack([X1_train, X2_train, X3_train])
    y1_train = y1[idx_train]
    y2_train = y2[idx_train]
    y3_train = y3[idx_train]

    X1_val = X1[idx_val]
    X2_val = X2[idx_val]
    X3_val = X3[idx_val]
    X_val = np.vstack([X1_val, X2_val, X3_val])
    y1_val = y1[idx_val]
    y2_val = y2[idx_val]
    y3_val = y3[idx_val]
    y_val = np.concatenate([y1_val, y2_val, y3_val])

    X1_test = X1[idx_test]
    X2_test = X2[idx_test]
    X3_test = X3[idx_test]
    X_test = np.vstack([X1_test, X2_test, X3_test])
    y1_test = y1[idx_test]
    y2_test = y2[idx_test]
    y3_test = y3[idx_test]
    y_test = np.concatenate([y1_test, y2_test, y3_test])
    
    ### Corrupt data
    y1_train_noise, y2_train_noise1 = corrupt_Y(y1_train, y2_train, args.p)
    y2_train_noise2, y3_train_noise = corrupt_Y(y2_train, y3_train, args.p)
    
    y1_val_noise, y2_val_noise1 = corrupt_Y(y1_val, y2_val, args.p)
    y2_val_noise2, y3_val_noise = corrupt_Y(y2_val, y3_val, args.p)

    X_train = np2tensor(X_train)
    X_val = np2tensor(X_val)
    X_test = np2tensor(X_test)

    y1_train_noise = np2tensor(y1_train_noise)
    y2_train_noise1 = np2tensor(y2_train_noise1)
    y2_train_noise2 = np2tensor(y2_train_noise2)
    y3_train_noise = np2tensor(y3_train_noise)

    y1_val_noise = np2tensor(y1_val_noise)
    y2_val_noise1 = np2tensor(y2_val_noise1)
    y2_val_noise2 = np2tensor(y2_val_noise2)
    y3_val_noise = np2tensor(y3_val_noise)

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
        y_pred = model(X_train)

        loss = rmse(y_pred[:len(idx_train)],y1_train_noise) + \
            rmse(y_pred[len(idx_train):-len(idx_train)],y2_train_noise1) + \
            rmse(y_pred[len(idx_train):-len(idx_train)],y2_train_noise2) + \
            rmse(y_pred[-len(idx_train):],y3_train_noise)

        if args.adapt:
            adapt_loss = mono_loss(y_pred[:len(idx_train)], y_pred[len(idx_train):-len(idx_train)]) + \
                        mono_loss(y_pred[len(idx_train):-len(idx_train)], y_pred[-len(idx_train):])
        else:
            adapt_loss = 0
        # import pdb;pdb.set_trace()
        total_loss = loss + args.lam * adapt_loss

        total_loss.backward()
        optimizer.step()

        model.eval()
        
        y_pred = model(X_val)
        if args.noise_free:
            val_loss = rmse(y_pred,y_val)
        else:
            val_loss = rmse(y_pred[:len(idx_val)],y1_train_noise) + \
                    rmse(y_pred[len(idx_val):-len(idx_val)],y2_train_noise1) + \
                    rmse(y_pred[len(idx_val):-len(idx_val)],y2_train_noise2) + \
                    rmse(y_pred[-len(idx_val):],y3_train_noise)
        val_loss = val_loss.item()
        # import pdb;pdb.set_trace()
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