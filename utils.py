from sklearn.model_selection import train_test_split
import numpy as np 
import torch 

def train_val_test_split(X,y,val_size, test_size, random_state):
    """
        X: feature matrix
        y: target valuies
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size/(1-test_size), random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test 

def corrupt_Y(Y1,Y2, p):
    """
        Randomly sample p% pairs, then swap 

        Y1, Y2: 1D numpy matrix
        p: float 0<p<1

    """

    Y = np.vstack([Y1, Y2]).T
    Y_corrupt = corrupt_X(Y,p)
    return Y_corrupt[:,0], Y_corrupt[:,1]

def corrupt_X(X, p):
    """
        Randomly sample p% rows, then permute each of them

        X: 2D numpy matrix
        p: float 0<p<1

    """

    n_samples = X.shape[0]
    n_features = X.shape[1]

    corrupt_idx = np.random.choice(n_samples, int(p*n_samples), replace=False)

    if n_features == 2:
        X_corrupt = X[:,[1,0]]    
        mask = np.expand_dims(np.in1d(np.arange(n_samples),corrupt_idx).astype(int),-1) ## 1 for corrupt, 0 for non-corrupt
        
        return mask*X_corrupt + (1-mask)*X

    else:
        #TODO: Parallelize this step, instead of using for-loop
        X_corrupt = np.copy(X)
        for idx in corrupt_idx:
            X_corrupt[idx] = X_corrupt[idx][np.random.permutation(n_features)]
        return X_corrupt

def np2tensor(X):
    X = torch.FloatTensor(X)
    if torch.cuda.is_available():
        X = X.cuda()
    
    return X  

