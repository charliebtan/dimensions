# some useful functions

import math
import torch
import numpy as np

# Corollary 2.4 in Mohammadi 2014
def alpha_estimator(m, X):
    # X is N by d matrix
    N = len(X)
    n = int(N/m) # must be an iteger
    Y = torch.sum(X.view(n, m, -1), 1)
    eps = np.spacing(1)
    Y_log_norm = torch.log(Y.norm(dim=1) + eps).mean()
    X_log_norm = torch.log(X.norm(dim=1) + eps).mean()
    diff = (Y_log_norm - X_log_norm) / math.log(m)
    return 1 / diff
