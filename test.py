import torch
import torch.nn as nn
import torch.optim as optim
import configs
from data.qmul_loader import get_batch, train_people, test_people
from data.AAF_loader import get_batch, train_people, test_people
from io_utils import parse_args_regression, get_resume_file
from methods.DKT_regression import DKT
from methods.gpnet_regression import GPNet
from methods.feature_transfer_regression import FeatureTransfer
import backbone
import numpy as np

def main(method, dataset, model, kernel_type, seed=1, n_test_epochs=10, n_support=5):
        
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"""
    method : {method}
    model : {model}
    dataset : {dataset}
    seed : {seed}
    type of seed : {type(seed)}
    n_test_epochs : {n_test_epochs}
    n_support : {n_support}
    """)
    checkpoint_dir = '%scheckpoints/%s/%s_%s' % (configs.save_dir, dataset, model, method)
    bb           = backbone.Conv3().cuda()

    if method=='DKT':
        model = DKT(bb,dataset,kernel_type).cuda()
        optimizer = None
    if method=='gpnet':
        bb = backbone.Conv3()
        model = GPNet(bb,dataset,kernel_type)
        optimizer = None
    elif method=='transfer':
        model = FeatureTransfer(bb, dataset).cuda()
        optimizer = optim.Adamax([{'params':model.parameters(),'lr':0.001}])
    else:
        ValueError('Unrecognised method')

    model.load_checkpoint(checkpoint_dir)

    mse_list = []
    for epoch in range(n_test_epochs):
        mse = float(model.test_loop(n_support, optimizer).cpu().detach().numpy())
        mse_list.append(mse)

    result = "Average MSE: " + str(np.mean(mse_list)) + " +- " + str(np.std(mse_list))
    print("-------------------")
    print(result)
    print("-------------------")
    return result