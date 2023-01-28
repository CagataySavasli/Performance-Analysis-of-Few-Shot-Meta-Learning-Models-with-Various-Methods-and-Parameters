import torch
import torch.nn as nn

import configs
from methods.DKT_regression import DKT
from methods.gpnet_regression import GPNet
from methods.feature_transfer_regression import FeatureTransfer
import backbone
import os
import numpy as np

def main(method, dataset, model, kernel_type, stop_epoch=100, seed=1):
    print(f"""
    method : {method}
    model : {model}
    dataset : {dataset}
    seed : {seed}
    type of seed : {type(seed)}
    stop_epoch : {stop_epoch}
    """)
    #params = parse_args_regression('train_regression')
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    checkpoint_dir = '%scheckpoints/%s/' % (configs.save_dir,  dataset)
    if not os.path.isdir( checkpoint_dir):
        os.makedirs( checkpoint_dir)
    checkpoint_dir = '%scheckpoints/%s/%s_%s' % (configs.save_dir,  dataset,  model,  method)

    bb           = backbone.Conv3().cuda()

    if  method=='DKT':
        model = DKT(bb,dataset,kernel_type).cuda()
    elif  method=='gpnet':
        bb = backbone.Conv3()
        model = GPNet(bb, dataset,kernel_type)
    elif  method=='transfer':
        model = FeatureTransfer(bb, dataset).cuda()
    else:
        ValueError('Unrecognised method')

    optimizer = torch.optim.SGD([{'params': model.model.parameters(), 'lr': 0.001},
                                {'params': model.feature_extractor.parameters(), 'lr': 0.001}])

    for epoch in range( stop_epoch):
        model.train_loop(epoch, optimizer)

    model.save_checkpoint( checkpoint_dir)

