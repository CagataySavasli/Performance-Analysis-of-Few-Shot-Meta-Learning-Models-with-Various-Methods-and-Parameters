## Original packages
import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math
import torch.nn.functional as F

## Our packages
import gpytorch
from time import gmtime, strftime
import random
from statistics import mean
from gpytorch.kernels import Kernel

from torch import Tensor
from methods.gencheb import gencheb

class GPNet(nn.Module):
    def __init__(self, backbone, dataset, kernel_type):
        super(GPNet, self).__init__()
        ## GP parameters
        self.feature_extractor = backbone
        self.dataset = dataset
        self.kernel_type = kernel_type
        self.get_model_likelihood_mll() #Init model, likelihood, and mll


    def get_model_likelihood_mll(self, train_x=None, train_y=None):
        if self.dataset == "AAF":
            # TODO: change the size of your inputs here!
            if(train_x is None): train_x=torch.ones(1, 1764)
            if(train_y is None): train_y=torch.ones(1)
        elif self.dataset == "QMUL":
            if(train_x is None): train_x=torch.ones(19, 2916)
            if(train_y is None): train_y=torch.ones(19)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPLayer(train_x=train_x, train_y=train_y, likelihood=likelihood, kernel=self.kernel_type, dataset=self.dataset)

        self.model      = model
        self.likelihood = likelihood
        self.mll        = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        self.mse        = nn.MSELoss()

        return self.model, self.likelihood, self.mll

    def set_forward(self, x, is_feature=False):
        pass

    def set_forward_loss(self, x):
        pass

    def train_loop(self, epoch, optimizer):
        if self.dataset == "AAF":
            from data.AAF_loader import get_batch, train_people, test_people, normalize_age, invert_normalize_age
        elif self.dataset == "QMUL":
            from data.qmul_loader import get_batch, train_people, test_people
        batch, batch_labels = get_batch(train_people)
        batch, batch_labels = batch, batch_labels
        for inputs, labels in zip(batch, batch_labels):
            optimizer.zero_grad()
            z = self.feature_extractor(inputs)

            self.model.set_train_data(inputs=z, targets=labels)
            predictions = self.model(z)
            loss = torch.abs(-self.mll(predictions, self.model.train_targets))

            loss.backward()
            optimizer.step()
            mse = self.mse(predictions.mean, labels)

            if (epoch%10==0):
                print('[%d] - Loss: %.3f  MSE: %.3f noise: %.3f' % (
                    epoch, loss.item(), mse.item(),
                    self.model.likelihood.noise.item()
                ))
                if self.dataset == "AAF":
                    mean = float(predictions.mean) * 100 + 50
                    print(mean, float(labels[0]*100 + 50), mean - 1.96 * np.sqrt(float(predictions.variance) * 100 + 50),
                        mean + 1.96 * np.sqrt(float(predictions.variance) * 100 + 50))

    def test_loop(self, n_support, optimizer=None): # no optimizer needed for GP
        if self.dataset == "AAF":
            from data.AAF_loader import get_batch, train_people, test_people, normalize_age, invert_normalize_age
            inputs, targets = get_batch(test_people)

            # TODO: modify output shape here!
            support_ind = list(np.random.choice(list(range(1)), replace=True, size=n_support))
            query_ind   = [i for i in range(1) if i not in support_ind]

        elif self.dataset == "QMUL":
            from data.qmul_loader import get_batch, train_people, test_people
            inputs, targets = get_batch(test_people)

            support_ind = list(np.random.choice(list(range(19)), replace=False, size=n_support))
            query_ind   = [i for i in range(19) if i not in support_ind]

        x_all = inputs
        y_all = targets

        x_support = inputs[:,support_ind,:,:,:]
        y_support = targets[:,support_ind]
        x_query   = inputs[:,query_ind,:,:,:]
        y_query   = targets[:,query_ind]

        # choose a random test person
        n = np.random.randint(0, len(test_people)-1)

        z_support = self.feature_extractor(x_support[n]).detach()
        self.model.set_train_data(inputs=z_support, targets=y_support[n], strict=False)

        self.model.eval()
        self.feature_extractor.eval()
        self.likelihood.eval()

        with torch.no_grad():
            z_query = self.feature_extractor(x_all[n]).detach()
            print(z_query)
            pred    = self.likelihood(self.model(z_query))
            lower, upper = pred.confidence_region() #2 standard deviations above and below the mean

        mse = self.mse(pred.mean, y_all[n])

        return mse

    def save_checkpoint(self, checkpoint):
        # save state
        gp_state_dict         = self.model.state_dict()
        likelihood_state_dict = self.likelihood.state_dict()
        nn_state_dict         = self.feature_extractor.state_dict()
        torch.save({'gp': gp_state_dict, 'likelihood': likelihood_state_dict, 'net':nn_state_dict}, checkpoint)

    def load_checkpoint(self, checkpoint):
        ckpt = torch.load(checkpoint)
        self.model.load_state_dict(ckpt['gp'])
        self.likelihood.load_state_dict(ckpt['likelihood'])
        self.feature_extractor.load_state_dict(ckpt['net'])

class ExactGPLayer(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel='linear', dataset="QMUL"):
        super(ExactGPLayer, self).__init__(train_x, train_y, likelihood)
        self.mean_module  = gpytorch.means.ConstantMean()

## Full
         ## Linear kernel
        if(kernel=='linear'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
        ## RBF kernel
        elif(kernel=='rbf' or kernel=='RBF'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        ## Matern kernel
        elif(kernel=='matern'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())
        ## Polynomial (p=1)
        elif(kernel=='poli1'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PolynomialKernel(power=1))
        ## Polynomial (p=2)
        elif(kernel=='poli2'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PolynomialKernel(power=2))
        ## RQKernel 
        elif(kernel=='rq'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel())
        ## Periodic kernel
        elif(kernel=='preio'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel())

## Partial :
        #Cosine kernel
        elif(kernel=='cosi'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.CosineKernel())
        ## PiecewisePolynomial Kernel
        elif(kernel=='piece'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PiecewisePolynomialKernel())


## Try Again :
        ## Spectral kernel
        elif(kernel=='spectral'):
            self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, ard_num_dims=2916)
        elif(kernel=='test'):
            self.covar_module =  gpytorch.kernels.ScaleKernel(gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, ard_num_dims=2916))
        elif(kernel=='test2'):
            self.covar_module =  gpytorch.kernels.ScaleKernel(gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, ard_num_dims=2916))
        ## Cylindrical Kernel
        elif(kernel=='cylin'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.CylindricalKernel())
        
        ## SpectralDeltaKernel 
        elif(kernel=='specdel'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.SpectralDeltaKernel())
        ## AdditiveKernel 
        elif(kernel=='add'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.AdditiveKernel())

        elif(kernel=='gencheb'):
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            self.covar_module =gpytorch.kernels.ScaleKernel(gencheb())
        else:
            raise ValueError("[ERROR] the kernel '" + str(kernel) + "' is not supported!")


    def forward(self, x):
        mean_x  = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)