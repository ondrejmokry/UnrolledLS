from os.path import exists
import numpy as np
import hdf5storage
import torch
import os
import json
from datetime import datetime as dt
from torch.utils.data import DataLoader

# model
from model import UFLS

# testing
from testing import test

# utils
from utils import load_data, plotfile, MSE_loss

# ugly trick to make it work
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
Global parameters
"""
if exists("../custom_config.json"):
    f = open("../custom_config.json")
else:
    f = open("../default_config.json")

parameters = json.load(f)

# use GPU if cuda device is available
useGPU = parameters["useGPU"]

# number of iterations of the Chambolle–Pock algorithm
iterations = parameters["iterations"]

# number of epochs for training
epochs = parameters["epochs"]

# share parameters across all the layers
tied = parameters["tied"]

# normalize the ground truth and results for comparison
normalize = parameters["normalize"]

# rectification method for keeping the parameters positive
constraint = parameters["constraint"]

# =============================================================================
# note that in fact, the constraint does not change the value of the parameters
# but it is applied every time the parameter is used
#
# however, when reading out the learned values of the parameters, the
# rectification is applied such that we see the values that the algorithm
# actually use
#
# variants:
#   relu .......... p -> torch.nn.functional.relu(w)
#   sqr ........... p -> torch.square(w)
#   sqrt .......... p -> torch.sqrt(torch.square(w) + 1e-8)
# =============================================================================

# batch size
batch_size = parameters["batch_size"]
# =============================================================================
# if batch_size > 1, the following parameters of the acquisition and
# reconstruction may not differ:
#   Nf ............ number of time frames
#   rpf ........... number of radials per frame
#   Nsamples ...... number of samples per radial
#
# if batch_size = 1, these parameters may differ
# =============================================================================

# radials per frame (of the fully sampled sequence)
rpf = parameters["rpf"]

# time subsampling factor
subsample = parameters["subsample"]

# components of APL
K = parameters["K"]

# learnable sigma and tau
learnst = parameters["learnst"]

# cartesian reconstruction
cart = parameters["cart"]

# activation
activation = parameters["activation"]

# datafolder
datafolder = parameters["datafold"]

# save model?
save = parameters["save"]

# regularization parameters
lambdaL = parameters["lambdaL"]
lambdaS = parameters["lambdaS"]

# scaling of the loss
msescaled = parameters["msescaled"]

"""
Setup environment for GPU/CPU usage and precompute some variables
"""
# check availability of the cuda device
if useGPU and torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

"""
Prepare the data
"""
print("================")
print("Data preparation")
print("================")
trainset, C, k_traj, w = load_data(datafolder, device=device, cart=cart, rpf=rpf, subsample=subsample)

"""
Create  dataloader
"""
trainloader = DataLoader(trainset, batch_size, shuffle=False)

"""
Create the algorithm object
"""
CP = UFLS(n_layers=iterations,
          device=device,
          tied=tied,
          constraint=constraint,
          C=C,
          K=K,
          learnst=learnst,
          activation=activation,
          initL=lambdaL,
          initS=lambdaS)

# %% Train
floss = MSE_loss(scaled=msescaled)
# floss = torch.nn.L1Loss(reduction="sum")
# floss = torch.nn.KLDivLoss(reduction="sum")

optimizer = torch.optim.Adam(CP.parameters(), lr=3e-3)
# optimizer = torch.optim.SGD(CP.parameters(),lr=1e-4)
scheduler = torch.optim.lr_scheduler.CyclicLR(
    optimizer, base_lr=2e-4, max_lr=1e-2, step_size_up=100, cycle_momentum=False)

print("========")
print("Training")
print("========")

# torch.autograd.set_detect_anomaly(True)

loss_CP_all = []
all_the_params = {p: [] for p, _ in CP.named_parameters()}

print("Epoch     |  Mean training loss")
for epoch in range(epochs):
    loss_CP = []
    for batch, (d, gtc, _) in enumerate(trainloader):
        optimizer.zero_grad()

        # perform forward step
        solution, _ = CP(d)
        if normalize:
            solution = solution / torch.norm(solution, p=torch.inf)
            gtc = gtc / torch.norm(gtc, p=torch.inf)
        loss = floss(solution, gtc)
        
        # backpropagate and update parameters
        loss.backward()
        optimizer.step()
        scheduler.step()
        grads = []
        for param in CP.parameters():
            if param.requires_grad:
                grads.append(param.grad.view(-1))
            
        # save the loss
        loss_CP.append(loss.item())

    print(f"{epoch+1:>4}/{epochs:<4} |  {np.mean(loss_CP):.3e}")
    
    # save the parameters
    loss_CP_all.append(loss_CP)

    for p, v in CP.named_parameters():
        all_the_params[p].append(v.tolist())

"""
Save the training results
"""
print("Freezing the model")
CP.freeze()
date = dt.now()
mdic = {"params":all_the_params,
        "loss":loss_CP_all}

outputname = datafolder + "/reconstruction/output_" + str(int(date.timestamp())) + ".mat"
modelname = datafolder + "/reconstruction/model_" + str(int(date.timestamp())) + ".pt"
settingsname = datafolder + "/reconstruction/settings_" + str(int(date.timestamp())) + ".json"

# save params and loss
print("Saving the learned params and traininng loss")
hdf5storage.savemat(outputname, mdic)

# save the whole model
print("Saving the model")
torch.save(CP, modelname)

# save current settings
print("Saving the current settings")      
# with open(settingsname, "w", encoding="utf-8") as f:
#     json.dump(parameters, f, ensure_ascii=False, indent=4)
with open(settingsname, "w") as f:
    json.dump(parameters, f, sort_keys=True, indent=4)

# %% Plot
# plotfile(modelname, outputname)

# %% Test
print("=======")
print("Testing")
print("=======")

test(modelname, datafolder, normalize=normalize, batch_size=batch_size, cart=cart, rpf=rpf, subsample=subsample, save=save, lambdaL=lambdaL, lambdaS=lambdaS)
if not save:
    # remove files
    os.remove(outputname)
    os.remove(modelname)
    os.remove(settingsname)

