from os.path import exists
import numpy as np
import torch
import json
import hdf5storage
from torch.utils.data import DataLoader
from datetime import datetime as dt

# model
from model import LS

# utils
from utils import load_data, MSE_loss

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

# cartesian reconstruction
cart = parameters["cart"]

# datafolder
datafolder = parameters["datafold"]

# radials per frame (of the fully sampled sequence)
rpf = parameters["rpf"]

# time subsampling factor
subsample = parameters["subsample"]

# batch size
batch_size = parameters["batch_size"]

# save?
save = parameters["save"]

# number of iterations
iterations = 40

# normalize the ground truth and results for comparison
normalize = parameters["normalize"]

"""
Setup environment for GPU/CPU usage
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
gridset, C, k_traj, w = load_data(datafolder, subfold="train", device=device, cart=cart, rpf=rpf, subsample=subsample)

"""
Create  dataloader
"""
gridloader = DataLoader(gridset, batch_size, shuffle=False)

"""
Set lambdas and iterate
"""
print("===================")
print("Searching the grid")
print("===================")
# lambdaLs = np.array((1e-4,1e-3,1e-2,1e-1,1,1.1,1.2,1.5,2,2.8))
# lambdaSs = np.array((1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,2,5,10))
lambdaLs = np.array((1e-2,1e-1,1,1.1,1.2,1.5,2,2.8,5,10,20,50,60,70,90,100,150,200,500,1000))
lambdaSs = np.array((1e-5,1e-4,1e-3,1e-2,1e-1,2e-1,3e-1,5e-1,1,5,10,15,20,50,100))

# MSE per sequence
mse = torch.zeros(len(lambdaLs), len(lambdaSs), len(gridset), device=device)
mse_scaled = torch.zeros(len(lambdaLs), len(lambdaSs), len(gridset), device=device)

# MSE per batch
# (should lead to the same mean MSE per sequence from the whole dateset)
mseb = torch.zeros(len(lambdaLs), len(lambdaSs), int(len(gridset)/batch_size), device=device)
mseb_scaled = torch.zeros(len(lambdaLs), len(lambdaSs), int(len(gridset)/batch_size), device=device)

MSE = MSE_loss(scaled=False)
MSE_scaled = MSE_loss(scaled=True)
for i in range(len(lambdaLs)):
    for j in range(len(lambdaSs)):
        print(f"lambdaL = {lambdaLs[i]} ({i+1}/{len(lambdaLs)}), lamndaS = {lambdaSs[j]} ({j+1}/{len(lambdaSs)})")
        for batch, (d, gtc, _ ) in enumerate(gridloader):
            print(f"Batch {batch+1} of {int(len(gridset)/batch_size)}")
            CP = LS(lambdaL=lambdaLs[i],
                    lambdaS=lambdaSs[j],
                    iterations=iterations,
                    C=C,
                    device=device,
                    verbose=True)
            solution, _ = CP(d)
            if normalize:
                solution = solution / torch.norm(solution, p=torch.inf)
                gtc = gtc / torch.norm(gtc, p=torch.inf)
            solution = torch.abs(solution)
            gtc = torch.abs(gtc)
            for b in range(batch_size):
                mse[i,j,batch*batch_size+b] = MSE(
                    solution[b,0,:,:,:].unsqueeze(0),
                    gtc[b,0,:,:,:].unsqueeze(0))
                mse_scaled[i,j,batch*batch_size+b] = MSE_scaled(
                    solution[b,0,:,:,:].unsqueeze(0),
                    gtc[b,0,:,:,:].unsqueeze(0))
            mseb[i,j,batch] = MSE(solution, gtc)
            mseb_scaled[i,j,batch] = MSE_scaled(solution, gtc)

mdic = {"mse":mse.cpu().numpy(),
        "mse_scaled":mse_scaled.cpu().numpy(),
        "mseb":mseb.cpu().numpy(),
        "mseb_scaled":mseb_scaled.cpu().numpy(),
        "lambdaLs":lambdaLs,
        "lambdaSs":lambdaSs,
        "iterations":iterations}

"""
Find minimum and save lambdas to json
"""
if save:
    ar = np.mean(mse_scaled.cpu().numpy(), axis=2)
    (indexL, indexS) = np.unravel_index(np.argmin(ar), ar.shape)
    parameters["lambdaL"] = lambdaLs[indexL]
    parameters["lambdaS"] = lambdaSs[indexS]
    if exists("../custom_config.json"):
        f = open("../custom_config.json", "w", encoding="utf-8")
    else:
        f = open("../default_config.json", "w", encoding="utf-8")
    json.dump(parameters, f, ensure_ascii=False, indent=4, sort_keys=True)
    f.close()

"""
Save the results
"""
if save:
    print("Saving the results")
    date = dt.now()
    hdf5storage.savemat(datafolder + "/reconstruction/grid_out_" + str(int(date.timestamp())) + ".mat", mdic)