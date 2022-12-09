from cmath import inf
import numpy as np
import hdf5storage
# import scipy.io as sio
from torchkbnufft import KbNufftAdjoint
import torch
from torch.utils.data import Dataset
import os
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class MRI_dataset(Dataset):
    def __init__(self, ds, gtcs, fnames):
        self.ds = ds
        self.gtcs = gtcs
        self.fnames = fnames
        
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx], self.gtcs[idx], self.fnames[idx]
    
"""
Custom Loss function
"""
def rescale(data: torch.Tensor, target: torch.Tensor):
    # preparation:
    origshape = data.shape # save for future use
    batch_size = origshape[0] # batch size
    data = data.reshape((batch_size, -1))
    # scaled = data.clone() # output variable
    scaleshift = torch.zeros((batch_size, 2), device=data.device) # scale + shift for each data in batch
    
    # computation:
    A = torch.abs(target.reshape((batch_size,-1)))
    B = torch.abs(data)
    for b in range(batch_size):
        AA = torch.tensor([[torch.sum(torch.square(B[b,:])), torch.sum(B[b,:])],
                           [torch.sum(B[b,:]), torch.numel(B[b,:])]])
        bb = torch.tensor([torch.sum(A[b,:]*B[b,:]), torch.sum(A[b,:])])
        scaleshift[b, :] = torch.linalg.solve(AA, bb)
    scaled = scaleshift[:, 0].unsqueeze(1)*data + scaleshift[:, 1].unsqueeze(1)
        
    # output:
    scaled = scaled.reshape(origshape)
    return scaled, scaleshift[:,0], scaleshift[:,1]

class MSE_loss():
    def __init__(self, scaled=False):
        self.scaled = scaled

    def __call__(self, data: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.scaled:
            data, _, _ = rescale(data, target)
        loss = torch.linalg.norm(data-target)**2 / torch.numel(data)
        return loss

"""
Calculate the density compensation function
"""
def DCF(kx, ky, Nsamples):
    # get rid of the repeated k-space locations, such as (0,0)
    K = np.column_stack((kx, ky))
    K, indices, counts = np.unique(K, axis=0, return_inverse=True, return_counts=True)

    # compute the Voronoi tessellation
    vor = Voronoi(K, qhull_options='Qbb')

    # compute the areas
    areas = np.zeros(len(K))
    for j in range(len(K)):
        # take out the infinity point (denoted by -1)
        if -1 in vor.regions[j]:
            vor.regions[j].remove(-1)
        # if there are at least 3 vertices left, compute the area
        if len(vor.regions[j]) > 2:
            x = vor.vertices[vor.regions[j],0]
            y = vor.vertices[vor.regions[j],1]
            ind = [*range(1,len(vor.regions[j])), 0]
            areas[j] = np.absolute(np.sum(0.5*(x * y[ind] - x[ind] * y)))

    # reorder the areas such that it corresponds to the order of the points in K
    areas = areas[np.array(vor.point_region)]

    # if the region corresponds to more input points, devide the area
    for j in range(len(K)):
        if counts[j] > 1:
            areas[j] = areas[j]/counts[j]

    # reorder once more (putting the repeated values back)
    areas = areas[indices]
    
    # do dirty hack to solve the boundary problem (works only for radial projections)
    for i in range(0, len(areas), Nsamples):
        tmp = areas[i:i+Nsamples]
        tmp[tmp > 2*np.median(tmp)] = 2*np.median(tmp)
        areas[i:i+Nsamples] = tmp

    return areas

"""
Reshape with fortran-like ordering
"""
def reshape_fortran(x, shape):
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))
 
"""   
Normalize sensitivities
"""
def normSens(C, factor=1):
    # read the original sizes
    if(C.ndim > 2):
        Y, X, Ncoils = C.shape
    else:
        C = torch.unsqueeze(C,2)
        Y, X, Ncoils = C.shape
        
    # elementiwise normalization of the sensitivities
    norms = torch.sqrt(torch.sum(torch.real(torch.conj(C)*C), axis=2))
    for c in range(Ncoils):
        C[:, :, c] = C[:, :, c] / norms
        
    # permuting the dimensions
    C = torch.permute(C, (2, 0, 1))
    
    # unsqueezing to have a time dimension
    C = torch.unsqueeze(C, 3)
    
    return C, norms

"""
Prepare and load all the data
"""
# we have large RAM we can preload all the data into the memory and prepare them, to speed up the training
def load_data(datafolder, subfold="train", cart=True, device=torch.device("cuda"), Nf=inf, normalize=True, rpf=None, subsample=1):

    # set the subfolder structure
    echofold = datafolder + "/" + subfold + "/" 
    
    # set the echo files
    files = os.listdir(echofold)
    files = [f for f in files if "aif" not in f]
    
    """
    Load data, trajectories and ground truth image sequence
    """
    # trajectory the same !!!!
    mat = hdf5storage.loadmat(echofold + files[0])
    d = torch.from_numpy(np.array(mat["EchoSignals"])).type(torch.cfloat)
    k_traj = torch.from_numpy(np.array(mat["k_traj"])).type(torch.float32)
    
    # add coil dimension if only one coil is simulated
    if len(d.shape) < 3:
        d = torch.unsqueeze(d, 2)
    Ncoils = d.shape[2]
    
    # load ground truth image sequence
    imgs = torch.from_numpy(np.array(mat["imgs"])).type(torch.cfloat)
    X, Y = imgs.shape[:2]
    Nsamples = d.shape[0]
    Nf = min(Nf, imgs.shape[2])

    # shorten the input data to equal maximum radials per frame based on number of frames and subsample
    Nrad = d.shape[1] # number of all the acquired radials
    if rpf is None:
        rpf = int(Nrad/Nf) # number of radials per frame in the original sequence
    while Nrad < Nf*rpf:
        Nf = Nf - 1
    Nrad_new = int(int(rpf*Nf/subsample)/Nf)*Nf

    # indices of the chosen radials
    radind = np.round(np.linspace(0,Nf*rpf-subsample,Nrad_new)).astype(int)
    
    # indices of the chosen k-space samples
    sampind = np.zeros(len(radind)*Nsamples, dtype=int)
    for i in range(len(radind)):
        sampind[i*Nsamples:(i+1)*Nsamples] = np.arange(Nsamples)+radind[i]*Nsamples

    # select only the trajectories that are not ommited
    k_traj = k_traj[:,sampind]
    Nrad = Nrad_new
   
    # prepare the k-space trajectories
    k_traj = k_traj[:-1, :] # we dont know 3D, so we ignore the last dimension
    k_traj = reshape_fortran(k_traj,(2, int(k_traj.shape[1]/Nf), Nf))
    k_traj = k_traj.contiguous().to(device)

    """
    Calculate the density compensation
    """
    print("Calculating density")
    w = np.zeros((1, k_traj.shape[1], Nf))
    k_traj_temp = k_traj.cpu().detach().numpy()
    k_traj_temp = k_traj_temp.astype(np.float32)
    for f in range(Nf):
        w[0,:,f] = np.transpose(DCF(k_traj_temp[0,:,f], k_traj_temp[1,:,f], Nsamples))
    w = torch.from_numpy(w).to(device)   
    w = w.unsqueeze(0)
    
    # define the adjoint operator
    if cart:
        Operator_adj = KbNufftAdjoint((Y, X), (int(np.floor(2.0*Y)), int(np.floor(2.0*X))), dtype=torch.cfloat, numpoints=(8, 8), device=device)

    """ 
    Load data
    """
    ds = []
    gtcs = []

    simple = torch.zeros((Ncoils, Y, X, Nf), dtype=torch.cfloat, device=device)
    
    for i in range(len(files)):
        mat = hdf5storage.loadmat(echofold + files[i])
        d = np.ascontiguousarray(np.array(mat["EchoSignals"]))
        if len(d.shape) < 3:
            d = np.expand_dims(d,2)
        d = d[:,radind,:]

        # groud truth (not scaled by the sensitivities)
        imgs = torch.from_numpy(np.ascontiguousarray(np.array(mat["imgs"]))).type(torch.cfloat)
        imgs = imgs.contiguous(memory_format=torch.contiguous_format).to(device)
        
        # load the sensitivities
        Sensitivities = torch.from_numpy(np.ascontiguousarray(np.array(mat["Sensitivities"]))).type(torch.cfloat)
        Sensitivities = Sensitivities.contiguous(memory_format=torch.contiguous_format).to(device)
        C, norms = normSens(Sensitivities, 1)
        imgs = imgs * norms.unsqueeze(2)
        
        # crop the sequences
        imgs = imgs[:, :, :Nf] 
        
        # reshape data
        d = reshape_fortran(torch.from_numpy(d).type(torch.cfloat), (int(d.shape[1]*d.shape[0]/Nf), Nf, Ncoils))
        d = d.contiguous(memory_format=torch.contiguous_format).to(device)
        d = torch.permute(d, (2, 0, 1))
        d = d.unsqueeze(0)
        if cart:
            print(f"Adjoint NUFFT for sequence {i+1} of {len(files)}")
            for f in range(Nf):
                simple[:,:,:,f] = Operator_adj(d[:,:,:,f]*w[:,:,:,f], k_traj[:,:,f])

            # simple=simple/torch.norm(simple)*torch.norm(imgs) ???? leave or delete?
            
            d = torch.fft.fftn(simple, dim=(1, 2), norm="ortho")
            
        gtcs.append(imgs.unsqueeze(0))
        ds.append(d)
    
    return MRI_dataset(ds, gtcs, files), C, k_traj, w
 
"""
Plot learning results from file
"""
def plotfile(modelfile, outputfile, verbose=True):

    if verbose:
        print("Loading the model")
    model = torch.load(modelfile)
    if verbose:
        print("Loading the outputs")
    loaded = hdf5storage.loadmat(outputfile) # TENTO KROK ZDRŽUJE!!!
    params = loaded["params"]
    # params = {name:np.squeeze(params[name][0,0]) for name in params.dtype.names}
    loss = loaded["loss"]
    
    # ======================
    # plotting loss function
    # ======================
    if verbose:
        print("Plotting loss function")
    plt.figure(figsize=(18, 12))
    arloss = np.array(loss)
    for b in range(arloss.shape[1]):
        plt.plot(np.arange(1, arloss.shape[0]),
                 np.array(loss)[1:, b],
                 label="loss in batch " + str(b))
    plt.plot(np.arange(1, arloss.shape[0]),
             np.mean(loss, axis=1)[1:],
             color="black", linewidth=3, label="mean loss")
    plt.yscale("log")
    plt.legend(loc="upper right")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("training")

    # ===========================
    # plotting all the parameters
    # ===========================
    if verbose:
        print("Plotting all the parameters")
    if model.tied:
        fig, ax = plt.subplots(len(params),
                               model.n_layers,
                               figsize=(60, 20),
                               constrained_layout=True)
    else:
        fig, ax = plt.subplots(round(len(params)/model.n_layers),
                               model.n_layers,
                               figsize=(60, 20),
                               constrained_layout=True)
    
    for col in range(model.n_layers):
        row = 0
        for key, arparams in params.items():
            arparams = np.array(arparams)
            if model.tied or "." + str(col) + "." in key or key.endswith("." + str(col)):
                if len(arparams.shape) > 1:
                    for j in range(arparams.shape[1]):
                        ax[row,col].plot(arparams[:,j], label=str(j))
                    ax[row,col].legend()
                else:
                    ax[row,col].plot(arparams)
                ax[row,col].set_title(key)
                ax[row,col].set_xlabel("epoch")
                row = row + 1
    fig.supxlabel("layer")
    fig.suptitle("Learning all the parameters", fontsize=16)
    
    # ======================
    # plotting learned proxs
    # ======================
    if verbose:
        print("Plotting learned proxs")
    fig, ax = plt.subplots(2, model.n_layers, figsize=(18,6), constrained_layout=True, sharey="row")
    maxbL = 1.00
    maxbS = 1.00
    xL = torch.linspace(0, maxbL, 1000, device=model.device)
    xS = torch.linspace(0, maxbS, 1000, device=model.device)
    for i in range(model.n_layers):
        if model.tied:
            val1 = model.proxL(torch.complex(xL, torch.tensor(0.0, device=model.device)))
            val2 = model.proxS(torch.complex(xS, torch.tensor(0.0, device=model.device)))
            ax[0,i].plot(xL.cpu(), torch.abs(val1).cpu())
            ax[1,i].plot(xS.cpu(), torch.abs(val2).cpu())
        else:
            val1 = model.proxL[i](torch.complex(xL, torch.tensor(0.0, device=model.device)))
            val2 = model.proxS[i](torch.complex(xS, torch.tensor(0.0, device=model.device)))
            ax[0,i].plot(xL.cpu(), torch.abs(val1).cpu())
            ax[1,i].plot(xS.cpu(), torch.abs(val2).cpu())
    ax[0,0].set_ylabel("prox L")
    ax[1,0].set_ylabel("prox S")
    fig.supxlabel("layer")
    fig.suptitle("Learned proxes (magnitude)", fontsize=16)
