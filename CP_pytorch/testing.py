import matplotlib.pyplot as plt
import torch
import numpy as np
import hdf5storage
import os
import shutil
from torch.utils.data import DataLoader
from utils import load_data, MSE_loss, rescale
from model import LS

def test(modelpath, datafolder, lambdaL=2e-1, lambdaS=2e-3, normalize=True, batch_size=1, cart=True, rpf=128, subsample=1, save=False):
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    if save:
        dirname = datafolder + "/reco_" + modelpath[-13:-3]
        if not os.path.exists(dirname):
            os.mkdir(dirname)
    
    """
    Load the data
    """
    testset, C, k_traj, w = load_data(datafolder, subfold="test", device=device, cart=cart, rpf=rpf, subsample=subsample)
    testloader = DataLoader(testset, batch_size, shuffle=False)

    """
    Create the objects
    """
    # unfolded L+S
    unfolded = torch.load(modelpath)
    unfolded.freeze()
    
    # repeated unfolded L+S)
    second_unfolded = torch.load(modelpath)
    second_unfolded.freeze()
    second_unfolded.n_layers = 100*unfolded.n_layers
    
    # classical L+S
    classical = LS(lambdaL=lambdaL, # this is from the grid search
                   lambdaS=lambdaS, # this is from the grid search
                   iterations=unfolded.n_layers,
                   objval=True,
                   device=device,
                   C=unfolded.C,
                   fft=unfolded.fft,
                   verbose=False,
                   scalesing=unfolded.scalesing)
    
    # second baseline
    """
    second_baseline = LS(lambdaL=unfolded.proxL.b.item()/0.3536, # this is what the network learned
                         lambdaS=unfolded.proxS.b.item(), # this is what the network learned
                         iterations=unfolded.n_layers,
                         objval=True,
                         device=device,
                         C=unfolded.C,
                         fft=unfolded.fft,
                         verbose=False,
                         scalesing=unfolded.scalesing)
    """
    second_baseline = LS(lambdaL=lambdaL, # this is from the grid search
                         lambdaS=lambdaS, # this is from the grid search
                         iterations=(second_unfolded.n_layers-110),
                         objval=True,
                         device=device,
                         C=unfolded.C,
                         fft=unfolded.fft,
                         verbose=False,
                         scalesing=unfolded.scalesing)
   
    floss_scaled = MSE_loss(scaled=True)
    floss = MSE_loss(scaled=False)
    
    for batch, (d, gtc, fname) in enumerate(testloader):
        print(f"Comparing with baselines, batch {batch+1} of {int(len(testset)/batch_size)}")
        
        plotframe = 100
        Nf = d.size(4)
        batch_size = d.size(0)
        gtc = gtc.to("cpu")
        
        # inverse NUFFT
        simple = torch.conj(C.to(device))*torch.fft.ifftn(d, dim=(2, 3), norm="ortho") # coil multiplication?
        simple = torch.sum(simple, axis=1, keepdim=True)
        simple = simple.to("cpu")
        
        # classical L+S
        baseline, progress_b, consistency, lowrank, sparse, objective = classical(d, target=gtc.to(device), metric=floss)
        baseline = baseline.to("cpu")
        
        # second baseline (not implemented yet)
        baseline2, progress_b2, consistency2, lowrank2, sparse2, objective2 = second_baseline(d, target=gtc.to(device), metric=floss)
        baseline2 = baseline2.to("cpu")
        
        # unfolded L+S
        solution, progress_s = unfolded(d, target=gtc.to(device), metric=floss)
        solution = solution.to("cpu")
        
        # second unfolded L+S
        solution2, progress_s2 = second_unfolded(d, target=gtc.to(device), metric=floss)
        solution2 = solution2.to("cpu")
        
        # normalize
        if normalize:
            gtc = gtc / torch.norm(gtc, p=torch.inf)
            simple = simple / torch.norm(simple, p=torch.inf)
            baseline = baseline / torch.norm(baseline, p=torch.inf)
            baseline2 = baseline2 / torch.norm(baseline2, p=torch.inf)
            solution = solution / torch.norm(solution, p=torch.inf)
            solution2 = solution2 / torch.norm(solution2, p=torch.inf)

            
        # make everything real by taking the absolute value
        gtc = torch.abs(gtc) # to be sure
        simple = torch.abs(simple)
        baseline = torch.abs(baseline)
        baseline2 = torch.abs(baseline2)
        solution = torch.abs(solution)
        solution2 = torch.abs(solution2)
        
        # save
        if save:
            print("Saving the sequences")
            data2 = [gtc, simple, baseline, baseline2, solution, solution2]
            methods = ["gtc", "simple", "CP", "CP2", "unfolded", "reunfolded"]
            for i in range(len(data2)):
                data2[i], _, _ = rescale(data2[i], gtc)
                for b in range(batch_size):
                    # duplicate the data file
                    oldname = datafolder + "/test/" + fname[b]
                    newname = dirname + "/" + fname[b][:-4] + "_" + methods[i] + ".mat"
                    shutil.copyfile(oldname, newname)
                    
                    # duplicate the aif file
                    # shutil.copyfile(oldname.replace("SyntheticEchoes","aif"), newname.replace("SyntheticEchoes","aif"))
                    
                    # create cell array from sequence
                    cell = np.empty((1, Nf), dtype=object)
                    for f in range(Nf):
                        cell[0, f] = torch.squeeze(data2[i][b,:,:,:,f]).numpy()
                        
                    # save
                    hdf5storage.savemat(newname, {"data2": cell})
        
        """
        Plot
        """
        # =============
        # example frame
        # =============
        fig1 = plt.figure(constrained_layout=True,figsize=(18,5*batch_size))
        gs = fig1.add_gridspec(batch_size, 6)
        
        for b in range(batch_size):
            a1 = fig1.add_subplot(gs[b,0])
            a1.set_title(f"data {b+1}/{batch_size}\n{fname[b]}\nground truth")
            im = a1.imshow(torch.abs(gtc[b, 0, :, :, plotframe]))
            fig1.colorbar(im, ax=a1)
            
            a2 = fig1.add_subplot(gs[b,1])
            a2.set_title(f"data {b+1}/{batch_size}\n{fname[b]}\nINUFFT")
            im = a2.imshow(torch.abs(simple[b, 0, :, :, plotframe]))
            fig1.colorbar(im, ax=a2)
            
            a3 = fig1.add_subplot(gs[b,2])
            a3.set_title(f"data {b+1}/{batch_size}\n{fname[b]}\nclassical L+S")
            im = a3.imshow(torch.abs(baseline[b, 0, :, :, plotframe]))
            fig1.colorbar(im, ax=a3)
            
            a4 = fig1.add_subplot(gs[b,3])
            a4.set_title(f"data {b+1}/{batch_size}\n{fname[b]}\nsecond baseline")
            im = a4.imshow(torch.abs(baseline2[b, 0, :, :, plotframe]))
            fig1.colorbar(im, ax=a4)
            
            a5 = fig1.add_subplot(gs[b,4])
            a5.set_title(f"data {b+1}/{batch_size}\n{fname[b]}\nunfolded L+S")
            im = a5.imshow(torch.abs(solution[b, 0, :, :, plotframe]))
            fig1.colorbar(im, ax=a5)
            
            a6 = fig1.add_subplot(gs[b,5])
            a6.set_title(f"data {b+1}/{batch_size}\n{fname[b]}\nunfolded L+S with repetition")
            im = a6.imshow(torch.abs(solution2[b, 0, :, :, plotframe]))
            fig1.colorbar(im, ax=a6)
            
            for ax in [a1, a2, a3, a4, a5, a6]:
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                
        fig1.suptitle(f"reconstruction from the testing set\nbatch {batch+1} of {int(len(testset)/batch_size)}\nframe {plotframe+1} of {Nf}")
        
        # ================================
        # differences (averaged over time)
        # ================================
        fig2 = plt.figure(constrained_layout=True,figsize=(12,5*batch_size))
        gs = fig2.add_gridspec(batch_size, 5)
        
        for b in range(batch_size):
            a1 = fig2.add_subplot(gs[b, 0])
            a1.set_title(f"data {b+1}/{batch_size}\n{fname[b]}\nmean relative difference for INUFFT [%]")
            im = a1.imshow(100*torch.mean(torch.abs((simple[b, 0, :, :, :] - gtc[b, 0, :, :, :]))/gtc[b, 0, :, :, :], axis=2, keepdims=False),
                           vmin=0, vmax=100)
            fig2.colorbar(im, ax=a1)
            
            a2 = fig2.add_subplot(gs[b, 1])
            a2.set_title(f"data {b+1}/{batch_size}\n{fname[b]}\nmean relative difference for classical L+S [%]")
            im = a2.imshow(100*torch.mean(torch.abs((baseline[b, 0, :, :, :] - gtc[b, 0, :, :, :]))/gtc[b, 0, :, :, :], axis=2, keepdims=False),
                           vmin=0, vmax=100)
            fig2.colorbar(im, ax=a2)
            
            a3 = fig2.add_subplot(gs[b, 2])
            a3.set_title(f"data {b+1}/{batch_size}\n{fname[b]}\nmean relative difference for second baseline [%]")
            im = a3.imshow(100*torch.mean(torch.abs((baseline2[b, 0, :, :, :] - gtc[b, 0, :, :, :]))/gtc[b, 0, :, :, :], axis=2, keepdims=False),
                           vmin=0, vmax=100)
            fig2.colorbar(im, ax=a3)
            
            a4 = fig2.add_subplot(gs[b, 3])
            a4.set_title(f"data {b+1}/{batch_size}\n{fname[b]}\nmean relative difference for unfolded L+S [%]")
            im = a4.imshow(100*torch.mean(torch.abs((solution[b, 0, :, :, :] - gtc[b, 0, :, :, :]))/gtc[b, 0, :, :, :], axis=2, keepdims=False),
                           vmin=0, vmax=100)
            fig2.colorbar(im, ax=a4)
            
            a5 = fig2.add_subplot(gs[b, 4])
            a5.set_title(f"data {b+1}/{batch_size}\n{fname[b]}\nmean relative difference for unfolded L+S with repetition [%]")
            im = a5.imshow(100*torch.mean(torch.abs((solution2[b, 0, :, :, :] - gtc[b, 0, :, :, :]))/gtc[b, 0, :, :, :], axis=2, keepdims=False),
                           vmin=0, vmax=100)
            fig2.colorbar(im, ax=a5)
            
        fig2.suptitle(f"differences of the reconstruction and the scaled ground truth\nbatch {batch+1} of {int(len(testset)/batch_size)}")
        
        # =================================
        # differences (averaged over space)
        # =================================
        fig3 = plt.figure(figsize=(18,18))
        gs = fig3.add_gridspec(batch_size, 1)
        
        for b in range(batch_size):
            fig3.add_subplot(gs[b, 0])
            
            plt.plot(torch.mean(torch.abs(simple[b, 0, :, :, :] - gtc[b, 0, :, :, :]), axis=(0,1)),
                    label="INUFFT")
            plt.plot(torch.mean(torch.abs(baseline[b, 0, :, :, :] - gtc[b, 0, :, :, :]), axis=(0,1)),
                    label="classical L+S")
            plt.plot(torch.mean(torch.abs(baseline2[b, 0, :, :, :] - gtc[b, 0, :, :, :]), axis=(0,1)),
                    label="classical L+S")
            plt.plot(torch.mean(torch.abs(solution[b, 0, :, :, :] - gtc[b, 0, :, :, :]), axis=(0,1)),
                    label="unfolded L+S")
            plt.plot(torch.mean(torch.abs(solution2[b, 0, :, :, :] - gtc[b, 0, :, :, :]), axis=(0,1)),
                    label="unfolded L+S with repetition")
            
            plt.title(f"data {b+1}/{batch_size} ({fname[b]})")
            plt.xlabel("frame")
            plt.legend()
            
        fig3.suptitle(f"mean absolute difference of the reconstruction and the scaled ground truth\nbatch {batch+1} of {int(len(testset)/batch_size)}")

        
        # ==========================
        # objective in classical L+S
        # ==========================
        if classical.objval:
            fig4 = plt.figure(constrained_layout=True,figsize=(12,9))
            gs = fig4.add_gridspec(3, 2)
        
            a1 = fig4.add_subplot(gs[0, 0])
            a1.set_title("consistency")
            a1.set_yscale("log")
            
            a2 = fig4.add_subplot(gs[1, 0])
            a2.set_title("low-rank")
            a2.set_yscale("log")
        
            a3 = fig4.add_subplot(gs[2, 0])
            a3.set_title("sparse")
            a3.set_yscale("log")
        
            a4 = fig4.add_subplot(gs[:, 1])
            a4.set_title("objective = consistency + low-rank + sparse")
            a4.set_yscale("log")
            
            for b in range(batch_size):
                a1.plot(consistency[b,:],
                        label=f"data {b+1}/{batch_size} ({fname[b]})")
                a2.plot(lowrank[b,:],
                        label=f"data {b+1}/{batch_size} ({fname[b]})")
                a3.plot(sparse[b,:],
                        label=f"data {b+1}/{batch_size} ({fname[b]})")
                a4.plot(objective[b,:],
                        label=f"data {b+1}/{batch_size} ({fname[b]})")

            a1.legend()
            a2.legend()
            a3.legend()
            a4.legend()
            
            fig4.suptitle(f"objective function for classical L+S (Chambolle-Pock)\nbatch {batch+1} of {int(len(testset)/batch_size)}")

        # ============================
        # objective in second baseline
        # ============================
        if classical.objval:
            fig5 = plt.figure(constrained_layout=True,figsize=(12,9))
            gs = fig5.add_gridspec(3, 2)
        
            a1 = fig5.add_subplot(gs[0, 0])
            a1.set_title("consistency")
            a1.set_yscale("log")
            
            a2 = fig5.add_subplot(gs[1, 0])
            a2.set_title("low-rank")
            a2.set_yscale("log")
        
            a3 = fig5.add_subplot(gs[2, 0])
            a3.set_title("sparse")
            a3.set_yscale("log")
        
            a4 = fig5.add_subplot(gs[:, 1])
            a4.set_title("objective = consistency + low-rank + sparse")
            a4.set_yscale("log")
            
            for b in range(batch_size):
                a1.plot(consistency2[b,:],
                        label=f"data {b+1}/{batch_size} ({fname[b]})")
                a2.plot(lowrank2[b,:],
                        label=f"data {b+1}/{batch_size} ({fname[b]})")
                a3.plot(sparse2[b,:],
                        label=f"data {b+1}/{batch_size} ({fname[b]})")
                a4.plot(objective2[b,:],
                        label=f"data {b+1}/{batch_size} ({fname[b]})")

            a1.legend()
            a2.legend()
            a3.legend()
            a4.legend()
            
            fig5.suptitle(f"objective function for classical L+S (Chambolle-Pock)\nbatch {batch+1} of {int(len(testset)/batch_size)}")
            
        # ====================
        # progress towards gtc
        # ====================
        fig6 = plt.figure(figsize=(18,18))
        gs = fig6.add_gridspec(batch_size, 1)
        
        for b in range(batch_size):
            fig6.add_subplot(gs[b, 0])
            
            plt.plot(progress_b[b, :],  label="classical L+S")
            plt.plot(progress_b2[b, :], label="classical L+S")
            plt.plot(progress_s[b, :],  label="unfolded L+S")
            plt.plot(progress_s2[b, :], label="unfolded L+S with repetition")
            
            plt.title(f"data {b+1}/{batch_size} ({fname[b]})")
            plt.xlabel("iteration or layer")
            plt.ylabel("scaled MSE")
            plt.xscale("log")
            plt.yscale("log")
            plt.legend()
            
        fig6.suptitle(f"progress towards the scaled ground truth\nbatch {batch+1} of {int(len(testset)/batch_size)}")

        plt.show()
        
        """
        Write MSE to console
        """
        mses = [floss(simple, gtc).item(),
                floss(baseline, gtc).item(),
                floss(baseline2, gtc).item(),
                floss(solution, gtc).item(),
                floss(solution2, gtc).item()
                ]
        mses_scaled = [floss_scaled(simple, gtc).item(),
                floss_scaled(baseline, gtc).item(),
                floss_scaled(baseline2, gtc).item(),
                floss_scaled(solution, gtc).item(),
                floss_scaled(solution2, gtc).item()
                ]
        methods = ["simple", "classical L+S", "second baseline", "unfolded L+S", "unfolded L+S with repetition"]
        print(f"MSE in batch {batch+1} of {int(len(testset)/batch_size)}:")
        for i in range(len(mses)):
            # print(f"  {methods[i]:<18} {mses[i]:.3e}")
            print(f"  {methods[i]:<34} scaled: {mses_scaled[i]:.3e}, not scaled: {mses[i]:.3e}")