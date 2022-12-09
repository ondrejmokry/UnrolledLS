import numpy as np
import matplotlib.pyplot as plt
import h5py
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

f = open("../../datafold.txt", "r")
datafold = f.read()
f.close()

# load the file
with h5py.File(filedialog.askopenfilename(initialdir=datafold + "/reconstruction"), "r") as hf:
    gt_ref = hf["gt_ref"][:]
    gtc_ref = hf["gtc_ref"][:]
    simple_ref = hf["simple_ref"][:]
    simple = hf["simple"][:]
    solution_ref = hf["solution_ref"][:]
    solution = hf["solution"][:]
    timeaxis = hf["timeaxis"][:]

# choose the frame to plot
Nf = solution.shape[2]
plotframe = np.random.randint(Nf)

# enable interactive mode
plt.ion()


"""
Plot the (reference) images, prepare the plot for perfusion curves
"""
figure3 = plt.figure(constrained_layout=True)
gs = figure3.add_gridspec(3, 3)

a1 = figure3.add_subplot(gs[0, 0])
a1.set_title("ground truth")
im = a1.imshow(np.abs(gt_ref[:,:,plotframe]))
figure3.colorbar(im, ax=a1)

a2 = figure3.add_subplot(gs[0, 1])
a2.set_title("INUFFT")
im = a2.imshow(np.abs(simple[:,:,plotframe]))
figure3.colorbar(im, ax=a2)

a3 = figure3.add_subplot(gs[0, 2])
a3.set_title("Chambolle-Pock")
im = a3.imshow(np.abs(solution[:,:,plotframe]))
figure3.colorbar(im, ax=a3)

a4 = figure3.add_subplot(gs[1, 0])
a4.set_title("ground truth scaled")
im = a4.imshow(np.abs(gtc_ref[:,:,plotframe]))
figure3.colorbar(im, ax=a4)

a5 = figure3.add_subplot(gs[1, 1])
a5.set_title("reference INUFFT")
im = a5.imshow(np.abs(simple_ref[:,:,plotframe]))
figure3.colorbar(im, ax=a5)

a6 = figure3.add_subplot(gs[1, 2])
a6.set_title("reference Chambolle-Pock")
im = a6.imshow(np.abs(solution_ref[:,:,plotframe]))
figure3.colorbar(im, ax=a6)

a7 = figure3.add_subplot(gs[2, :]) # this will be filled based on mouse click

figure3.suptitle("frame " + str(plotframe+1) + " of " + str(Nf))

axes3 = [a1, a2, a3, a4, a5, a6, a7]
for ax in axes3[:6]:
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

"""
Prepare the plot for the differences
"""
figure4, axes4 = plt.subplots(2,2) # this will be filled based on mouse click

"""
Plot data based on mouse click
"""
while True:
    # set the figure with solution, get input
    plt.figure(figure3.number)
    coord = plt.ginput(timeout=-1)
    x = round(coord[0][0])
    y = round(coord[0][1])

    # set and clear the perfusions
    plt.sca(axes3[6])
    plt.cla()

    # set and clear the differences
    plt.sca(axes4[0,0])
    plt.cla()
    plt.sca(axes4[0,1])
    plt.cla()
    plt.sca(axes4[1,0])
    plt.cla()
    plt.sca(axes4[1,1])
    plt.cla()

    # plot perfusions
    axes3[6].set_title("perfusion curves at [" + str(x) + ", " + str(y) + "]")
    axes3[6].plot(timeaxis, np.abs(gtc_ref[y,x,:Nf]), label="ground truth scaled")
    axes3[6].plot(timeaxis, np.abs(simple[y,x,:Nf]), label="INUFFT")
    axes3[6].plot(timeaxis, np.abs(solution[y,x,:Nf]), label="Chambolle-Pock")
    axes3[6].plot(timeaxis, np.abs(simple_ref[y,x,:Nf]), label="reference INUFFT")
    axes3[6].plot(timeaxis, np.abs(solution_ref[y,x,:Nf]), label="reference Chambolle-Pock")
    axes3[6].set_xlabel("time (s)")
    axes3[6].legend()

    # plot differences
    axes4[0,0].plot(timeaxis, np.abs(simple[y,x,:Nf] - simple_ref[y,x,:Nf]))
    axes4[0,0].set_title("absolute inverse NUFFT difference")
    axes4[0,0].set_xlabel("time (s)")
    axes4[0,1].plot(timeaxis, np.abs(solution[y,x,:Nf] - solution_ref[y,x,:Nf]))
    axes4[0,1].set_title("absolute solution difference")
    axes4[0,1].set_xlabel("time (s)")
    axes4[1,0].plot(timeaxis, np.abs(simple[y,x,:Nf] - simple_ref[y,x,:Nf])/np.abs(simple_ref[y,x,:Nf]))
    axes4[1,0].set_title("relative inverse NUFFT difference")
    axes4[1,0].set_xlabel("time (s)")
    axes4[1,1].plot(timeaxis, np.abs(solution[y,x,:Nf] - solution_ref[y,x,:Nf])/np.abs(solution_ref[y,x,:Nf]))
    axes4[1,1].set_title("relative solution difference")
    axes4[1,1].set_xlabel("time (s)")

plt.show(block=True)