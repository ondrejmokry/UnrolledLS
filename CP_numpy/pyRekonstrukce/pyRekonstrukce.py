import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import h5py
import matlab.engine
from pynufft import NUFFT
from PIL import Image
from DCF import DCF

f = open("../../datafold.txt", "r")
df = f.read()
f.close()

"""
Global parameters and switches
"""
rpf = 43 # radials per frame
Nf = 100 # number of frames

# regularizers (parameters of the model)
lambdaS = 0.125
lambdaL = 0.5

# use the subsampled sensitivities from Matlab?
useC = False

# use the parameters sigma and tau from Matlab?
useST = False

# use the Matlab engine to perform SVD?
useSVD = False

# use the Matlab engine to perform NUFFT?
useNUFFT = False

# use the Matlab engine to compute SOFT?
useSOFT = False

# use the Matlab engine to compute matrix multiplication after SVD?
useMULT = False

# use the Matlab engine to compute some of the Kadjy parts?
useKADJY = False

"""
Functions
"""
# (possibly complex) signum function
def sign(x):
    if np.iscomplexobj(x):
        y = np.zeros(x.shape,dtype=np.complex_)
        y[np.abs(x) > 0] = x[np.abs(x) > 0]/np.abs(x[np.abs(x) > 0])
        return y
    else:
        return np.sign(x)

# soft thresholding
def soft(x, t):
    return sign(x)*np.maximum(np.abs(x) - t, 0)

# soft thresholding using Matlab engine
def msoft(engine, x, t):
    if np.iscomplexobj(x):
        return np.array(engine.sign(matlab.double(x.tolist(),is_complex=True)),dtype=np.complex_)*np.maximum(np.abs(x) - t, 0)
    else:
        return np.array(engine.sign(matlab.double(x.tolist())))*np.maximum(np.abs(x) - t, 0)

# print the number of frame being processed
def printframe(f, Nf):
    print("   frame " + str(f+1) + " of " + str(Nf), end="\r")

# create cell array of NUFFT operators within Matlab workspace attached to the engine
def createNUFFT(engine, k, w, shift, imsize):
    engine.workspace["k"] = matlab.double(k.tolist(),is_complex=True)
    engine.workspace["w"] = matlab.double(w.tolist())
    engine.workspace["shift"] = matlab.double(shift)
    engine.workspace["imSize"] = matlab.double(imsize)
    engine.createNUFFT(nargout=0)

# compute forward NUFFT using Matlab engine
def mNUFFT(engine, frame, I):
    engine.workspace["I"] = matlab.double(I.tolist(),is_complex=True)
    engine.workspace["f"] = frame + 1
    return np.array(engine.eval("FFT{f}*I"),dtype=np.complex_)

# compute adjoint NUFFT using Matlab engine
def mINUFFT(engine, frame, data):
    engine.workspace["data"] = matlab.double(data.tolist(),is_complex=True)
    engine.workspace["f"] = frame + 1
    return np.array(engine.eval("FFT{f}'*data"),dtype=np.complex_)

"""
Load the data
"""
# name and location of the reference file to read
filename = "CP_rpf_" + str(rpf) + "_Nf_" + str(Nf)
reffold = df + "/reconstruction/"

# name and location of the data file to read
datafold = df + "/simulation/"
datafile = "SyntheticEchoes_nufft.mat"

# echoes
mat = sio.loadmat(datafold + datafile)
Angles = np.squeeze(np.array(mat["Angles"]))
EchoSignals = np.array(mat["EchoSignals"])

# sensitivities
mat = sio.loadmat(datafold + "sensitivities.mat")
Sensitivities = np.array(mat["Sensitivities"])

# reference sequences
mat = sio.loadmat(reffold + filename +  ".mat")
cropsamples = int(mat["cropsamples"])
iterations = int(mat["iterations"])
gt_ref = np.array(mat["gt"]) # ground truth
gtc_ref = np.array(mat["gtc"]) # groud truth taking the sensitivities into account
simple_ref = np.array(mat["simple"]) # straightforward inufft
solution_ref = np.array(mat["solution"]) # Chambolle-Pock
timeaxis = np.squeeze(np.array(mat["timeaxis"])) # x-values for plotting the perfusion curves

# shortening the sequences
Angles = Angles[cropsamples:]
EchoSignals = EchoSignals[:,cropsamples:,:]

# starting the Matlab engine
if useSVD or useNUFFT or useSOFT or useMULT or useKADJY:
    print("Starting the Matlab engine")
    eng = matlab.engine.start_matlab()

"""
Read the original sizes
"""
Y, X, Ncoils = Sensitivities.shape
Nsamples, N = EchoSignals.shape[:2]

"""
Subsampling and normalization of the sensitivites
"""
# subsampling factor
factor = X/gt_ref.shape[0]

# new dimensions of the frame
X = int(X/factor)
Y = int(Y/factor)

if useC:
    # use the subsampled and normalized sensitivities from Matlab
    C = np.array(mat["C"])
else:
    # subsampling of the sensitivities
    C = np.zeros((Y,X,Ncoils),dtype=np.complex_)
    for c in range(Ncoils):
        C[:,:,c] = np.array(Image.fromarray(np.real(Sensitivities[:,:,c])).resize((Y, X), resample=Image.BICUBIC))\
           + 1j*np.array(Image.fromarray(np.imag(Sensitivities[:,:,c])).resize((Y, X), resample=Image.BICUBIC))

    # elementiwise normalization of the sensitivities
    norms = np.sqrt(np.sum(np.conjugate(C)*C, axis=2))
    for c in range(Ncoils):
        C[:,:,c] = C[:,:,c] / norms

"""
Prepare the NUFFT operators
"""
if useNUFFT:
    k = np.zeros((Nsamples*rpf, Nf),dtype=np.complex_)

if not useNUFFT:
    FFT = [] # it will be a list of operators

w = np.zeros((Nsamples*rpf, Nf))

print("Defining the NUFFT")
for f in range(Nf):    
    # compute the k-space locations of the samples (taking into account the convention of DCF and NUFFT)
    fAngles = Angles[f*rpf:(f+1)*rpf]
    fAngles = np.reshape(fAngles,(-1,1))
    kx = np.transpose(np.arange(-Nsamples/2,Nsamples/2)) * np.sin(fAngles)
    ky = np.transpose(np.arange(-Nsamples/2,Nsamples/2)) * np.cos(fAngles)
    kx = np.reshape(kx,(-1,1))/X
    ky = np.reshape(ky,(-1,1))/Y

    # compute the density compensation
    w[:,f] = DCF(kx, ky)*X*Y

    # recompute the unreliable values on the edge of the Voronoi diagram
    for r in range(rpf):
        # the first value per each radial
        w[r*Nsamples,f] = 2*w[r*Nsamples+1,f] - w[r*Nsamples+2,f]

        # the last value per each radial
        w[(r+1)*Nsamples-1,f] = 2*w[(r+1)*Nsamples-2,f] - w[(r+1)*Nsamples-3,f]

    if useNUFFT:
        k[:,f] = kx[:,0] + 1j*ky[:,0]

    if not useNUFFT:
        printframe(f, Nf)

        # rescale the coordinates such that the locations are w.r.t. the interval [-pi, pi]
        # (needed for pynufft)
        om = np.column_stack((kx,ky))*2*np.pi

        # set-up the NUFFT operator
        op = NUFFT()
        op.plan(om, (Y,X), (int(np.floor(1.5*Y)),int(np.floor(1.5*X))), (6, 6))
        FFT.append(op)

if useNUFFT:
    createNUFFT(eng, k, w, [0, 0], [Y, X])


"""
Reorganize the data
"""
# the goal is to have it in an array of size length(NUFFT) x Nf * Ncoils, where length(NUFFT) = rpf * Nsamples 
d = EchoSignals[:,:rpf*Nf,:]
d = d.reshape((rpf*Nsamples,Nf,Ncoils),order="F")

# density compensation
for c in range(Ncoils):
    d[:,:,c] = d[:,:,c]*np.sqrt(w)

# data normalization
d = d/(np.sqrt(X*factor*Y*factor)*factor)

# set the scale of FFT[].forward and FFT[].adjoint, such that the results are comparable with Matlab using NUFFT
forscale = 1 / np.sqrt(X*Y) # sqrt(w) * forscale * FFT[].forward() = NUFFT()
adjscale = 1.5**2 * np.sqrt(X*Y) # adjscale * FFT[].adjoint(sqrt(w)*()) = NUFFT'()

"""
Compute and plot the inverse NUFFT of the data
"""
# compute
print("Computing the simple solution (inverse NUFFT)")
simple = np.zeros((Y,X,Nf),dtype=np.complex_)
for f in range(Nf):
    printframe(f, Nf)
    for c in range(Ncoils):
        if useNUFFT:
            simple[:,:,f] = simple[:,:,f] + np.conjugate(C[:,:,c])*mINUFFT(eng, f, d[:,f,c])
        else:
            simple[:,:,f] = simple[:,:,f] + np.conjugate(C[:,:,c])*adjscale*FFT[f].adjoint(np.sqrt(w[:,f]) * d[:,f,c])

# choose the frame to plot
plotframe = np.random.randint(Nf)

# plot the simple solution
plt.ion()
figure1, axes1 = plt.subplots(1,2)
im = axes1[0].imshow(np.abs(simple[:,:,plotframe]))
figure1.colorbar(im, ax=axes1[0])
axes1[0].set_title("inverse NUFFT")
axes1[0].get_xaxis().set_visible(False)
axes1[0].get_yaxis().set_visible(False)

# initialize the plot for the solution of Chambolle-Pock
iter_im = axes1[1].imshow(np.abs(simple[:,:,plotframe]))
figure1.colorbar(im, ax=axes1[1])
axes1[1].set_title("Chambolle-Pock")
axes1[1].get_xaxis().set_visible(False)
axes1[1].get_yaxis().set_visible(False)

iter_title = figure1.suptitle("frame " + str(plotframe+1) + " of " + str(Nf) + "\n iteration 0 of " + str(iterations))

"""
Estimate the operator norms
"""
if not useST:
    # this is done using the (normalized) power method to estimate the largest
    # eigenvalue of the operator FFT[f].adjoint(FFT[f].forward()), the square root of which is
    # the norm of FFT[f].forward() (including the weighting and scaling in all the operators)
    print("Computing the operator norms")
    normiterations = 100
    normf = np.empty((normiterations, Nf))

    for f in range(Nf):
        x = simple[:,:,f]
        printframe(f, Nf)
        for i in range(normiterations):
            if useNUFFT:
                y = mINUFFT(eng, f, mNUFFT(eng, f, x))
            else:
                y = adjscale*FFT[f].adjoint(w[:,f] * FFT[f].forward(x) * forscale)
            normf[i,f] = np.max(np.abs(y.reshape(-1,1)))
            x = y / normf[i,f]

# set-up Chambolle-Pock
theta = 1
if useST:
    sigma = mat["sigma"]
    tau = mat["tau"]
else:
    sigma = 1/np.sqrt(4*np.max(normf.reshape(-1,1)) + 4)
    tau = 1/np.sqrt(4*np.max(normf.reshape(-1,1)) + 4)

"""
The Chambolle-Pock algorithm
"""
# the variable in the co-domain of K
y1 = np.zeros(d.shape,dtype=np.complex_)
y2 = np.zeros((Y,X,Nf),dtype=np.complex_)

# the main variable in the domain of K
x_new1 = simple
x_new2 = simple

# the auxiliary variable in the domain of K
u1 = x_new1
u2 = x_new2

# K u(i)
Ku1 = np.zeros(d.shape,dtype=np.complex_)
Ku2 = np.zeros((Y,X,Nf),dtype=np.complex_)

# K* y(i+1)
Kadjy1 = np.zeros((Y,X,Nf),dtype=np.complex_)
Kadjy2 = np.zeros((Y,X,Nf),dtype=np.complex_)

# argument of prox_{sigma f*}
argf1 = np.zeros(d.shape,dtype=np.complex_)
argf2 = np.zeros((Y,X,Nf),dtype=np.complex_)

# argument of prox_{tau g}
argg1 = np.zeros((Y,X,Nf),dtype=np.complex_)
argg2 = np.zeros((Y,X,Nf),dtype=np.complex_)

# evaluate objective?
objval = True
if objval:
    objective = np.full(iterations, np.inf)
    consistency = np.full(iterations, np.inf)
    lowrank = np.full(iterations, np.inf)
    sparse = np.full(iterations, np.inf)

    figure2 = plt.figure(constrained_layout=True)
    gs = figure2.add_gridspec(3, 2)
    
    a1 = figure2.add_subplot(gs[0, 0])
    a1.set_title("consistency")
    cons, = a1.plot(consistency)
    a1.set_yscale("log")

    a2 = figure2.add_subplot(gs[1, 0])
    a2.set_title("low-rank")
    lowr, = a2.plot(lowrank)
    a2.set_yscale("log")

    a3 = figure2.add_subplot(gs[2, 0])
    a3.set_title("sparse")
    spar, = a3.plot(sparse)
    a3.set_yscale("log")

    a4 = figure2.add_subplot(gs[:, 1])
    a4.set_title("objective = consistency + low-rank + sparse")
    obje, = a4.plot(objective)
    a4.set_yscale("log")

    axes2 = [a1, a2, a3, a4]

# iterations
print("The Chambolle-Pock algorithm")
for i in range(iterations):
    print("   iteration " + str(i+1) + " of " + str(iterations), end="\r")

    # keep the solution from the previous iteration
    x_old1 = x_new1
    x_old2 = x_new2

    # precompute the argument of prox_{sigma f*}
    # argf = y(i) + sigma K u(i)
    for f in range(Nf):
        for c in range(Ncoils):
            if useNUFFT:
                Ku1[:,f,c] = mNUFFT(eng, f, C[:,:,c]*(u1[:,:,f] + u2[:,:,f])).ravel()
            else:
                Ku1[:,f,c] = np.sqrt(w[:,f])*FFT[f].forward(C[:,:,c]*(u1[:,:,f] + u2[:,:,f]))*forscale
    Ku2[:,:,:-1] = np.diff(u2,axis=2)
    Ku2[:,:,-1] = np.zeros((Y,X),dtype=np.complex_)
    argf1 = y1 + sigma*Ku1
    argf2 = y2 + sigma*Ku2

    # apply prox_{sigma f*}
    # y(i+1) = prox_{sigma f*}( argf )
    y1 = argf1 - sigma*(argf1 + d)/(1 + sigma)
    if useSOFT:
        y2 = argf2 - sigma*msoft(eng, argf2/sigma, lambdaS/sigma)
    else:
        y2 = argf2 - sigma*soft(argf2/sigma, lambdaS/sigma)

    # precompute the argument of prox_{tau g}
    # argg = x(i) - tau K* y(i+1)
    Kadjy1 = np.zeros((Y,X,Nf),dtype=np.complex_)
    for f in range(Nf):
        for c in range(Ncoils):
            if useNUFFT:
                Kadjy1[:,:,f] = Kadjy1[:,:,f] + np.conjugate(C[:,:,c])*mINUFFT(eng, f, y1[:,f,c])
            else:
                Kadjy1[:,:,f] = Kadjy1[:,:,f] + np.conjugate(C[:,:,c])*adjscale*FFT[f].adjoint(np.sqrt(w[:,f]) * y1[:,f,c])
    Kadjy2 = Kadjy1
    if useKADJY:
        # Kadjy2[:,:,0] = Kadjy2[:,:,0] - y2[:,:,0]
        eng.workspace["Kadjy2"] = matlab.double(Kadjy2.tolist(),is_complex=True)
        eng.workspace["y2"] = matlab.double(y2.tolist(),is_complex=True)
        eng.kadjy_1(nargout=0)
        Kadjy2 = np.array(eng.workspace["Kadjy2"],dtype=np.complex_)

        Kadjy2[:,:,1:-1] = Kadjy2[:,:,1:-1] - np.diff(y2[:,:,:-1],axis=2)
        # eng.kadjy_2(nargout=0)

        Kadjy2[:,:,-1] = Kadjy2[:,:,-1] + y2[:,:,-2]
        # eng.kadjy_3(nargout=0)
    else:
        Kadjy2[:,:,0] = Kadjy2[:,:,0] - y2[:,:,0]
        Kadjy2[:,:,1:-1] = Kadjy2[:,:,1:-1] - np.diff(y2[:,:,:-1],axis=2)
        Kadjy2[:,:,-1] = Kadjy2[:,:,-1] + y2[:,:,-2]

    argg1 = x_old1 - tau*Kadjy1
    argg2 = x_old2 - tau*Kadjy2

    # apply prox_{tau g}
    # x(i+1) = prox_{tau g}( argg )
    if useSVD:
        decomposed = eng.svd(matlab.double(argg1.reshape((X*Y, Nf),order="F").tolist(), is_complex=True),nargout=3)
        U = np.array(decomposed[0], dtype=complex)
        U = U[:,:Nf]
        S = np.diagonal(np.array(decomposed[1]))
        V = np.conjugate(np.array(decomposed[2], dtype=complex).T)
    else:
        U, S, V = np.linalg.svd(argg1.reshape((X*Y, Nf),order="F"),full_matrices=False)
    if useSOFT:
        S = msoft(eng, S, S[0]*tau*lambdaL).ravel()
    else:
        S = soft(S, S[0]*tau*lambdaL).ravel()
    if useMULT:
        eng.workspace["U"] = matlab.double(U.tolist(),is_complex=True)
        eng.workspace["S"] = matlab.double(S.tolist(),is_complex=True)
        eng.workspace["V"] = matlab.double(V.tolist(),is_complex=True)
        x_new1 = np.array(eng.eval("U*diag(S)*V"),dtype=np.complex_)
    else:
        x_new1 = (U * S) @ V
    x_new1 = x_new1.reshape((Y, X, Nf),order="F")
    x_new2 = argg2

    # update the auxiliary variable
    # u(i+1) = x(n+1) + theta ( x(n+1) - x(n) )
    u1 = x_new1 + theta*(x_new1 - x_old1)
    u2 = x_new2 + theta*(x_new2 - x_old2)

    # evaluate the objective function
    if objval:
        Kx = np.zeros(d.shape,dtype=np.complex_)
        for f in range(Nf):
            for c in range(Ncoils):
                if useNUFFT:
                    Kx[:,f,c] = mNUFFT(eng, f, C[:,:,c]*(x_new1[:,:,f] + x_new2[:,:,f])).ravel()
                else:
                    Kx[:,f,c] = np.sqrt(w[:,f])*FFT[f].forward(C[:,:,c]*(x_new1[:,:,f] + x_new2[:,:,f]))*forscale
        
        consistency[i] = 0.5*np.linalg.norm(d.reshape(-1,1) - Kx.reshape(-1,1))**2
        if useSVD:
            ssigma = np.array(eng.svd(matlab.double(x_new1.reshape((X*Y, Nf),order="F").tolist(), is_complex=True),nargout=1))
            ssigma = ssigma[:,0]
        else:
            ssigma = np.linalg.svd(x_new1.reshape((X*Y, Nf),order="F"),compute_uv=False)

        lowrank[i] = lambdaL*ssigma[0]*np.linalg.norm(ssigma,ord=1)
        Kx = np.zeros((Y,X,Nf),dtype=np.complex_)
        Kx[:,:,:-1] = np.diff(x_new2,axis=2);
        Kx[:,:,-1] = -x_new2[:,:,-1];

        sparse[i] = lambdaS * np.linalg.norm(Kx.reshape(-1,1),ord=1)
        objective[i] = consistency[i] + lowrank[i] + sparse[i]

    # build the solution
    solution = np.abs(x_new1 + x_new2)

    # plot the solution
    iter_im.set_data(np.abs(solution[:,:,plotframe]))
    iter_title.set_text("frame " + str(plotframe+1) + " of " + str(Nf) + "\n iteration " + str(i+1) + " of " + str(iterations))
    figure1.canvas.draw()
    figure1.canvas.flush_events()

    # plot the objective
    if objval:
        cons.set_ydata(consistency)
        lowr.set_ydata(lowrank)
        spar.set_ydata(sparse)
        obje.set_ydata(objective)
        for ax in axes2:
            ax.relim()
            ax.autoscale_view()
        figure2.canvas.draw()
        figure2.canvas.flush_events()

"""
Save the data
"""
if useC:
    filename = filename + "_useC"
if useST:
    filename = filename + "_useST"
if useSVD:
    filename = filename + "_useSVD"
if useNUFFT:
    filename = filename + "_useNUFFT"
if useSOFT:
    filename = filename + "_useSOFT"
if useMULT:
    filename = filename + "_useMULT"
if useKADJY:
    filename = filename + "_useKADJY"

with h5py.File(df + "/reconstruction/" + filename + ".h5", "w") as hf:
    hf.create_dataset("cropsamples", data=cropsamples)
    hf.create_dataset("gt_ref", data=gt_ref)
    hf.create_dataset("gtc_ref", data=gtc_ref)
    hf.create_dataset("iterations", data=iterations)
    hf.create_dataset("Nf", data=Nf)
    hf.create_dataset("rpf", data=rpf)
    hf.create_dataset("simple_ref", data=simple_ref)
    hf.create_dataset("simple", data=simple)
    hf.create_dataset("solution_ref", data=solution_ref)
    hf.create_dataset("solution", data=solution)
    hf.create_dataset("timeaxis", data=timeaxis)

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