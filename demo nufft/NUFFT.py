import numpy as np
import matplotlib.pyplot as plt
from pynufft import NUFFT
from DCF import DCF
from PIL import Image
import scipy.io as sio
import matlab.engine

eng = matlab.engine.start_matlab()

f = open("../datafold.txt", "r")
datafold = f.read()
f.close()

"""
Load data, set parameters
"""
mat = sio.loadmat(datafold + "/nufft_reference.mat")
Nsamples = int(mat["Nsamples"]) # number of acquired samples per each radial line
X = int(mat["X"]) # image width
Y = int(mat["Y"]) # image height
rads = int(mat["rads"]) # number of radial lines
ref_im = np.array(mat["I"]) # reference image (subsampled in matlab)
ref_im_2 = np.array(mat["recI"]) # reference reconstructed image
ref_data = np.array(mat["nufftdata"]) # reference NUFFT data
ref_w = np.array(mat["w"]) # weighting vector

"""
Functions
"""
def loadpanda(height,width):
    """
    Load the image, convert to grayscale, scale
    """
    margW = round(width/4)
    margH = round(height/4)
    im = Image.open(datafold + "/panda.png")
    im = im.convert("LA")
    im = im.resize((width-margW, height-margH), resample=Image.BICUBIC)
    part_array = np.asarray(im)
    part_array = part_array[:,:,0] # get rid of the transparency layer
    im_array = np.zeros((height,width))
    im_array[round(margH/2):height-round(margH/2), round(margW/2):width-round(margW/2)] = part_array.astype("float32") / 255.0
    return im_array

def createNUFFT(engine, k, w, shift, imsize):
    """
    Create cell array of NUFFT operators within Matlab workspace attached to the engine
    """
    engine.workspace["k"] = matlab.double(k.tolist(),is_complex=True)
    engine.workspace["w"] = matlab.double(w.tolist())
    engine.workspace["shift"] = matlab.double(shift)
    engine.workspace["imSize"] = matlab.double(imsize)
    engine.createNUFFT(nargout=0)

def mNUFFT(engine, frame, I):
    engine.workspace["I"] = matlab.double(I.tolist())
    engine.workspace["f"] = frame + 1
    return np.array(engine.eval("FFT{f}*I"),dtype=np.complex_)

def mINUFFT(engine, frame, data):
    engine.workspace["data"] = matlab.double(data.tolist(),is_complex=True)
    engine.workspace["f"] = frame + 1
    return np.array(engine.eval("FFT{f}'*data"),dtype=np.complex_)

"""
Load panda, prepare the figure
"""
im = loadpanda(Y, X)
fig1, ax = plt.subplots(1,3)
img = ax[0].imshow(im)
fig1.colorbar(img, ax=ax[0])
ax[0].set_title("python")
ax[0].get_xaxis().set_visible(False)
ax[0].get_yaxis().set_visible(False)

img = ax[1].imshow(ref_im)
fig1.colorbar(img, ax=ax[1])
ax[1].set_title("matlab")
ax[1].get_xaxis().set_visible(False)
ax[1].get_yaxis().set_visible(False)

img = ax[2].imshow(np.abs(ref_im-im))
fig1.colorbar(img, ax=ax[2])
ax[2].set_title("difference")
ax[2].get_xaxis().set_visible(False)
ax[2].get_yaxis().set_visible(False)

fig1.canvas.set_window_title("Panda")

"""
Simualte the acquisition
"""
forscale = 1 / np.sqrt(X*Y) # sqrt(w) * forscale * FFT[].forward() = NUFFT()
adjscale = 1.5**2 * np.sqrt(X*Y) # adjscale * FFT[].adjoint(sqrt(w)*()) = NUFFT'()

# acquisition angles
golden = 2*np.pi/(1+np.sqrt(5))
angles = np.arange(rads)*golden
angles = np.reshape(angles,(-1,1))

# these are the acquired locations in the k-space w.r.t. the interval [-0.5, 0.5]
kx = np.transpose(np.arange(-Nsamples/2,Nsamples/2)) * np.sin(angles)
ky = np.transpose(np.arange(-Nsamples/2,Nsamples/2)) * np.cos(angles)
kx = np.reshape(kx,(-1,1))/X
ky = np.reshape(ky,(-1,1))/Y
om = np.column_stack((kx,ky))

# rescale such that the locations are w.r.t. the interval [-pi, pi]
# (needed for pynufft)
om = om*2*np.pi

# plot the k-space locations
fig2= plt.figure("Acquired k-space locations")
plt.plot(om[:,0], om[:,1],'.')
plt.xlabel("real part")
plt.ylabel("imaginary part")

# set-up the NUFFT operator
op = NUFFT()
op.plan(om, (Y,X), (int(np.floor(1.5*Y)),int(np.floor(1.5*X))), (6, 6))

# compute the density compensation
w = DCF(kx, ky)*X*Y

# compute the forward transform (using the reference image)
kdata = np.multiply(np.sqrt(w), op.forward(ref_im)) * forscale

"""
Compute the reference using the Matlab engine
"""
createNUFFT(eng, kx + 1j*ky, w, [0, 0], [Y, X])
mdata = mNUFFT(eng, 0, ref_im)

"""
Compare the weighting vectors
"""
fig3, ax = plt.subplots(3,1)
fig3.canvas.set_window_title("NUFFT weights")

ax[0].plot(w)
ax[0].set_title("python")

ax[1].plot(ref_w[:,0])
ax[1].set_title("matlab")

ax[2].plot(w - ref_w[:,0])
ax[2].set_title("difference")

"""
Compare with the reference NUFFT data
"""
fig4, ax = plt.subplots(5,1)
fig4.canvas.set_window_title("NUFFT data")

ax[0].plot(np.abs(ref_data[:,0]))
ax[0].set_title("matlab")

ax[1].plot(np.abs(kdata))
ax[1].set_title("python")

ax[2].plot(np.abs(mdata))
ax[2].set_title("matlab via python")

ax[3].plot(np.abs(ref_data[:,0] - kdata))
ax[3].set_title("difference (python)")

ax[4].plot(np.abs(ref_data[:,0] - mdata[:,0]))
ax[4].set_title("difference (matlab via python)")

"""
Simulate the reconstruction (from the matlab-generated data)
"""
# compute the backward transform
im_2 = op.adjoint(np.multiply(ref_data[:,0], np.sqrt(w))) * adjscale

# compute the backwards transform via matlab engine
im_3 = mINUFFT(eng, 0, ref_data[:,0])

# plot the reconstructed image
fig5, ax = plt.subplots(1,5)
fig5.canvas.set_window_title("Panda reconstruction")

img = ax[0].imshow(np.abs(ref_im_2))
fig5.colorbar(img, ax=ax[0])
ax[0].set_title("matlab")
ax[0].get_xaxis().set_visible(False)
ax[0].get_yaxis().set_visible(False)

img = ax[1].imshow(np.abs(im_2))
fig5.colorbar(img, ax=ax[1])
ax[1].set_title("python")
ax[1].get_xaxis().set_visible(False)
ax[1].get_yaxis().set_visible(False)

img = ax[2].imshow(np.abs(im_3))
fig5.colorbar(img, ax=ax[2])
ax[2].set_title("matlab via python")
ax[2].get_xaxis().set_visible(False)
ax[2].get_yaxis().set_visible(False)

img = ax[3].imshow(np.abs(ref_im_2 - im_2))
fig5.colorbar(img, ax=ax[3])
ax[3].set_title("difference (python)")
ax[3].get_xaxis().set_visible(False)
ax[3].get_yaxis().set_visible(False)

img = ax[4].imshow(np.abs(ref_im_2 - im_3))
fig5.colorbar(img, ax=ax[4])
ax[4].set_title("difference (matlab via python)")
ax[4].get_xaxis().set_visible(False)
ax[4].get_yaxis().set_visible(False)

"""
Show the figures
"""
plt.show()