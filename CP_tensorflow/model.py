from numpy import dtype
import tensorflow as tf
import tensorflow_mri as tfmri
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from torch import tensor


class CP_model(tf.keras.Model):
    def __init__(self, kspace,w=1,C=1, n_layers=500, tied=False, constraint="relu", lambdaL_init=8e-1, lambdaS_init=1e-2,cartesian_reco=False,X=64,Y=64):
        super(CP_model, self).__init__()
        self.frozen = False
        self.n_layers = n_layers
        self.tied = tied
        self.C = tf.cast(C,dtype=tf.complex64)  # to make sure 
        self.constraint = constraint
        w=w
        self.w=w 
        self.sq_w=tf.sqrt(w)
        self.kspace=kspace
        self.cart=cartesian_reco
        self.X=tf.cast(X,dtype=tf.complex64)
        self.Y=tf.cast(Y,dtype=tf.complex64)
        if self.tied:
            n_par=1
        else:
            n_par=n_layers

        self.lambdaS=self.add_weight( shape=(n_par),initializer=tf.keras.initializers.Constant(lambdaS_init), trainable=True,dtype=tf.complex64)
        self.lambdaL=self.add_weight( shape=(n_par),initializer=tf.keras.initializers.Constant(lambdaL_init), trainable=True,dtype=tf.complex64)


    # def create_circular_mask(self, h, center=None, radius=None) -> tf.Tensor:
    #     if center is None:  # use the center of the image
    #         center = (int(h/2), int(h/2))
    #     if radius is None:  # use the smallest distance between the center and image walls
    #         radius = min(center[0], center[1], h-center[0], h-center[1])

    #     Y, X = numpy.ogrid[:h, :h]
    #     dist_from_center = numpy.sqrt((X - center[0])**2 + (Y-center[1])**2)

    #     mask = dist_from_center <= radius
    #     return tf.from_numpy(numpy.double(mask)).complex64()

            
    # replacing denormal numbers (caused problems in the sign function)
    def replace_denormals(self, x: tf.Tensor, threshold=1e-20) -> tf.Tensor:
        

        tmp=tf.cast(tf.abs(x) > threshold,tf.complex64)

        y=x*tmp
        return y

    # (possibly complex) signum function
    def sign(self, x: tf.Tensor) -> tf.Tensor:
        
        y = tf.exp(1j*tf.cast(tf.math.angle(self.replace_denormals(x)),dtype=tf.complex64))
            
        return y

    # soft thresholding
    def soft(self, x: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
        return self.sign(x)*tf.cast(tf.maximum(tf.abs(x) - tf.abs(t), 0.0),tf.complex64)


    # def soft(self,x: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
    #     return tf.where(tf.abs(x) - tf.abs(t)<0.0,tf.cast(0,dtype=tf.complex64),x)
        
    def rectify(self, w: tf.Tensor) -> tf.Tensor:
        if self.constraint == "relu":
            return tf.complex(tf.maximum(tf.math.real(w),0),tf.maximum(tf.math.imag(w),0))  # Complex Relu?
        elif self.constraint == "sqr":
            return tf.square(w)
        elif self.constraint == "sqrt":
            return tf.sqrt(tf.square(w) + 1e-8)
        else:
            return w

    def call(self, d: tf.Tensor) -> tf.Tensor:
        

        

        norm=tf.sqrt(tf.cast(tf.reduce_sum(tf.abs(d)**2),dtype=tf.complex64)) # Do the normalization so that the regularization is transferrable
        
        d=d/(norm)
        
        if self.cart:
            forscale= 1
            adjscale=1
        else:
            forscale= tf.math.reciprocal(tf.sqrt(self.X*self.Y))
            adjscale=tf.math.reciprocal(tf.sqrt(self.X*self.Y))

        if self.cart==False:
            d=d*self.sq_w # !!!!!!!!!!!!!!!!!!!!!!!
            startpoint=tfmri.signal.nufft(d*self.sq_w,grid_shape= (self.X,self.Y),
                            points=self.kspace,transform_type='type_1',fft_direction='backward')*adjscale
        else:
            startpoint=tfmri.signal.ifft(d,axes=(3,4),norm="ortho")


        batch_size, Nc,Nf, Y, X = startpoint.shape
        

        
        
        
        if self.cart==False:
            print("Computing the operator norms")
            normiterations = 50
            normf = tf.experimental.numpy.empty((normiterations, Nf),dtype=tf.complex64)
            x = startpoint
            for i in range(normiterations):
                print("Norm_iter " + str(i)+"/"+str(normiterations))
                tmp=self.w*tfmri.signal.nufft(x*forscale,grid_shape= (self.X,self.Y),
                            points=self.kspace,transform_type='type_2',fft_direction="forward")

                y = tfmri.signal.nufft(tmp,grid_shape=(self.X,self.Y),points=self.kspace,transform_type='type_1',fft_direction='backward')*adjscale

                normf= tf.cast(tf.math.reduce_max(tf.abs(y),axis=(3,4),keepdims=True),dtype=tf.complex64)
                x = y / normf
                

            normf=tf.math.reduce_max(tf.abs(normf))    
        else:
            normf=tf.cast(1,dtype=tf.complex64)


        # parameters of the algorithm
        theta = 1
        boost=1
        sigma = 1/tf.sqrt((tf.cast(4 + 4*normf,dtype=tf.complex64)))*boost
        tau = 1/tf.sqrt((tf.cast(4 + 4*normf,dtype=tf.complex64)))*boost


        # sigma_tau=tf.cast(tf.sqrt(1/normf**2)*0.95,dtype=tf.complex64)

        # sigma=sigma_tau
        # tau=sigma_tau

        # # the variable in the co-domain of K
        # y1 = tf.zeros(d.shape, dtype=tf.complex64)
        # y2 = tf.zeros(startpoint.shape, dtype=tf.complex64)
        y1=d*0
        y2=startpoint*0
        # the main variable in the domain of K
        x_new1 = startpoint
        x_new2 = startpoint

        # the auxiliary variable in the domain of K
        u1 = x_new1
        u2 = x_new2

        # # K u(i)
        # Ku1 = tf.zeros(d.shape, dtype=tf.complex64)
        # Ku2 = tf.zeros(startpoint.shape, dtype=tf.complex64)

        # # K* y(i+1)
        # Kadjy1 = tf.zeros(
        #     startpoint.shape, dtype=tf.complex64)
        # Kadjy2 = tf.zeros(
        #     startpoint.shape, dtype=tf.complex64)

        # # argument of prox_{sigma f*}
        # argf1 = tf.zeros(d.shape, dtype=tf.complex64)
        # argf2 = tf.zeros(
        #     startpoint.shape, dtype=tf.complex64)

        # # argument of prox_{tau g}
        # argg1 = tf.zeros(
        #     startpoint.shape, dtype=tf.complex64)
        # argg2 = tf.zeros(
        #     startpoint.shape, dtype=tf.complex64)

        # Ksparse = tf.zeros(
        #     startpoint.shape, dtype=tf.complex64)

        # Ku2_zer=tf.zeros((startpoint.shape[0],startpoint.shape[1],1,startpoint.shape[3],startpoint.shape[4]),dtype=tf.complex64)
        Ku2_zer=tf.reduce_sum(0*startpoint,axis=2,keepdims=True)
        for i in range(self.n_layers):
            print("CP iter " + str(i)+"/"+str(self.n_layers))
            # keep the solution from the previous iteration
            x_old1 = x_new1  # Cloning?
            x_old2 = x_new2

            # precompute the argument of prox_{sigma f*}
            # argf = y(i) + sigma K u(i)
            if self.cart==False:
                Ku1 = self.sq_w*tfmri.signal.nufft(self.C*(u1+u2)*forscale,grid_shape= (self.X,self.Y),
                            points=self.kspace,transform_type='type_2',fft_direction='forward')
            else:
                Ku1 = tfmri.signal.fft(self.C*(u1+u2),axes=(3,4),norm="ortho")

            tmp_Ku2 = tf.experimental.numpy.diff(u2, axis=2)  # not subsriptable issue
            Ku2=tf.concat([tmp_Ku2,Ku2_zer],axis=2)

            argf1 = y1 + sigma*Ku1
            argf2 = y2 + sigma*Ku2

            y1 = argf1 - sigma*(argf1 + d)/(1 + sigma)
            if self.tied:
                y2 = argf2 - sigma * \
                    self.soft(argf2/sigma, self.rectify(self.lambdaS)/sigma)
            else:
                y2 = argf2 - sigma * \
                    self.soft(argf2/sigma, self.rectify(self.lambdaS[i])/sigma)


            # precompute the argument of prox_{tau g}
            # argg = x(i) - tau K* y(i+1)
            if self.cart==False:
                Kadjy1 = tf.math.conj(self.C)*tfmri.signal.nufft(y1*self.sq_w,grid_shape= (self.X,self.Y),
                            points=self.kspace,transform_type='type_1',fft_direction='backward')*adjscale
            else:
                Kadjy1 = tf.math.conj(self.C)*tfmri.signal.ifft(y1,axes=(3,4),norm="ortho")

            Kadjy1 = tf.math.reduce_sum(Kadjy1, axis=1, keepdims=True)

            Kadjy2 = Kadjy1

            tmp1 = tf.expand_dims(Kadjy2[:, :, 0, :, :] - y2[:, :, 0, :, :],axis=2)
            tmp2 = Kadjy2[:, :, 1:-1, :, :] - \
                tf.experimental.numpy.diff(y2[:, :, :-1, :, :], axis=2)
            tmp3 = tf.expand_dims(Kadjy2[:, :, -1, :, :] + y2[:, :, -2, :, :],axis=2)


            Kadjy2=tf.concat([tmp1,tmp2,tmp3],axis=2)

            argg1 = x_new1 - tau*Kadjy1
            argg2 = x_new2 - tau*Kadjy2
            with tf.device('/cpu:0'):
                S, U, V = tf.linalg.svd(tf.transpose(
                tf.reshape(argg1,(batch_size, Nf, X*Y)), perm=(0, 2, 1)), full_matrices=False)

            S=tf.cast(S,dtype=tf.complex64)
            U=tf.cast(U,dtype=tf.complex64)
            V=tf.cast(V,dtype=tf.complex64)

            if self.tied:
                S = self.soft(S, S[:, 0]*tau*self.rectify(self.lambdaL))
            else:
                S = self.soft(S, S[:, 0]*tau *
                              self.rectify(self.lambdaL[i]))
            x_new1 = tf.transpose(U@tf.linalg.diag(S)@tf.linalg.adjoint(V), perm=(0, 2, 1))



            x_new1 = tf.reshape(x_new1,startpoint.shape)

            #x_new1=argg1 # temp test!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! (excludes SVD)

            x_new2 = argg2

            # update the auxiliary variable
            # u(i+1) = x(n+1) + theta ( x(n+1) - x(n) )
            u1 = (x_new1 + theta*(x_new1 - x_old1))
            u2 = (x_new2 + theta*(x_new2 - x_old2))

            
            solution = ((x_new1 + x_new2))
            plt.imshow(tf.abs(solution[0,0,100,:,:]))
            a=1
       


        return solution
