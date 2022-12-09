import torch

def replace_denormals(x: torch.Tensor, threshold=1e-10) -> torch.Tensor:
    y = x.clone()
    y[torch.abs(x) < threshold] = 0
    return y

"""
Adaptive piece-wise linear unit
"""
class APL(torch.nn.Module):
    def __init__(self, K, constraint="relu", device="cpu", a=-1.0, b=1.0):
        super(APL, self).__init__()
        self.K = K
        self.constraint = constraint
        self.device = device
        self.ReLU = torch.nn.ReLU()
        self.a = torch.nn.Parameter(a*torch.rand(self.K).to(self.device), requires_grad=True)
        self.b = torch.nn.Parameter(b*torch.rand(self.K).to(self.device), requires_grad=True)

    def positive(self, w: torch.Tensor) -> torch.Tensor:
        if self.constraint == "relu":
            return torch.nn.functional.relu(w)
        elif self.constraint == "sqr":
            return torch.square(w)
        elif self.constraint == "sqrt":
            return torch.sqrt(torch.square(w) + 1e-8)
        else:
            return w
    
    def negative(self, w: torch.Tensor) -> torch.Tensor:
        return -self.positive(-w)
    
    def forward(self, x):
        if torch.is_complex(x):
            out = self.ReLU(torch.abs(x))
            for j in range(self.K):
                out = out + self.negative(self.a[j])*self.ReLU(-torch.abs(x) + self.b[j])
            out = self.ReLU(out)
            return out*torch.exp(1j*torch.angle(replace_denormals(x)))
        else:
            out = self.ReLU(x)
            for j in range(self.K):
                out = out + self.negative(self.a[j])*self.ReLU(-x + self.b[j])
            out = self.ReLU(out)
            return out
        
"""
Soft thresholding
"""        
class soft(torch.nn.Module):
    def __init__(self, constraint="relu", device="cpu", a=1.0, b=1.0, learnslope=False):
        super(soft, self).__init__()
        self.constraint = constraint
        self.device = device
        self.ReLU = torch.nn.ReLU()
        self.learnslope = learnslope
        self.a = torch.nn.Parameter(torch.tensor(a).to(self.device), requires_grad=self.learnslope) # slope
        self.b = torch.nn.Parameter(torch.tensor(b).to(self.device), requires_grad=True) # threshold
    
    def positive(self, w: torch.Tensor) -> torch.Tensor:
        if self.constraint == "relu":
            return torch.nn.functional.relu(w)
        elif self.constraint == "sqr":
            return torch.square(w)
        elif self.constraint == "sqrt":
            return torch.sqrt(torch.square(w) + 1e-8)
        else:
            return w
        
    def forward(self, x):
        if torch.is_complex(x):
            out = self.ReLU(torch.abs(x) - self.positive(self.b))*self.positive(self.a)
            return out*torch.exp(1j*torch.angle(replace_denormals(x)))
        else:
            out = self.ReLU(x - self.positive(self.b)) - self.ReLU(-x - self.positive(self.b))*self.positive(self.a)
            return out

"""
Super class for low-rank + sparse models
"""
class SuperLS(torch.nn.Module):
    def __init__(self, device="cpu", C=None, fft=False, verbose=False, scalesing=True):
        super().__init__()
        self.device = device
        self.C = torch.tensor(1.0).to(self.device) if C is None else C.to(self.device)
        self.fft = fft
        self.datatype = torch.float if C is None else torch.cfloat
        self.verbose = verbose
        self.scalesing = scalesing
        
    # ===================================
    # functions related to the operator K
    # ===================================
    def data2sequence(self, d) -> torch.Tensor:
        if self.fft:
            return torch.sum(torch.conj(self.C)*torch.fft.ifftn(d, dim=(2, 3), norm="ortho"), 1, keepdim=True)
        else:
            return torch.sum(torch.conj(self.C)*d, 1, keepdim=True)
        
    def sequence2data(self, X) -> torch.Tensor:
        if self.fft:
            return torch.fft.fftn(self.C*X, dim=(2, 3), norm="ortho")
        else:
            return self.C*X        
        
    def diff(self, X) -> torch.Tensor:
        return torch.nn.functional.pad(torch.diff(X, axis=4), (0, 1))
    
    def transdiff(self, X) -> torch.Tensor:
        Y = torch.zeros_like(X)
        Y[:, :, :, :, 0] = -X[:, :, :, :, 0]
        Y[:, :, :, :, 1:-1] = -torch.diff(X[:, :, :, :, :-1], axis=4)
        Y[:, :, :, :, -1] = X[:, :, :, :, -2]
        return Y

"""
Classical L+S
"""
class LS(SuperLS):
    def __init__(self,
                 lambdaL=2e-1,
                 lambdaS=2e-3,
                 sigma=0.3536,
                 tau=0.3536,
                 iterations=100,
                 objval=False,
                 device="cpu",
                 C=None,
                 fft=False,
                 scalesing=True,
                 verbose=False):
        super().__init__(device, C, fft, verbose, scalesing)
        self.lambdaL = lambdaL
        self.lambdaS = lambdaS
        self.sigma = sigma
        self.tau = tau
        self.iterations = iterations
        self.objval = objval
        
    # ==================
    # proximal operators
    # ==================
    def soft(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if torch.is_complex(x):
            sgnx = torch.exp(1j*torch.angle(replace_denormals(x)))
        else:
            sgnx = torch.sign(x)
        return sgnx*torch.maximum(torch.abs(x) - t, torch.tensor(0))
            
    # =============
    # main function
    # =============
    def forward(self, d: torch.Tensor, target=None, metric=None):
                
        normd = torch.norm(d, p=torch.inf)
        d = d/normd
        
        # if we will not use fft, move from the k-space just once
        if not self.fft:
            d = torch.fft.ifftn(d, dim=(2, 3), norm="ortho")    
        startpoint = self.data2sequence(d)
        batch_size, _, Y, X, Nf = startpoint.shape
        if target is not None:
            progress = torch.full([batch_size, self.iterations], torch.inf)
        else:
            progress = None

        # the variable in the co-domain of K
        M = torch.zeros(d.shape, dtype=self.datatype).to(self.device)
        N = torch.zeros(startpoint.shape, dtype=self.datatype).to(self.device)

        # the main variable in the domain of K
        L = startpoint.to(self.device)
        S = startpoint.to(self.device)

        # the auxiliary variable in the domain of K
        L_bar = L
        S_bar = S

        # more auxiliary variables
        K1 = torch.zeros(d.shape, dtype=self.datatype).to(self.device)
        K2 = torch.zeros(startpoint.shape, dtype=self.datatype).to(self.device)

        # tracking of the objective
        if self.objval:
            objective = torch.full([batch_size, self.iterations], torch.inf)
            consistency = torch.full([batch_size, self.iterations], torch.inf)
            lowrank = torch.full([batch_size, self.iterations], torch.inf)
            sparse = torch.full([batch_size, self.iterations], torch.inf)

        # iterations
        for i in range(self.iterations):
            if self.verbose:
                # print(f"Iteration {i+1} of {self.iterations}", end="\r")
                print(f"\rIteration {i+1} of {self.iterations}", end="")
                
            # keep the solution from the previous iteration
            L_old = torch.clone(L)
            S_old = torch.clone(S)

            # compute prox_{sigma f*}
            K1 = self.sequence2data(L_bar + S_bar)
            K2 = self.diff(S_bar)

            M = (M + self.sigma*K1 - self.sigma*d)/(1 + self.sigma)
            N = N + self.sigma*K2 - self.soft(N + self.sigma*K2, self.lambdaS)

            # compute prox_{tau g}
            AM = self.data2sequence(M)
            argSVT = L - self.tau*AM

            U, singulars, V = torch.linalg.svd(torch.permute(
                argSVT.reshape(batch_size, Nf, X*Y), (0, 2, 1)), full_matrices=False)
            if self.scalesing:
                singulars = self.soft(singulars, singulars[:, 0].unsqueeze(1)*self.tau*self.lambdaL)
            else:
                singulars = self.soft(singulars, self.tau*self.lambdaL)
            L = torch.permute((U * singulars.unsqueeze(1)) @ V, (0, 2, 1))
            L = L.reshape(startpoint.shape)
            
            S = S - self.tau*AM - self.tau*self.transdiff(N)
            
            # update the auxiliary variable (fixed theta = 1)
            L_bar = 2*L - L_old
            S_bar = 2*S - S_old
        
            # evaluate the objective function
            if self.objval:
                # data-fitting term
                consistency[:, i] = 0.5*torch.norm(d.reshape(batch_size, -1) -
                                                   self.sequence2data(L + S).reshape(batch_size, -1), dim=1)**2
                
                # low-rank term
                argSVD = L.reshape(batch_size, Nf, X*Y)
                argSVD = torch.permute(argSVD, (0, 2, 1))
                singulars = torch.linalg.svdvals(argSVD)
                if self.scalesing:
                    lowrank[:, i] = self.lambdaL*singulars[:,0]*torch.norm(singulars, p=1, dim=1)
                else:
                    lowrank[:, i] = self.lambdaL*torch.norm(singulars, p=1, dim=1)
            
                # sparse term
                sparse[:, i] = self.lambdaS*torch.norm(self.diff(S).reshape(batch_size, -1), p=1, dim=1)
            
                # sum of all three
                objective[:, i] = consistency[:, i] + lowrank[:, i] + sparse[:, i]
            
            # update the "distance" from target
            if target is not None:
                for b in range(batch_size):
                    progress[b, i] = metric((L + S)*normd, target)
                
        if self.verbose:
            print()

        solution = (L + S)*normd

        if self.objval:
            return solution, progress, consistency, lowrank, sparse, objective
        else:
            return solution, progress

"""
Unfolded L+S
"""
class UFLS(SuperLS):
    def __init__(self,
                 initL=2e-1,
                 initS=2e-3,
                 n_layers=100,
                 device="cpu",
                 tied=False,
                 constraint="relu",
                 C=None,
                 K=5,
                 fft=False,
                 learnst=True,
                 activation="APL",
                 verbose=False,
                 scalesing=True):
        super().__init__(device, C, fft, verbose, scalesing)
        self.frozen = False
        self.n_layers = n_layers
        self.tied = tied
        self.constraint = constraint
        self.K = K
        
        # parameter initialization
        if self.tied:
            if activation == "APL":
                self.proxL = APL(self.K, self.constraint, self.device, a=-1.0, b=initL*0.3536)
                self.proxS = APL(self.K, self.constraint, self.device, a=-1.0, b=initS)
            elif activation == "soft":
                self.proxL = soft(self.constraint, self.device, a=1.0, b=initL*0.3536, learnslope=True)
                self.proxS = soft(self.constraint, self.device, a=1.0, b=initS, learnslope=True)
            else:
                self.proxL = soft(self.constraint, self.device, a=1.0, b=initL*0.3536, learnslope=False)
                self.proxS = soft(self.constraint, self.device, a=1.0, b=initS, learnslope=False)
            self.sigma = torch.nn.Parameter(0.3536*torch.ones(5).to(self.device), requires_grad=learnst) # initialized to 1/sqrt(8)
            self.tau = torch.nn.Parameter(0.3536*torch.ones(3).to(self.device), requires_grad=learnst)
        else:
            if activation == "APL":
                self.proxL = torch.nn.ModuleList([APL(self.K, self.constraint, self.device, a=-1.0, b=initL*0.3536) for i in range(self.n_layers)])
                self.proxS = torch.nn.ModuleList([APL(self.K, self.constraint, self.device, a=-1.0, b=initS) for i in range(self.n_layers)])
            elif activation == "soft":
                self.proxL = torch.nn.ModuleList([soft(self.constraint, self.device, a=1.0, b=initL*0.3536, learnslope=True) for i in range(self.n_layers)]) 
                self.proxS = torch.nn.ModuleList([soft(self.constraint, self.device, a=1.0, b=initS, learnslope=True) for i in range(self.n_layers)])
            else:
                self.proxL = torch.nn.ModuleList([soft(self.constraint, self.device, a=1.0, b=initL*0.3536, learnslope=False) for i in range(self.n_layers)]) 
                self.proxS = torch.nn.ModuleList([soft(self.constraint, self.device, a=1.0, b=initS, learnslope=False) for i in range(self.n_layers)])
            self.sigma = torch.nn.ParameterList([
                torch.nn.Parameter(0.3536*torch.ones(5).to(self.device), requires_grad=learnst) for i in range(self.n_layers)])
            self.tau = torch.nn.ParameterList([
                torch.nn.Parameter(0.3536*torch.ones(3).to(self.device), requires_grad=learnst) for i in range(self.n_layers)])
    
    # ===================================
    # functions related to the parameters
    # ===================================
    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False
        self.frozen = True

    def unfreeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = True
        self.frozen = False

    def rectify(self, w: torch.Tensor) -> torch.Tensor:
        if self.constraint == "relu":
            return torch.nn.functional.relu(w)
        elif self.constraint == "sqr":
            return torch.square(w)
        elif self.constraint == "sqrt":
            return torch.sqrt(torch.square(w) + 1e-8)
        else:
            return w

    # =============
    # main function
    # =============
    def forward(self, d: torch.Tensor, target=None, metric=None):

        normd = torch.norm(d, p=torch.inf)
        d = d/normd        

        # if we will not use fft, move from the k-space just once
        if not self.fft:
            d = torch.fft.ifftn(d, dim=(2, 3), norm="ortho")
        startpoint = self.data2sequence(d)
        batch_size, _, Y, X, Nf = startpoint.shape
        if target is not None:
            progress = torch.full([batch_size, self.n_layers], torch.inf)
        else:
            progress = None

        # the variable in the co-domain of K
        M = torch.zeros(d.shape, dtype=self.datatype).to(self.device)
        N = torch.zeros(startpoint.shape, dtype=self.datatype).to(self.device)

        # the main variable in the domain of K
        L = startpoint.to(self.device)
        S = startpoint.to(self.device)

        # the auxiliary variable in the domain of K
        L_bar = L
        S_bar = S

        # more auxiliary variables
        K1 = torch.zeros(d.shape, dtype=self.datatype).to(self.device)
        K2 = torch.zeros(startpoint.shape, dtype=self.datatype).to(self.device)

        # iterations
        for layer in range(self.n_layers):

            if self.tied:
                # keep the solution from the previous iteration
                L_old = torch.clone(L)
                S_old = torch.clone(S)

                # compute prox_{sigma f*}
                K1 = self.sequence2data(L_bar + S_bar)
                K2 = self.diff(S_bar)
                
                # M = (M + sigma*K1 + sigma*d)/(1 + sigma)
                M = (M + self.rectify(self.sigma[0])*K1 - self.rectify(self.sigma[1])*d)/(1 + self.rectify(self.sigma[2]))
                
                # N = N + sigma*K2 - soft(N + sigma*K2, lambdaS)
                N = N + self.rectify(self.sigma[3])*K2 - self.proxS(N + self.rectify(self.sigma[4])*K2)
                
                # compute prox_{tau g}
                AM = self.data2sequence(M)
                argSVT = L - self.rectify(self.tau[0])*AM
                U, singulars, V = torch.linalg.svd(torch.permute(
                    argSVT.reshape(batch_size, Nf, X*Y), (0, 2, 1)), full_matrices=False)
                if self.scalesing:
                    largest = torch.clone(singulars[:, 0].unsqueeze(1))
                    singulars = singulars / largest
                singulars = self.proxL(singulars)
                if self.scalesing:
                    singulars = singulars * largest
                L = torch.permute((U * singulars.unsqueeze(1)) @ V, (0, 2, 1))
                L = L.reshape(startpoint.shape)
                S = S - self.rectify(self.tau[1])*AM - self.rectify(self.tau[2])*self.transdiff(N)
                
                # update the auxiliary variable (fixed theta = 1)
                L_bar = 2*L - L_old
                S_bar = 2*S - S_old
            else:
                # if we decide to increase the number of "layers" for testing, we
                # recompute the layer index to correspond to what has been actually
                # learned
                # i = layer % len(self.proxL)
                i = int(layer*len(self.proxL)/self.n_layers)
                
                # keep the solution from the previous iteration
                L_old = torch.clone(L)
                S_old = torch.clone(S)

                # compute prox_{sigma f*}
                K1 = self.sequence2data(L_bar + S_bar)
                K2 = self.diff(S_bar)
                
                # M = (M + sigma*K1 + sigma*d)/(1 + sigma)
                M = (M + self.rectify(self.sigma[i][0])*K1 - self.rectify(self.sigma[i][1])*d)/(1 + self.rectify(self.sigma[i][2]))
                
                # N = N + sigma*K2 - soft(N + sigma*K2, lambdaS)
                N = N + self.rectify(self.sigma[i][3])*K2 - self.proxS[i](N + self.rectify(self.sigma[i][4])*K2)

                # compute prox_{tau g}
                AM = self.data2sequence(M)
                argSVT = L - self.rectify(self.tau[i][0])*AM
                U, singulars, V = torch.linalg.svd(torch.permute(
                    argSVT.reshape(batch_size, Nf, X*Y), (0, 2, 1)), full_matrices=False)
                if self.scalesing:
                    largest = torch.clone(singulars[:, 0].unsqueeze(1))
                    singulars = singulars / largest
                singulars = self.proxL[i](singulars)
                if self.scalesing:
                    singulars = singulars * largest
                L = torch.permute((U * singulars.unsqueeze(1)) @ V, (0, 2, 1))
                L = L.reshape(startpoint.shape)
                S = S - self.rectify(self.tau[i][1])*AM - self.rectify(self.tau[i][2])*self.transdiff(N)

                # update the auxiliary variable (fixed theta = 1)
                L_bar = 2*L - L_old
                S_bar = 2*S - S_old
            
            # update the "distance" from target
            if target is not None:
                for b in range(batch_size):
                    progress[b, layer] = metric((L + S)*normd, target)

        solution = (L + S)*normd
        return solution, progress
