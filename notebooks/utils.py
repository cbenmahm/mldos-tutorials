import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import brentq, minimize
from scipy.interpolate import interp1d
from sklearn.model_selection import KFold

def fd_distribution(x, mu, beta):
    """Fermi-Dirac distribution
        INPUTS:
        =======
        x: array energy axis (eV)
        mu: Fermi energy (eV)
        beta: inverse temperature (eV)
        """
    y = (x-mu)*beta
    ey = np.exp(-np.abs(y))
    if hasattr(x,"__iter__"):
        negs = (y<0)
        pos = (y>=0)
        try:
            y[negs] = 1 / (1+ey[negs])        
            y[pos] = ey[pos] / (1+ey[pos])
        except:
            print (x, negs, pos)
            raise
        return y
    else:
        if y<0: return 1/(1+ey)
        else: return ey/(1+ey)
        
def nelec(dos, mu, beta, xdos):
    """ computes the number of electrons from the DOS 
        INPUTS:
        =======
        dos: array of the DOS
        mu: Fermi energy
        beta: inverse temperature
        xdos: array energy axis
        """
    return trapezoid(dos * fd_distribution(xdos, mu, beta), xdos)

def getmu(dos, beta, xdos, n=2.):
    """ computes the Fermi energy of structures based on the DOS 
        INPUTS:
        =======
        dos: array of the DOS
        beta: inverse temperature
        xdos: array energy axis
        n: number of electrons
        """
    return brentq(lambda x: nelec(dos ,x ,beta, xdos)-n, xdos.min(), xdos.max())

def get_dos_fermi(dos, mu, xdos):
    """retrun the DOS value at the Fermi energy for one structure
        INPUTS:
        =======
        dos: array of the DOS
        mu: Fermi energy
        xdos: array energy axis
        """
    idos = interp1d(xdos, dos)
    dos_fermi = idos(mu)
    return dos_fermi

def get_band_energy(dos, mu, xdos, beta):
    """compute the band energy of one srtucture
        INPUTS:
        =======
        dos: array of the DOS
        mu: Fermi energy
        xdos: array energy axis
        beta: inverse temperature
        """
    return trapezoid(dos * xdos * fd_distribution(xdos, mu, beta), xdos)

def get_aofd(ldos, mu, xdos, beta):
    """compute the exciataion spectrum of one structure"""
    dx = xdos[1] - xdos[0]
    xxc = np.asarray(range(len(xdos)), float)*dx
    lxc = np.zeros(len(xxc))
    for i in range(len(xdos)):
        lxc[i] = np.sum(ldos[:len(xdos)-i] * fd_distribution(xdos[:len(xdos)-i], mu, beta) *
                              ldos[i:] * (1 - fd_distribution(xdos[i:], mu, beta)))
    lxc *= dx
    return xxc, lxc

def get_charge(local_dos, mu, xdos, beta, nel):
    """compute the local charges of one srtucture
        INPUTS:
        =======
        local_dos: array of the LDOS
        mu: Fermi energy
        xdos: array energy axis
        beta: inverse temperature
        nel: number of valence electrons
        """
    return nel - trapezoid(local_dos * fd_distribution(xdos, mu, beta), xdos, axis=1)


def gauss(x):
    return np.exp(-0.5*x**2)
def build_dos(sigma, eeigv, dx, emin, emax, natoms=None, weights=None):
    """build the DOS (per state) knowing the energy resolution required in eV
        works with FHI-aims, needs to be modified for QuantumEspresso
        INPUTS:
        =======
        sigma: Gaussian broadening
        eeigv: list of eigenergies of all the structures
        dx: energy grid spacing
        emin: minimum energy value on the grid
        emax: maximum energy value on the grid
        natoms: array of the number of atoms per structure
        weights: if you are using FHI-aims, keep value equal to None. If you are using QuantumEspresso, provide the the k-point weights. 
        
        OUTPUTS:
        xdos: energy grid
        ldos: array containing the DOS"""
    
    if natoms is None:
        raise Exception("please provide 'natoms' array containing the number of atoms per structure")
        
    beta = 1. / sigma

    ndos = int((emax-emin+3) / dx)
    xdos = np.linspace(emin-1.5, emax+1.5, ndos) # extend the energy grid by 3eV 
    ldos = np.zeros((len(eeigv), ndos))
    
    if weights == None:
        for i in range(len(eeigv)):    
            for ei in eeigv[i].flatten():
                iei = int((ei-(emin-1.5))*2/sigma)
                ldos[i] += np.exp(-0.5*((xdos[:]-ei)/sigma)**2)
            ldos[i] *= 1/np.sqrt(2*np.pi*sigma**2)/natoms[i]/len(eeigv[i])
            
    else:
        for i in range(len(eeigv)):
            for j in range(len(eeigv[i])):
                for ei in eeigv[i][j].flatten():
                    ldos[i,: ] += weights[i][j]*gauss((xdos[:]-ei)/sigma)
            ldos[i] *= 1/np.sqrt(2*np.pi*sigma**2)
    return xdos, ldos


def get_regression_weights(train_target, regularization=1e-3, kMM=[], kNM=[], jitter=1e-9):
    """get the regression weights.. can be used without the train_model function.. follows the same logic in librascal train_gap_model"""
    KNM = kNM.copy()
    Y = train_target.copy()
    
    KMM = kMM.copy()
    KMM[np.diag_indices_from(KMM)] += jitter
    
    nref = len(kMM)
    delta = np.var(train_target) / kMM.trace() / nref
    
    KNM /= regularization / delta
    Y /= regularization / delta
    
    K = kMM + KNM.T @ KNM
    Y = KNM.T @ Y
    
    weights = np.linalg.lstsq(K, Y, rcond=None)[0]
    return weights

def get_rmse(a, b, xdos=None, perc=False):
    """ computes  Root Mean Squared Error (RMSE) of array properties (DOS/aofd).
         a=pred, b=target, xdos, perc: if False return RMSE else return %RMSE"""
    
    if xdos is not None:
        rmse = np.sqrt(trapezoid((a - b)**2, xdos, axis=1).mean())
        if not perc:
            return rmse
        else:
            mean = b.mean(axis=0)
            std = np.sqrt(trapezoid((b - mean)**2, xdos, axis=1).mean())
            return 100 * rmse / std
    else:
        rmse = np.sqrt(((a - b)**2).mean())
        if not perc:
            return rmse
        else:
            return 100 * rmse / b.std(ddof=1)
        

def pred_error(i_regularization, train_target, kNM, kMM, cv, train_idx, xdos):
    """helper function for the train_model function"""
    kfold = KFold(n_splits=cv, shuffle=False)
    regularization = np.exp(i_regularization[0])
    temp_err = 0.
    for train, test in kfold.split(train_idx):
        w = get_regression_weights(train_target[train], 
                                   kMM=kMM, 
                                   regularization=regularization, 
                                   kNM=kNM[train])
        target_pred = kNM @ w
        temp_err += get_rmse(target_pred[test], train_target[test], xdos, perc=True)
    return temp_err/cv

def train_model(train_target, kNM=[], kMM=[], cv=2, i_regularization=1e-6, maxiter=8, xdos=None):
    """returns the weights of the trained model
        INPUTS:
        =======
        train_target: DOS of the training set (with or without their mean
        kNM: KNM matrix of the training set
        kMM: kernle matrix of the sparse points
        cv: number of the folds for the cross-validation
        i_regularization: initial guess for the regularizer
        maxiter: number of max iterations for the optimizer
        xdos: energy grid of the DOS"""
    
    train_idx = np.arange(len(train_target))
    rmin = minimize(pred_error, [np.log(i_regularization)], args=(train_target, kNM, kMM, cv, train_idx, xdos), method="Nelder-Mead", options={"maxiter":maxiter})
    print(rmin)
    regularization = np.exp(rmin["x"])[0]
    print(regularization)

    # weights of the model
    print(train_target.shape)
    weights = get_regression_weights(train_target, 
                                 kMM=kMM, 
                                 regularization=regularization,
                                 kNM=kNM)
    return weights

def build_truncated_dos(basis, coeffs, mean, n_pc=10):
    """ builds an approximate DOS providing the basis elements and coeffs""" 
    return coeffs @ basis[:, :n_pc].T + mean

def build_pc(dos, dosmean, n_pc=10):
    """
    n_pc: the number of prinicpal components to keep
    """
   
    #dosmean = dos.mean(axis=0)
    cdos = dos - dosmean
    doscov = (cdos.T @ cdos) / len(dos)
    doseva, doseve = np.linalg.eigh(doscov)
    doseva = np.flip(doseva, axis = 0)
    doseve = np.flip(doseve, axis = 1)     
    print('Variance covered with {} PCs is = {}'.format(n_pc, doseva[:n_pc].sum()/doseva.sum()))
    return doseva, doseve[:, :n_pc]
        
def build_coeffs(dos, doseve):
    """ finds basis elements and projection coefs of the DOS 
        INPUTS:
        =======
        dos: DOS of the strcutures, should be centered wrt to training set
        doseve: the principal components
        OUPUTS:
        dosproj: projection coefficients on the retained """
    
    dosproj = dos @ doseve 
    return dosproj
