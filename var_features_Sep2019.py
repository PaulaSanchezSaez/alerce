#code with all the features for an AGN light curve
import numpy as np
import emcee
import corner
import scipy.stats as st
import scipy  as sp
from scipy.stats import chi2
import turbofats as FATS
import GPy
from numpy.linalg import inv

################################################################################

def var_parameters(jd,mag,err):
    """
    Calculate the probability of a light curve to be variable
    and the excess variance.

    inputs:
    jd: julian days array
    mag: magnitudes array
    err: error of magnitudes array

    outputs:
    p_chi: Probability that the source is intrinsically variable (Pvar)
    ex_var: excess variance, a measure of the intrinsic variability amplitude
    ex_verr: error of the excess variance

    """

    mean = np.mean(mag)
    nepochs = float(len(jd))

    chi = np.sum( (mag - mean)**2. / err**2. )
    p_chi = chi2.cdf(chi,(nepochs-1))


    a = (mag-mean)**2
    ex_var = (np.sum(a-err**2)/((nepochs*(mean**2))))
    sd = np.sqrt((1./(nepochs-1))*np.sum(((a-err**2)-ex_var*(mean**2))**2))
    ex_verr = sd/((mean**2)*np.sqrt(nepochs))


    return p_chi, ex_var, ex_verr

################################################################################

def GP_DRW(t, magnitude, err):
    """
    Based on Matthew Graham's method to model DRW with gaussian process

    inputs:
    t: julian days array
    mag: magnitudes array
    err: error of magnitudes array

    outputs:
    sigma^2: variance of the light curve at short time scales
    tau: decorrelation time scale
    """

    # Fit GP OU model
    mag = magnitude-magnitude.mean()
    kern = GPy.kern.OU(1)
    m = GPy.models.GPHeteroscedasticRegression(t[:, None], mag[:, None], kern)
    m['.*het_Gauss.variance'] = abs(err ** 2.)[:, None] # Set the noise parameters to the error in Y
    m.het_Gauss.variance.fix() # We can fix the noise term, since we already know it
    m.optimize()
    pars = [m.OU.variance.values[0], m.OU.lengthscale.values[0]] # sigma^2, tau
    return pars[0], pars[1]

################################################################################
#functions to compute an IAR model. Author: Felipe Elorrieta.

def IAR_phi_loglik(x,sT,y,delta,standarized=True):
    """
    Based on Felipe Elorrieta's method to model DRW with IAR

    inputs:
    sT: julian days array
    y: magnitudes array
    delta: error of magnitudes array

    outputs:
    s1: IAR likelihood
    """

    n = len(y)
    sigma = 1
    if standarized == False:
        sigma = np.var(y,ddof=1)
    d = np.diff(sT)
    phi = x**d
    yhat = phi*y[0:(n-1)]
    y2 = np.vstack((y[1:n],yhat))
    cte = 0.5*n*np.log(2*np.pi)
    s1 = cte+0.5*np.sum(np.log(sigma*(1-phi**2)+delta[1:]**2)+(y2[0,]-y2[1,])**2/(sigma*(1-phi**2)+delta[1:]**2))
    return s1

def IAR_loglik(sT,y,delta):
    """
    Based on Felipe Elorrieta's method to model DRW with IAR

    inputs:
    sT: julian days array
    y: magnitudes array
    delta: error of magnitudes array

    outputs:
    phi: IAR phi
    tau: DRW tau (decorrelation time scale)
    """

    ynorm = (y-np.mean(y))/np.sqrt(np.var(y,ddof=1))
    deltanorm = delta/np.sqrt(np.var(y,ddof=1))


    out = sp.optimize.minimize_scalar(IAR_phi_loglik,args=(sT,ynorm,deltanorm,True),
        bounds=(0, 1),method='bounded',options={'xatol': 1e-12, 'maxiter': 50000})

    phi = out.x
    tau = -1.0/np.log(phi)

    return phi, tau


################################################################################
#functions to compute an IAR model with Kalman filter. Author: Felipe Elorrieta.

def IAR_phi_kalman(x,t,y,yerr,standarized=True,c=0.5):
    n=len(y)
    Sighat=np.zeros(shape=(1,1))
    Sighat[0,0]=1
    if standarized == False:
         Sighat=np.var(y)*Sighat
    xhat=np.zeros(shape=(1,n))
    delta=np.diff(t)
    Q=Sighat
    phi=x
    F=np.zeros(shape=(1,1))
    G=np.zeros(shape=(1,1))
    G[0,0]=1
    sum_Lambda=0
    sum_error=0
    if np.isnan(phi) == True:
        phi=1.1
    if abs(phi) < 1:
        for i in range(n-1):
            Lambda=np.dot(np.dot(G,Sighat),G.transpose())+yerr[i+1]**2
            if (Lambda <= 0) or (np.isnan(Lambda) == True):
                sum_Lambda=n*1e10
                break
            phi2=phi**delta[i]
            F[0,0]=phi2
            phi2=1-phi**(delta[i]*2)
            Qt=phi2*Q
            sum_Lambda=sum_Lambda+np.log(Lambda)
            Theta=np.dot(np.dot(F,Sighat),G.transpose())
            sum_error= sum_error + (y[i]-np.dot(G,xhat[0:1,i]))**2/Lambda
            xhat[0:1,i+1]=np.dot(F,xhat[0:1,i])+np.dot(np.dot(Theta,inv(Lambda)),(y[i]-np.dot(G,xhat[0:1,i])))
            Sighat=np.dot(np.dot(F,Sighat),F.transpose()) + Qt - np.dot(np.dot(Theta,inv(Lambda)),Theta.transpose())
        yhat=np.dot(G,xhat)
        out=(sum_Lambda + sum_error)/n
        if np.isnan(sum_Lambda) == True:
            out=1e10
    else:
        out=1e10
    return out


def IAR_kalman(sT,y,delta=0,standarized=True):
    if np.sum(delta)==0:
        delta=np.zeros(len(y))

    ynorm = (y-np.mean(y))/np.sqrt(np.var(y,ddof=1))
    deltanorm = delta/np.sqrt(np.var(y,ddof=1))

    out=sp.optimize.minimize_scalar(IAR_phi_kalman,args=(sT,ynorm,deltanorm,standarized),bounds=(0,1),method="bounded",options={'xatol': 1e-12, 'maxiter': 50000})

    phi = out.x
    tau = -1.0/np.log(phi)

    return phi[0][0], tau[0][0]


################################################################################
#functions needed to compute the Structure Function (SF),
#based on Schmidt et al. 2010

def SFarray(jd,mag,err):
    """
    calculate an array with (m(ti)-m(tj)), whit (err(t)^2+err(t+tau)^2) and another with tau=dt

    inputs:
    jd: julian days array
    mag: magnitudes array
    err: error of magnitudes array

    outputs:
    tauarray: array with the difference in time (ti-tj)
    sfarray: array with |m(ti)-m(tj)|
    errarray: array with err(ti)^2+err(tj)^2
    """

    sfarray=[]
    tauarray=[]
    errarray=[]
    for i, item in enumerate(mag):
        for j in range(i+1,len(mag)):
            dm=mag[i]-mag[j]
            sigma=err[i]**2+err[j]**2
            dt=(jd[j]-jd[i])
            sfarray.append(np.abs(dm))
            tauarray.append(dt)
            errarray.append(sigma)
    sfarray=np.array(sfarray)
    tauarray=np.array(tauarray)
    errarray=np.array(errarray)
    return (tauarray,sfarray,errarray)


def Vmod(dt,A,gamma):
    """SF power model: SF = A*(dt/365)**gamma"""
    return ( A*((dt/365.0)**gamma) )


def Veff2(dt,sigma,A,gamma):
    """SF power model plus the error"""
    return ( (Vmod(dt,A,gamma))**2 + sigma )

def like_one(theta,dt,dmag,sigma):
    """likelihood for one value of dmag (one pair of epochs)"""

    gamma, A = theta
    aux=(1/np.sqrt(2*np.pi*Veff2(dt,sigma,A,gamma)))*np.exp(-1.0*(dmag**2)/(2.0*Veff2(dt,sigma,A,gamma)))

    return aux

def lnlike(theta, dtarray, dmagarray, sigmaarray):
    """likelihood for the whole light curve"""
    gamma, A = theta

    aux=np.sum(np.log(like_one(theta,dtarray,dmagarray,sigmaarray)))

    return aux


def lnprior(theta):
    """priors for gamma and A, as in Schmidt et al. 2010"""
    gamma, A = theta

    if 0.0 < gamma and 0.0 < A < 2.0 :
        return ( np.log(1.0/A) + np.log(1.0/(1.0+(gamma**2.0))) )

    return -np.inf


def lnprob(theta, dtarray, dmagarray, sigmaarray):
    """logatithm of the posterior as in  Schmidt et al. 2010"""
    lp = lnprior(theta)

    if not np.isfinite(lp):
    #if (lp==-(10**32)):
        return -np.inf
        #return -(10**32)
    return lp +lnlike(theta, dtarray, dmagarray, sigmaarray)

################################################################################
#functions to compute the SF maximazing the likelihood

def neg_lnlike(theta, dtarray, dmagarray, sigmaarray):
    """ negative value of the likelihood, to perfor maximization using the minimize function """
    return(-1.0*lnlike(theta, dtarray, dmagarray, sigmaarray))

def SF_ML(jd,mag,errmag,x0=[0.5, 0.5],bnds=((0.0, 3.0), (0.0,3.0))):
    """
    fit the model A*tau^gamma to the SF, finding the maximum value of the likelihood

    inputs:
    jd: julian days array
    mag: magnitudes array
    errmag: error of magnitudes array
    x0: first guess for the values of A and gamma, default=[0.5, 0.5]
    bnds: boundaries for the minimization, default=((0.0, 3.0), (0.0,3.0))

    outputs:
    a_min: amplitude of the SF at 1 year
    g_min: logarithmic gradient of the change in magnitude of the SF
    """

    dtarray, dmagarray, sigmaarray = SFarray(jd,mag,errmag)
    ndt=np.where((dtarray<=365))
    dtarray=dtarray[ndt]
    dmagarray=dmagarray[ndt]
    sigmaarray=sigmaarray[ndt]


    x0 = [0.5, 0.5]
    bnds = ((0.0, 3.0), (0.0,3.0))

    #res = sp.optimize.minimize(neg_lnlike, x0, args=(dtarray, dmagarray, sigmaarray),
    #               method='L-BFGS-B', bounds=bnds, options={'ftol': 1e-15, 'gtol': 1e-10, 'eps': 1e-08, 'maxfun': 150000, 'maxiter': 150000, 'maxls': 40})

    #res = sp.optimize.minimize(neg_lnlike, x0, args=(dtarray, dmagarray, sigmaarray),
    #               method='Nelder-Mead', bounds=bnds, options={'fatol': 1e-10, 'xatol': 1e-10, 'maxiter': 15000})


    res = sp.optimize.minimize(neg_lnlike, x0, bounds=bnds, args=(dtarray, dmagarray, sigmaarray),
                   method='SLSQP', options={'ftol': 1e-10, 'maxiter': 15000})

    g_min = res.x[0]
    a_min = res.x[1]

    return(g_min, a_min)

################################################################################
#functions to compute the SF with a bayesian aproach

def fitSF_mcmc(jd,mag,errmag,ndim=2,nwalkers=50,nit=150,nthr=1):
    """
    Function that fits the values of A and gamma using mcmc with the package emcee,
    following the approach of Schmidt et al. 2010.

    inputs:
    jd: julian days array
    mag: magnitudes array
    errmag: error of magnitudes array
    ndim: number of dimensions of the model (default=2, for SF)
    nwalkers: number of walkers for emcee (default=50)
    nit: number of iterations for emcee (default=150)
    nthr: number of cores used for emcee (default=1)

    outputs:
    A_mcmc: array with the median value, lower error and uper error of A
            (amplitude at 1 year)
    g_mcmc: array with the median value, lower error and uper error of gamma
            (logarithmic gradient of the change in magnitude)

    """

    #we calculate the arrays of dm, dt and sigma
    dtarray, dmagarray, sigmaarray = SFarray(jd,mag,errmag)

    ndt=np.where((dtarray<=365) & (dtarray>=10))
    dtarray=dtarray[ndt]
    dmagarray=dmagarray[ndt]
    sigmaarray=sigmaarray[ndt]

    #definition of the optimal initial position of the walkers
    p0 = np.random.rand(ndim*nwalkers).reshape((nwalkers,ndim)) #gess to start the burn in fase

    #run mcmc
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=nthr, args=(dtarray, dmagarray, sigmaarray))

    pos, prob, state = sampler.run_mcmc(p0,100) #from pos we have a best gess of the initial walkers
    sampler.reset()
    print("Running MCMC...")
    sampler.run_mcmc(pos, nit,rstate0=state)
    print("Done.")

    # Compute the quantiles.
    samples=sampler.chain[:,50:,:].reshape((-1,ndim))

    A_fin=samples[:,1]
    gamma_fin=samples[:,0]

    A_mcmc=(np.percentile(A_fin, 50),np.percentile(A_fin, 50)-np.percentile(A_fin, 15.865),np.percentile(A_fin, 84.135)-np.percentile(A_fin, 50))
    g_mcmc=(np.percentile(gamma_fin, 50),np.percentile(gamma_fin, 50)-np.percentile(gamma_fin, 15.865),np.percentile(gamma_fin, 84.135)-np.percentile(gamma_fin, 50))

    sampler.reset()
    return (g_mcmc, A_mcmc)

################################################################################
#functions to compute the SF with the arithmetic formula of Caplar  et al. 2017

def bincalc(nbin=0.1,bmin=5,bmax=2000):
    """
    calculate the bin range, in logscale

    inputs:
    nbin: size of the bin in log scale
    bmin: minimum value of the bins
    bmax: maximum value of the bins

    output: bins array
    """

    logbmin=np.log10(bmin)
    logbmax=np.log10(bmax)

    logbins=np.arange(logbmin,logbmax,nbin)

    bins=10**logbins

    #bins=np.linspace(bmin,bmax,60)
    return (bins)


def SF_formula(jd,mag,errmag,nbin=0.1,bmin=5,bmax=2000):


    dtarray, dmagarray, sigmaarray = SFarray(jd,mag,errmag)
    ndt=np.where((dtarray<=365) & (dtarray>=5))
    dtarray=dtarray[ndt]
    dmagarray=dmagarray[ndt]
    sigmaarray=sigmaarray[ndt]

    bins=bincalc(nbin,bmin,bmax)

    sf_list=[]
    tau_list=[]
    numobj_list=[]

    for i in range(0,len(bins)-1):
        n=np.where((dtarray>=bins[i]) & (dtarray<bins[i+1]))
        nobjbin=len(n[0])
        if nobjbin>=6:
            dmag1=(dmagarray[n])**2
            derr1=(sigmaarray[n])
            sf=(dmag1-derr1)
            sff=np.sqrt(np.mean(sf))
            sf_list.append(sff)
            numobj_list.append(nobjbin)
            #central tau for the bin
            tau_list.append((bins[i]+bins[i+1])*0.5)


    SF=np.array(sf_list)
    nob=np.array(numobj_list)
    tau=np.array(tau_list)
    nn=np.where((nob>6) & (SF>-99))
    tau=tau[nn]
    SF=SF[nn]


    return (tau/365.,SF)


def fitSF_formula(jd,mag,errmag,nbin,bmin,bmax):
    """
    fit the model A*tau^gamma to the SF, the fit only consider the bins with more than 6 pairs.

    inputs:
    jd: julian days array
    mag: magnitudes array
    errmag: error of magnitudes array
    nbin: size of the bin in log scale
    bmin: minimum value of the bins
    bmax: maximum value of the bins

    outputs:
    A: amplitude of the SF at 1 year
    gamma: logarithmic gradient of the change in magnitude of the SF
    """

    tau,sf = SF_formula(jd,mag,errmag,nbin,bmin,bmax)

    y=np.log10(sf)
    x=np.log10(tau)
    x=x[np.where((tau<=0.5) & (tau>0.01))]
    y=y[np.where((tau<=0.5) & (tau>0.01))]
    coefficients = np.polyfit(x, y, 1)

    A=10**(coefficients[1])
    gamma=coefficients[0]

    return(gamma, A)





def SFSchmidt10(jd,mag,errmag,nbin=0.1,bmin=5,bmax=2000):
    """
    calculate the SF using the formula (2) in Schmidt et al. 2010

    inputs:
    jd: julian days array
    mag: magnitudes array
    errmag: error of magnitudes array
    nbin: size of the bin in log scale
    bmin: minimum value of the bins
    bmax: maximum value of the bins

    outputs:
    tau: average value of the time bins, normalized at 1 year
    SF: SF value at each tau

    """

    dtarray, dmagarray, sigmaarray = SFarray(jd,mag,errmag)
    ndt=np.where((dtarray<=365) & (dtarray>=5))
    dtarray=dtarray[ndt]
    dmagarray=dmagarray[ndt]
    sigmaarray=sigmaarray[ndt]

    bins=bincalc(nbin,bmin,bmax)
    #print(len(bins))


    sf_list=[]
    tau_list=[]
    numobj_list=[]

    for i in range(0,len(bins)-1):
        n=np.where((dtarray>=bins[i]) & (dtarray<bins[i+1]))
        nobjbin=len(n[0])
        if nobjbin>=6:
            dmag1=np.abs(dmagarray[n])
            derr1=np.sqrt(sigmaarray[n])
            sf=(np.sqrt(np.pi/2.0)*dmag1-derr1)
            sff=np.mean(sf)
            sf_list.append(sff)
            numobj_list.append(nobjbin)
            #central tau for the bin
            tau_list.append((bins[i]+bins[i+1])*0.5)


    SF=np.array(sf_list)
    nob=np.array(numobj_list)
    tau=np.array(tau_list)
    nn=np.where(nob>6)
    tau=tau[nn]
    SF=SF[nn]


    return (tau/365.,SF)

def fitSF_Sch10(jd,mag,errmag,nbin,bmin,bmax):
    """
    fit the model A*tau^gamma to the SF, the fit only consider the bins with more than 6 pairs.

    inputs:
    jd: julian days array
    mag: magnitudes array
    errmag: error of magnitudes array
    nbin: size of the bin in log scale
    bmin: minimum value of the bins
    bmax: maximum value of the bins

    outputs:
    A: amplitude of the SF at 1 year
    gamma: logarithmic gradient of the change in magnitude of the SF
    """

    tau,sf = SFSchmidt10(jd,mag,errmag,nbin,bmin,bmax)

    y=np.log10(sf)
    x=np.log10(tau)
    x=x[np.where((tau<=1) & (tau>0.01))]
    y=y[np.where((tau<=1) & (tau>0.01))]
    coefficients = np.polyfit(x, y, 1)

    A=10**(coefficients[1])
    gamma=coefficients[0]

    return(gamma, A)

################################################################################
def run_fats(jd, mag, err):
    """
    function tu run fats and return the features in an array

    inputs:
    jd: julian days array
    mag: magnitudes array
    err: error of magnitudes array

    output: dictionary with the features
    """

    #list with the features to be calculated
    feature_list = [
        #'Mean',
        #'Std',
        #'Meanvariance',
        #'MedianBRP',
        #'Rcs',
        #'PeriodLS',
        #'Period_fit',
        #'Color',
        #'Autocor_length',
        #'SlottedA_length',
        #'StetsonK',
        #'StetsonK_AC',
        #'Eta_e',
        #'Amplitude',
        #'PercentAmplitude',
        #'Con',
        #'LinearTrend',
        #'Beyond1Std',
        #'FluxPercentileRatioMid20',
        #'FluxPercentileRatioMid35',
        #'FluxPercentileRatioMid50',
        #'FluxPercentileRatioMid65',
        #'FluxPercentileRatioMid80',
        #'PercentDifferenceFluxPercentile',
        #'Q31',
        'ExcessVar',
        'Pvar',
        'CAR_sigma',
        'CAR_mean',
        'CAR_tau',
        'SF_ML_amplitude',
        'SF_ML_gamma',
        'GP_DRW_sigma',
        'GP_DRW_tau',
    ]

    data_array = np.array([mag, jd, err])
    data_ids = ['magnitude', 'time', 'error']
    feat_space = FATS.FeatureSpace(featureList=feature_list, Data=data_ids)
    feat_vals = feat_space.calculateFeature(data_array).result(method='dict')

    return (feat_vals)
