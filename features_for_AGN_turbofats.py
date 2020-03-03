#functions for turbofats
import numpy as np
import scipy.stats as st
import scipy  as sp
from scipy.stats import chi2
import turbofats as FATS
import GPy

class Pvar(Base):
    """
    Calculate the probability of a light curve to be variable.
    """
    def __init__(self):
        self.Data = ['magnitude', 'error']

    def fit(self, data):
        magnitude = data[0]
        error = data[2]


        mean_mag = np.mean(magnitude)
        nepochs = float(len(magnitude))

        chi = np.sum( (magnitude - mean_mag)**2. / error**2. )
        p_chi = chi2.cdf(chi,(nepochs-1))

        return p_chi

################################################################################


class ExcessVar(Base):
    """
    Calculate the excess variance,which is a measure of the intrinsic variability amplitude.
    """
    def __init__(self):
        self.Data = ['magnitude', 'error']

    def fit(self, data):
        magnitude = data[0]
        error = data[2]

        mean_mag=np.mean(magnitude)
        nepochs=float(len(magnitude))

        a = (magnitude-mean_mag)**2
        ex_var = (np.sum(a-error**2)/((nepochs*(mean_mag**2))))

        return ex_var

################################################################################

class GP_DRW_sigma(Base):
    """
    Based on Matthew Graham's method to model DRW with gaussian process.
    """

    def __init__(self):
        self.Data = ['magnitude', 'time', 'error']

    def fit(self, data):
        magnitude = data[0]
        t = data[1]
        err = data[2]


        mag = magnitude-magnitude.mean()
        kern = GPy.kern.OU(1)
        m = GPy.models.GPHeteroscedasticRegression(t[:, None], mag[:, None], kern)
        m['.*het_Gauss.variance'] = abs(err ** 2.)[:, None] # Set the noise parameters to the error in Y
        m.het_Gauss.variance.fix() # We can fix the noise term, since we already know it
        m.optimize()
        pars = [m.OU.variance.values[0], m.OU.lengthscale.values[0]] # sigma^2, tau

        sigmaDRW = pars[0]
        global tauDRW
        tauDRW = pars[1]
        return sigmaDRW


class GP_DRW_tau(Base):

    def __init__(self):

        self.Data = ['magnitude', 'time', 'error']

    def fit(self, data):

        try:
            return tauDRW
        except:
            print("error: please run GP_DRW_sigma first to generate values for GP_DRW_tau")


################################################################################

class SF_ML_amplitude(Base):
    """
    Fit the model A*tau^gamma to the SF, finding the maximum value of the likelihood.
    Based on Schmidt et al. 2010.
    """

    def __init__(self):
        self.Data = ['magnitude', 'time', 'error']


    def SFarray(self,jd,mag,err):
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


    def SFVmod(self,dt,A,gamma):
        """SF power model: SF = A*(dt/365)**gamma"""
        return ( A*((dt/365.0)**gamma) )


    def SFVeff2(self,dt,sigma,A,gamma):
        """SF power model plus the error"""
        return ( (self.SFVmod(dt,A,gamma))**2 + sigma )

    def SFlike_one(self,theta,dt,dmag,sigma):
        """likelihood for one value of dmag (one pair of epochs)"""

        gamma, A = theta
        aux=(1/np.sqrt(2*np.pi*self.SFVeff2(dt,sigma,A,gamma)))*np.exp(-1.0*(dmag**2)/(2.0*self.SFVeff2(dt,sigma,A,gamma)))

        return aux

    def SFlnlike(self,theta, dtarray, dmagarray, sigmaarray):
        """likelihood for the whole light curve"""
        gamma, A = theta

        aux=-1.0*np.sum(np.log(self.SFlike_one(theta,dtarray,dmagarray,sigmaarray)))

        return aux


    def SF_Lik(self, theta, t, mag, err):

        dtarray, dmagarray, sigmaarray = self.SFarray(t,mag,err)
        ndt=np.where((dtarray<=365))
        dtarray=dtarray[ndt]
        dmagarray=dmagarray[ndt]
        sigmaarray=sigmaarray[ndt]

        return self.SFlnlike(theta, dtarray, dmagarray, sigmaarray)

    def fit(self, data):
        mag = data[0]
        t = data[1]
        err = data[2]

        x0 = [0.5, 0.5]
        bnds = ((0.0, 5), (0.0, 8))

        #res = minimize(self.SF_Lik, x0, args=(t,mag,err),
        #               method='Nelder-Mead', options={'fatol': 1e-10, 'xatol': 1e-10, 'maxiter': 15000})


        res = minimize(self.SF_Lik, x0, bounds=bnds, args=(t,mag,err),
                       method='SLSQP', options={'ftol': 1e-10, 'maxiter': 15000})

        aSF_min = res.x[1]
        if (aSF_min<1e-3): aSF_min = 0.0
        global gSF_min
        gSF_min = res.x[0]
        if (aSF_min<1e-3): gSF_min = 0.0
        return aSF_min


class SF_ML_gamma(Base):

    def __init__(self):

        self.Data = ['magnitude', 'time', 'error']

    def fit(self, data):

        try:
            return gSF_min
        except:
            print("error: please run SF_ML_amplitude first to generate values for SF_ML_gamma")




################################################################################

class IAR_phi(Base):
    """
    functions to compute an IAR model with Kalman filter.
    Author: Felipe Elorrieta.
    """

    def __init__(self):
        self.Data = ['magnitude', 'time', 'error']



    def IAR_phi_kalman(self,x,t,y,yerr,standarized=True,c=0.5):
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
                xhat[0:1,i+1]=np.dot(F,xhat[0:1,i])+np.dot(np.dot(Theta,np.linalg.inv(Lambda)),(y[i]-np.dot(G,xhat[0:1,i])))
                Sighat=np.dot(np.dot(F,Sighat),F.transpose()) + Qt - np.dot(np.dot(Theta,np.linalg.inv(Lambda)),Theta.transpose())
            yhat=np.dot(G,xhat)
            out=(sum_Lambda + sum_error)/n
            if np.isnan(sum_Lambda) == True:
                out=1e10
        else:
            out=1e10
        return out

    def fit(self, data):
        magnitude = data[0]
        time = data[1]
        error = data[2]

        if np.sum(error)==0:
            error=np.zeros(len(magnitude))

        ynorm = (magnitude-np.mean(magnitude))/np.sqrt(np.var(magnitude,ddof=1))
        deltanorm = error/np.sqrt(np.var(magnitude,ddof=1))

        out=sp.optimize.minimize_scalar(self.IAR_phi_kalman,args=(time,ynorm,deltanorm),bounds=(0,1),method="bounded",options={'xatol': 1e-12, 'maxiter': 50000})

        phi = out.x
        try: phi = phi[0][0]
        except: phi = phi

        return phi

################################################################################

class CIAR_phiR(Base):
    """
    functions to compute an IAR model with Kalman filter.
    Author: Felipe Elorrieta.
    """

    def __init__(self):
        self.Data = ['magnitude', 'time', 'error']

    def CIAR_phi_kalman(self,x,t,y,yerr,mean_zero=True,standarized=True,c=0.5):
        n=len(y)
        Sighat=np.zeros(shape=(2,2))
        Sighat[0,0]=1
        Sighat[1,1]=c
        if standarized == False:
             Sighat=np.var(y)*Sighat
        if mean_zero == False:
             y=y-np.mean(y)
        xhat=np.zeros(shape=(2,n))
        delta=np.diff(t)
        Q=Sighat
        phi_R=x[0]
        phi_I=x[1]
        F=np.zeros(shape=(2,2))
        G=np.zeros(shape=(1,2))
        G[0,0]=1
        phi=complex(phi_R, phi_I)
        Phi=abs(phi)
        psi=np.arccos(phi_R/Phi)
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
                phi2_R=(Phi**delta[i])*np.cos(delta[i]*psi)
                phi2_I=(Phi**delta[i])*np.sin(delta[i]*psi)
                phi2=1-abs(phi**delta[i])**2
                F[0,0]=phi2_R
                F[0,1]=-phi2_I
                F[1,0]=phi2_I
                F[1,1]=phi2_R
                Qt=phi2*Q
                sum_Lambda=sum_Lambda+np.log(Lambda)
                Theta=np.dot(np.dot(F,Sighat),G.transpose())
                sum_error= sum_error + (y[i]-np.dot(G,xhat[0:2,i]))**2/Lambda
                xhat[0:2,i+1]=np.dot(F,xhat[0:2,i])+np.dot(np.dot(Theta,np.linalg.inv(Lambda)),(y[i]-np.dot(G,xhat[0:2,i])))
                Sighat=np.dot(np.dot(F,Sighat),F.transpose()) + Qt - np.dot(np.dot(Theta,np.linalg.inv(Lambda)),Theta.transpose())
            yhat=np.dot(G,xhat)
            out=(sum_Lambda + sum_error)/n
            if np.isnan(sum_Lambda) == True:
                out=1e10
        else:
            out=1e10
        return out

    def fit(self, data):
        magnitude = data[0]
        time = data[1]
        error = data[2]

        niter=10
        seed=1234

        ynorm = (magnitude-np.mean(magnitude))/np.sqrt(np.var(magnitude,ddof=1))
        deltanorm = error/np.sqrt(np.var(magnitude,ddof=1))

        np.random.seed()
        aux=1e10
        value=1e10
        br=0
        if np.sum(error)==0:
            deltanorm=np.zeros(len(y))
        for i in range(niter):
            phi_R=2*np.random.uniform(0,1,1)-1
            phi_I=2*np.random.uniform(0,1,1)-1
            bnds = ((-0.9999, 0.9999), (-0.9999, 0.9999))
            out=minimize(self.CIAR_phi_kalman,np.array([phi_R, phi_I]),args=(time,ynorm,deltanorm),bounds=bnds,method='L-BFGS-B')
            value=out.fun
            if aux > value:
                par=out.x
                aux=value
                br=br+1
            if aux <= value and br>1 and i>math.trunc(niter/2):
                break
            #print br
        if aux == 1e10:
           par=np.zeros(2)
        return par[0]
