import numpy as np
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
from astroML.time_series import generate_damped_RW
import time


def lc_sim(nn, delt, mean_lc, model, alpha):

    '''
    Python function to simulate a light curve given a model power spectrum
    Based on paper Timmer, J., & Koenig, M. 1995, A&A, 300, 707
    nn = number of points in simulated light curve
    delt = time sampling of simulated light curve in seconds
    mean_lc = mean value of simulated light curve
    model = "unbroken" or "slow" to designate what type of power spectrum the simulated
             light curve will be based on
    '''

    #Seed the random number generator
    #np.random.seed(6543289)

    #Fourier frequencies
    fi = np.arange(1, nn/2+1, dtype='float64')/nn/delt
    #print(fi)

    '''
    Power at each frequency depending on power spectrum model
    Unbroken model: S = N*(nu/nu0)^(-beta)
    Slowly bending "knee" model: S = N*nu^(alpha_low)/(1+(nu/nu_k))^(alpha_hi - alpha_lo)

    '''
    if model == 'power-law':

        a = 500.0
        nu_0 = 0.005
        beta = alpha
        noise = 0.0

        s = a*(fi/nu_0)**(-beta) + noise

    elif model == 'bending-pl':

        a = 1000.0
        nu_knee = 0.005
        alpha_lo = 1.0
        alpha_hi = alpha
        noise = 0.0

        s = (a*fi**(-alpha_lo))/(1+(fi/nu_knee))**(alpha_hi - alpha_lo) + noise

    #Generate two sets of normally distributed random numbers
    aa = np.random.randn(len(fi))
    bb = np.random.randn(len(fi))

    #Fourier transform of light curve  = SQRT(S/2) * (A + B*i)
    flc = np.sqrt(.5*s)*(aa + bb*1.j)
    #plt.plot(np.real(flc))

    if np.mod(nn, 2) == 0:
        flc[-1] = np.sqrt(.5*s[-1])*1

    del aa, bb, s

    #Put the mean of the light curve at frequency = 0
    #flc = np.hstack([mean_lc, flc])

    #Take the inverse fourier transform to generate synthetic light curve
    lc = np.fft.irfft(flc, n=nn)

    return lc+mean_lc




def gen_lc_long(seed,time_range,dtime,mag,errmag,model,alpha,sampling,timesamp):
    #function to generate a light curve with a given PSD model
    #seed: needed to avoid problems in the random number generation when the method is used with multiprocessing
    #time_range=2000 light curve length
    #dtime=2 candence
    #mag=19.5 mean magnitude
    #errmag=0.03 photometric error
    #model = "unbroken" or "slow"
    #alpha: for "unbroken" model is the value os beta (the other parameters are fixed),
    #for "slow" model is the alpha_hi value, the other parameters are fixed.
    #sampling: True if you are using the sampling of a given light curve. False if you whant a regularly sampled light curve
    #timesamp: if sampling is True, timesamp is the array with the JDs of the desired light curve.
    tbase=36500
    dtbase=1
    t_drwbase=np.arange(0,tbase,dtbase)

    #t_drw=np.arange(0,time_range,dtime)

    np.random.seed(int((time.clock()+seed*np.random.rand())))
    #np.random.seed(int(seed))
    y=lc_sim(tbase, dtbase, mag, model, alpha)

    ysig=np.zeros(len(y))
    #ysig_obs=np.ones(len(y))*errmag
    #y_obs = y+np.random.normal(0., np.random.normal(errmag, 0.005, len(y)))#np.random.normal(y, errmag)
    ysig_obs=np.abs(np.random.normal(errmag, 0.004, len(y)))#np.ones(len(y))*errmag
    y_obs = y+np.random.normal(0., ysig_obs)  #np.random.normal(y, errmag)



    if sampling:

        timesamp=np.round(timesamp,decimals=0).astype(np.int)
        tstar = np.random.randint(14600,tbase-np.int((timesamp[-1]-timesamp[0])),size=1)
        t_drw = tstar+((timesamp-timesamp[0]))
        print(t_drw)

        y=y[t_drw]
        ysig=ysig[t_drw]
        ysig_obs=ysig_obs[t_drw]
        y_obs=y_obs[t_drw]

        t_drw=t_drw-t_drw[0]


    else:

        tstar = np.random.randint(14600,tbase-time_range,size=1)
        tend = tstar+time_range
        t_drw=np.arange(tstar,tend,dtime)
        y=y[t_drw]
        ysig=ysig[t_drw]
        ysig_obs=ysig_obs[t_drw]
        y_obs=y_obs[t_drw]

        t_drw=t_drw-t_drw[0]



    return (t_drw,y,ysig,y_obs,ysig_obs)



def gen_DRW_long(seed,time_range=2000,dtime=2,mag=19.5,errmag=0.03,tau=400,SFinf=0.2,sampling=False,timesamp=0.0):
    #function to generate a light curve with a DRW model, for a given tau and sigma
    #seed: needed to avoid problems in the random number generation when the method is used with multiprocessing
    #recomended values:
    #time_range=2000 light curve length
    #dtime=2 candence
    #mag=19.5 mean magnitude
    #errmag=0.03 photometric error
    #tau=400 tau for the DRW model
    #SFinf=0.2  amplitude of the variability at long time scales
    #sampling: True if you are using the sampling of a given light curve. False if you whant a regularly sampled light curve
    #timesamp: if sampling is True, timesamp is the array with the JDs of the desired light curve.
    tbase=36500
    dtbase=1
    t_drwbase=np.arange(0,tbase,dtbase)

    #t_drw=np.arange(0,time_range,dtime)

    #np.random.seed(int((time.clock()+seed)))
    np.random.seed(int(seed))

    #sigma=SFinf*np.sqrt(2.0/tau)
    #y = mag + cm.carma_process(t_drw, sigma, np.atleast_1d(-1.0 / tau),ma_coefs=[1.0])

    #generating clean light curve
    y=generate_damped_RW(t_drwbase, tau, z=0, SFinf=SFinf,xmean=mag)
    ysig=np.zeros(len(y))#array with ceros for the clean light curve photometric errors
    #adding noise to the clean light curve
    ysig_obs=np.abs(np.random.normal(errmag, 0.004, len(y)))#np.ones(len(y))*errmag
    y_obs = y+np.random.normal(0., ysig_obs)  #np.random.normal(y, errmag)

    #plt.plot(t_drwbase,y,'b.')
    #plt.xlabel('days')
    #plt.ylabel(r'mag')
    #plt.xlim(0,36500)
    #plt.savefig('long_lc.pdf')
    #plt.show()

    if sampling:

        timesamp=np.round(timesamp,decimals=0).astype(np.int)
        tstar = np.random.randint(14600,tbase-np.int((timesamp[-1]-timesamp[0])),size=1)
        t_drw = tstar+((timesamp-timesamp[0]))
        #print(t_drw)

        y=y[t_drw]
        ysig=ysig[t_drw]
        ysig_obs=ysig_obs[t_drw]
        y_obs=y_obs[t_drw]

        t_drw=t_drw-t_drw[0]


    else:

        tstar = np.random.randint(14600,tbase-time_range,size=1)
        tend = tstar+time_range
        t_drw=np.arange(tstar,tend,dtime)
        y=y[t_drw]
        ysig=ysig[t_drw]
        ysig_obs=ysig_obs[t_drw]
        y_obs=y_obs[t_drw]

        t_drw=t_drw-t_drw[0]



    '''
    plt.plot(t_drw,y,'b*')
    plt.xlabel('days')
    plt.ylabel(r'mag')
    #plt.savefig('final_lc.pdf')
    plt.show()

    plt.errorbar(t_drw,y_obs,yerr=ysig_obs,fmt='b*')
    plt.xlabel('days')
    plt.ylabel(r'mag')
    #plt.savefig('final_lc_noise.pdf')
    plt.show()
    '''

    return (t_drw,y,ysig,y_obs,ysig_obs)


def white_noise(seed,time_range=2000,dtime=2,mag=19.5,errmag=0.03,amplitude=0.3,sampling=False,timesamp=0.0):

    tbase=36500
    dtbase=1
    t_drwbase=np.arange(0,tbase,dtbase)


    np.random.seed(int((time.clock()+seed)))

    y=np.random.normal(mag, amplitude, size=len(t_drwbase))#generate_damped_RW(t_drwbase, tau, z=0, SFinf=SFinf,xmean=mag)
    ysig=np.zeros(len(y))#array with ceros for the clean light curve photometric errors
    #adding noise to the clean light curve
    #ysig_obs=np.ones(len(y))*errmag
    #y_obs = y+np.random.normal(0., np.random.normal(errmag, 0.005), len(y))  #np.random.normal(y, errmag)
    ysig_obs=np.abs(np.random.normal(errmag, 0.004, len(y)))#np.ones(len(y))*errmag
    y_obs = y+np.random.normal(0., ysig_obs)  #np.random.normal(y, errmag)

    if sampling:

        timesamp=np.round(timesamp,decimals=0).astype(np.int)
        tstar = np.random.randint(14600,tbase-np.int((timesamp[-1]-timesamp[0])),size=1)
        t_drw = tstar+((timesamp-timesamp[0]))
        #print(t_drw)

        y=y[t_drw]
        ysig=ysig[t_drw]
        ysig_obs=ysig_obs[t_drw]
        y_obs=y_obs[t_drw]

        t_drw=t_drw-t_drw[0]


    else:

        tstar = np.random.randint(14600,tbase-time_range,size=1)
        tend = tstar+time_range
        t_drw=np.arange(tstar,tend,dtime)
        y=y[t_drw]
        ysig=ysig[t_drw]
        ysig_obs=ysig_obs[t_drw]
        y_obs=y_obs[t_drw]

        t_drw=t_drw-t_drw[0]

    return (t_drw,y,ysig,y_obs,ysig_obs)
