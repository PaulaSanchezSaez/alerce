import os
import astropy.io.fits as pf
import numpy as np
import pandas as pd
import glob
from pathos.multiprocessing import ProcessingPool as Pool
import time
import var_features_June2019 as vf
import sys, getopt
import ast
import turbofats as FATS

###############################################################################
#modify these and only these variables:

folder = '../test_QUEST_lc/train_all/' #Folder where FITS files are located.

ncores=10 #number of cores used to compute the features

output = '../test_QUEST_lc/var_features/QUEST_SDSS_train_samp_all.csv' #Output text file with calculated features.

patern = 'bin3*fits' # Pattern with wildcard to match desired FITS files.

use_z = 'False' # If True compute rest frame time for light curves

train_samp = 'True' # the lc correspond to the training sample?
###############################################################################
# To modify the parameters from the terminal
myopts, args = getopt.getopt(sys.argv[1:],"f:n:p:o:z:t:")

###############################
# o == option
# a == argument passed to the o
###############################
for o, a in myopts:
    if o == '-f':
        folder = a
    elif o == '-n':
        ncores = a
    elif o == '-p':
        patern = a
    elif o == '-o':
        output = a
    elif o == '-z':
        use_z = a
    elif o == '-t':
        train_samp = a


folder = str(folder)

patern = str(patern)

ncores = int(ncores)
print(ncores)

output = str(output)

use_z = ast.literal_eval(use_z)

train_samp = ast.literal_eval(train_samp)

###############################################################################
###############################################################################
#feature list for FATS

featurelist = [
'Mean',
'Std',
'Meanvariance',
'MedianBRP',
'Rcs',
#'PeriodLS',
#'Period_fit',
#'Color',
'Autocor_length',
#'SlottedA_length',
'StetsonK',
#'StetsonK_AC',
'Eta_e',
#'Amplitude',
'PercentAmplitude',
'Con',
'LinearTrend',
'Beyond1Std',
#'FluxPercentileRatioMid20',
#'FluxPercentileRatioMid35',
#'FluxPercentileRatioMid50',
#'FluxPercentileRatioMid65',
#'FluxPercentileRatioMid80',
#'PercentDifferenceFluxPercentile',
'Q31',
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

################################################################################
#function to run FATS

def fats_feats(jd, mag, errmag):

    lc = np.array([mag, jd, errmag])

    a = FATS.FeatureSpace(featureList=featurelist)

    a=a.calculateFeature(lc).result(method='dict')

    return(a)


###############################################################################
#funcion to read every lc

def read_lc(lc_name):

    print("########## reading lc %s  ##########" % (lc_name))

    arch = pf.open(lc_name)

    if train_samp:

        head = arch[0].header
        datos = arch[1].data
        jd = datos['JD']
        flux = datos['fluxQ']
        errflux = datos['errfluxQ']
        mag = datos['Q']
        errmag = datos['errQ']
        zspec = head['ZSPEC']
        type = head['TYPE_SPEC']
        umag = head['UMAG']
        gmag= head['GMAG']
        rmag = head['RMAG']
        imag = head['IMAG']
        zmag = head['ZMAG']
        ra = head['ALPHA']
        dec = head['DELTA']


        return(ra,dec,jd,flux,errflux,mag,errmag,umag,gmag,rmag,imag,zmag,zspec,type)

    else:

        head = arch[0].header
        datos = arch[1].data
        jd = datos['JD']
        flux = datos['fluxQ']
        errflux = datos['errfluxQ']
        mag = datos['Q']
        errmag = datos['errQ']
        ra = head['ALPHA']
        dec = head['DELTA']

        return(ra,dec,jd,flux,errflux,mag,errmag)

###############################################################################

def comp_feat(lc_name):

    if train_samp:


        ra,dec,jd,flux,errflux,mag,errmag,umag,gmag,rmag,imag,zmag,zspec,type = read_lc(lc_name)

        num_epochs = len(jd)
        time_range = jd[-1] - jd[0]
        time_rest =  time_range/(1.0+zspec)

        #print(num_epochs,time_range )

        u_g = umag - gmag
        g_r = gmag - rmag
        r_i = rmag - imag
        i_z = imag - zmag

        if use_z: jd = jd/(1.0+zspec)

        #try:
        phi, tau_iar = vf.IAR_kalman(jd, mag, errmag)
        #except:
        #    phi, tau_iar = -99,-99


        try:
            gamma_sch10,A_sch10 = vf.fitSF_Sch10(jd, mag, errmag,0.1,10,2000)
        except:
            gamma_sch10,A_sch10 = -99,-99

        other_feat = [ra,dec,zspec,type,num_epochs,time_range,time_rest,umag,gmag,rmag,imag,zmag,u_g,g_r,r_i,i_z,phi, tau_iar,gamma_sch10,A_sch10]

        fats_results = fats_feats(jd, mag, errmag)

        fats_list = []

        for feat in featurelist:

            fats_list.append(fats_results[feat])

        feat_list = other_feat+ fats_list


        return(feat_list)

    else:

        ra,dec,jd,flux,errflux,mag,errmag = read_lc(lc_name)

        num_epochs = len(jd)
        time_range = jd[-1] - jd[0]


        if use_z: jd = jd/(1.0+zspec)

        try:
            phi, tau_iar = vf.IAR_kalman(jd, mag, errmag)
        except:
            phi, tau_iar = -99,-99


        try:
            gamma_sch10,A_sch10 = vf.fitSF_Sch10(jd, mag, errmag,0.1,10,2000)
        except:
            gamma_sch10,A_sch10 = -99,-99

        other_feat = [ra,dec,num_epochs,time_range,phi, tau_iar,gamma_sch10,A_sch10]

        fats_results = fats_feats(jd, mag, errmag)

        fats_list = []

        for feat in featurelist:

            fats_list.append(fats_results[feat])

        feat_list = other_feat+ fats_list


        return(feat_list)


###############################################################################

def multi_run_wrapper(args):
    #function necessary for the use of pool.map with different arguments
    return comp_feat(*args)



def run_parallele():

    agn=sorted(glob.glob(folder+patern))

    print("number of lc to process = %d" % (len(agn)))
    '''
    pool = Pool(processes=ncores)
    results = pool.map(comp_feat,list(agn))
    pool.close()
    pool.join()
    '''

    proc_pool = Pool(ncpus=int(ncores),processes=ncores)
    results = proc_pool.map(comp_feat,list(agn))

    proc_pool.close()
    proc_pool.join()

    if train_samp:
        head=['ra','dec','zspec','TYPE','num_epochs','time_range','time_rest','umag','gmag','rmag','imag','zmag','u_g','g_r','r_i','i_z','phi', 'tau_iar','gamma_sch10','A_sch10']
        for feat in featurelist:
            head.append(feat)


    else:
        head=['ra','dec','num_epochs','time_range','phi', 'tau_iar','gamma_sch10','A_sch10']
        for feat in featurelist:
            head.append(feat)


    df = pd.DataFrame(results)
    df.to_csv(output, header=head, index=None)
    print("File %s writen" % (output))
    return (results)


run_parallele()
