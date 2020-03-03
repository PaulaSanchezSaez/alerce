import psycopg2
import pandas as pd
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
import numpy.core.defchararray as np_f
from astropy import units as u
from astropy.coordinates import SkyCoord
import turbofats as FATS

################################################################################

#modify these and only these variables:

detections_file = '../training_set_v2/detections.csv' #Folder where FITS files are located.

oids_file = '../training_set_v2/labels.pkl'

ncores=1 #number of cores used to compute the features

output = '../var_features/ZTF_trainingset_features_20190912.csv' #Output text file with calculated features.

################################################################################
#reading the files

labels_df = pd.read_pickle(oids_file)
oids_list = list(labels_df.index.values)
oids_list = sorted(oids_list)
#oids_list = oids_list[:100]
################################################################################

#connecting to alerce data base

params = {'dbname': 'ztf_v2', 'user': 'alerce', 'host': 'db.alerce.online', 'password': 'ETgW4GTdR337gjP7'}

conn = psycopg2.connect(dbname=params['dbname'], user=params['user'], host=params['host'], password=params['password'])
cur = conn.cursor()

################################################################################

# Function to query data more easily

def sql_query(query):
    cur.execute(query)
    result = cur.fetchall()

    # Extract the column names
    col_names = []
    for elt in cur.description:
        col_names.append(elt[0])

    #Convert to dataframe
    df = pd.DataFrame(np.array(result), columns = col_names)
    return(df)

################################################################################

################################################################################


featurelist = [
 'Amplitude',
 'AndersonDarling',
 'Autocor_length',
 'Beyond1Std',
 'Con',
 'Eta_e',
 'Gskew',
 'MaxSlope',
 'Mean',
 'Meanvariance',
 'MedianAbsDev',
 'MedianBRP',
 'PairSlopeTrend',
 'PercentAmplitude',
 'Q31',
 'PeriodLS_v2',
 'Period_fit_v2',
 'Psi_CS_v2',
 'Psi_eta_v2',
 'Rcs',
 'Skew',
 'SmallKurtosis',
 'Std',
 'StetsonK',
 'Pvar',
 'ExcessVar',
 'GP_DRW_sigma',
 'GP_DRW_tau',
 'SF_ML_amplitude',
 'SF_ML_gamma',
 'SF_amplitude',
 'SF_gamma',
 'IAR_phi',
 'LinearTrend',
 'Harmonics'
]

################################################################################
#function to run FATS

def fats_feats(jd, mag, errmag):

    lc = np.array([mag, jd, errmag])

    a = FATS.FeatureSpace(featureList=featurelist)

    #a=a.calculateFeature(lc).result(method='dict')
    a=a.calculateFeature(lc).result().tolist()

    return(a)




################################################################################


def comp_feat(oid):

    print("downloading lc %s " % (oid))

    query = "select  mjd, fid, ra, dec, magnr, isdiffpos, magpsf, sigmapsf, magpsf_corr, sigmapsf_corr, distpsnr1, sgscore1 from detections where oid='"+str(oid)+"'"

    lc = sql_query(query)
    lc = lc.sort_values(['mjd'])
    lc.drop_duplicates(['mjd'], inplace=True)
    lc = lc[lc['sigmapsf_corr']<1.0]

    print("processing lc %s" % (oid))

    sgscore1 = np.median(lc['sgscore1'].values)
    ra = np.mean(lc['ra'].values)
    dec = np.mean(lc['dec'].values)

    c_icrs = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')

    gal = c_icrs.galactic
    gal_b = gal.b.value
    gal_l = gal.l.value

    print("object %s with ra %f dec %f  and gal_b %f gal_l %f" % (oid, ra,dec, gal_b, gal_l))

    other_feat = [oid,ra,dec,gal_b,gal_l,sgscore1]

    #separating the bands
    lc_g = lc[lc['fid']==1]
    lc_r = lc[lc['fid']==2]

    #g band
    jd_g = lc_g['mjd'].values
    mag_g = lc_g['magpsf_corr'].values
    errmag_g = lc_g['sigmapsf_corr'].values

    num_epochs_g = len(jd_g)

    if num_epochs_g>5:
        time_range_g = jd_g[-1] - jd_g[0]
        try:

            fats_results = fats_feats(jd_g, mag_g, errmag_g)
        except:
            fats_results = np.ones(len(featurelist)+13)*-99

    else:
        time_range_g = -99
        fats_results = np.ones(len(featurelist)+13)*-99

    fats_list_g = [num_epochs_g,time_range_g]

    for feat in fats_results:

        fats_list_g.append(feat)

    #r band
    jd_r = lc_r['mjd'].values
    mag_r = lc_r['magpsf_corr'].values
    errmag_r = lc_r['sigmapsf_corr'].values


    num_epochs_r = len(jd_r)

    if num_epochs_r>5:
        time_range_r = jd_r[-1] - jd_r[0]

        try:
            fats_results = fats_feats(jd_r, mag_r, errmag_r)
        except:
            fats_results = np.ones(len(featurelist)+13)*-99

    else:
        time_range_r = -99
        fats_results = np.ones(len(featurelist)+13)*-99


    fats_list_r = [num_epochs_r,time_range_r]

    for feat in fats_results:

        fats_list_r.append(feat)



    feat_list = other_feat+ fats_list_g + fats_list_r


    return(feat_list)

###############################################################################

def multi_run_wrapper(args):
    #function necessary for the use of pool.map with different arguments
    return comp_feat(*args)



def run_parallele():

    obj=oids_list

    print("number of lc to process = %d" % (len(obj)))
    '''
    pool = Pool(processes=ncores)
    results = pool.map(comp_feat,list(obj))
    pool.close()
    pool.join()
    '''

    proc_pool = Pool(ncpus=int(ncores),processes=ncores)
    results = proc_pool.map(comp_feat,list(obj))

    proc_pool.close()
    proc_pool.join()


    head=['oid','ra','dec','gal_b','gal_l','sgscore1','num_epochs_g','time_range_g']
    for feat in featurelist[:-1]:
        head.append(feat+'_g')

    feat_harmonics_g = ['Harmonics_mag_1_g','Harmonics_mag_2_g','Harmonics_mag_3_g','Harmonics_mag_4_g','Harmonics_mag_5_g','Harmonics_mag_6_g','Harmonics_mag_7_g','Harmonics_phase_2_g','Harmonics_phase_3_g','Harmonics_phase_4_g','Harmonics_phase_5_g','Harmonics_phase_6_g','Harmonics_phase_7_g','Harmonics_mse_g']

    head = head + feat_harmonics_g

    head.append('num_epochs_r')
    head.append('time_range_g')

    for feat in featurelist[:-1]:
        head.append(feat+'_r')

    feat_harmonics_r = ['Harmonics_mag_1_r','Harmonics_mag_2_r','Harmonics_mag_3_r','Harmonics_mag_4_r','Harmonics_mag_5_r','Harmonics_mag_6_r','Harmonics_mag_7_r','Harmonics_phase_2_r','Harmonics_phase_3_r','Harmonics_phase_4_r','Harmonics_phase_5_r','Harmonics_phase_6_r','Harmonics_phase_7_r','Harmonics_mse_r']

    head = head + feat_harmonics_r

    df = pd.DataFrame(results)
    df.to_csv(output, header=head, index=None)
    print("File %s writen" % (output))
    return (results)


run_parallele()
