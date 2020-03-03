import os
import sys
import pandas as pd
import numpy as np
from mpi4py import MPI
import time
import traceback
import paps 


dt1 = 100
dt2 = 10

def compute_det_features(det,fid):

    det_fid     = det[ det.fid == fid ]

    count = len(det_fid)
    result = { 'n_det_fid_%d'%(fid)                     : count,
               'n_neg_%d'%(fid)                         : np.nan,
               'delta_mjd_fid_%d'%(fid)                 : np.nan,
               'delta_mag_fid_%d'%(fid)                 : np.nan,
               'positive_fraction_%d'%(fid)             : np.nan,
               'min_mag'                                : np.nan,
               'mean_mag'                               : np.nan,
               'first_mag'                              : np.nan,
               'paps_ratio_%d'%(fid)                         : np.nan,
               'paps_low_%d'%(fid)                           : np.nan,
               'paps_high_%d'%(fid)                          : np.nan,
               'paps_non_zero_%d'%(fid)                      : np.nan,
               'paps_PN_flag_%d'%(fid)                       : np.nan,
               'sgscore1'                               : det_fid['sgscore1'].median(),
               'rb'                                     : det_fid['rb'].median()
    }

    if count == 0:
        return result

    n_pos = len(det_fid[ det_fid.isdiffpos > 0 ])
    n_neg = len(det_fid[ det_fid.isdiffpos < 0 ])

    min_mag   = det_fid['magpsf_corr'].min()
    first_mag = det_fid.iloc[0]['magpsf_corr']

    result2 = {
               'n_pos_%d'%(fid)                         : n_pos,
               'n_neg_%d'%(fid)                         : n_neg,
               'delta_mjd_fid_%d'%(fid)                 : det_fid.iloc[count-1]['mjd'] - det_fid.iloc[0]['mjd'],
               'delta_mag_fid_%d'%(fid)                 : det_fid['magpsf_corr'].max() - min_mag,
               'positive_fraction_%d'%(fid)             : n_pos/(n_pos+n_neg),
               'min_mag'                                : min_mag,
               'mean_mag'                               : det_fid['magpsf_corr'].mean(),
               'first_mag'                              : first_mag
    }

    ratio,low,high,non_zero,PN_flag = paps.statistics(det_fid.magpsf_corr.values,
                                                      det_fid.sigmapsf_corr.values,
                                                          det_fid.mjd.values,
                                                          dt1,
                                                          dt2)
    result2['paps_ratio_%d'%(fid)]=ratio
    result2['paps_low_%d'%(fid)] = low
    result2['paps_high_%d'%(fid)] = high
    result2['paps_non_zero_%d'%(fid)] = non_zero
    result2['paps_PN_flag_%d'%(fid)] = PN_flag



    result.update(result2)

    return result


def compute_before_features(det_result,non_det,fid):

    non_det_fid  = non_det[ non_det.fid == fid ]

    count = len(non_det_fid)
    result = {
                'n_non_det_before_fid_%d'%(fid)             : count,
                'max_diffmaglim_before_fid_%d'%(fid)        : np.nan,
                'median_diffmaglim_before_fid_%d'%(fid)     : np.nan,
                'last_diffmaglim_before_fid_%d'%(fid)       : np.nan,
                'last_mjd_before_fid_%d'%(fid)              : np.nan,
                'dmag_non_det_fid_%d'%(fid)                 : np.nan,
                'dmag_first_det_fid_%d'%(fid)               : np.nan
    }

    if len(non_det_fid) == 0:
        return result

    last_element = non_det_fid.iloc[count-1]

    median_diffmaglim_before = non_det_fid['diffmaglim'].median()

    result2 = {
                'max_diffmaglim_before_fid_%d'%(fid)        : non_det_fid['diffmaglim'].max(),
                'median_diffmaglim_before_fid_%d'%(fid)     : median_diffmaglim_before,
                'last_diffmaglim_before_fid_%d'%(fid)       : last_element['diffmaglim'],
                'last_mjd_before_fid_%d'%(fid)              : last_element['mjd'],
                'dmag_non_det_fid_%d'%(fid)                 : median_diffmaglim_before - det_result['min_mag'],
                'dmag_first_det_fid_%d'%(fid)               : last_element['diffmaglim'] - det_result['first_mag']
    }

    result.update(result2)

    return result

def compute_after_features(non_det,fid):

    non_det_fid = non_det[ non_det.fid == fid ]

    count = len(non_det_fid)
    result = {
               'n_non_det_after_fid_%d' % (fid)         : count,
               'max_diffmaglim_after_fid_%d' % (fid)    : np.nan,
               'median_diffmaglim_after_fid_%d' % (fid) : np.nan
    }
    if count == 0:
        return result

    result2 = {
               'max_diffmaglim_after_fid_%d' % (fid)    : non_det_fid['diffmaglim'].max(),
               'median_diffmaglim_after_fid_%d' % (fid) : non_det_fid['diffmaglim'].median()
    }

    result.update(result2)

    return result


def compute_features(detections,non_detections):

    firstmjd = np.Infinity

    if len(detections) > 0:
        det_copy = detections.copy()
        det_copy.sort_values('mjd',inplace=True)
        firstmjd = det_copy.iloc[0]['mjd']

    non_det_copy = non_detections.copy()
    non_det_copy.sort_values('mjd',inplace=True)

    #DET

    #g-band
    fid = 1
    det_result_1= compute_det_features(det_copy,fid)

    #r-band (fid=2)
    fid = 2
    det_result_2 = compute_det_features(det_copy,fid)

    #BEFORE NON DET
    non_det_before = non_det_copy[ non_det_copy.mjd < firstmjd ]

    #g-band
    fid = 1
    before_result_1= compute_before_features(det_result_1,non_det_before,fid)

    #r-band (fid=2)
    fid = 2
    before_result_2 = compute_before_features(det_result_2,non_det_before,fid)

    #AFTER NON DET
    non_det_after = non_det_copy[ non_det_copy.mjd > firstmjd ]

    #g-band
    fid = 1
    after_result_1= compute_after_features(non_det_after,fid)

    #r-band (fid=2)
    fid = 2
    after_result_2 = compute_after_features(non_det_after,fid)

    #g-r
    colors = {
                'g-r_max'  : det_result_1['min_mag'] - det_result_2['min_mag'],
                'g-r_mean' : det_result_1['mean_mag'] - det_result_2['mean_mag']
    }

    del det_result_1['min_mag']
    del det_result_1['mean_mag']
    del det_result_1['first_mag']

    del det_result_2['min_mag']
    del det_result_2['mean_mag']
    del det_result_2['first_mag']

    #merge results
    result = {
                **det_result_1,
                **det_result_2,
                **before_result_1,
                **before_result_2,
                **after_result_1,
                **after_result_2,
                **colors
    }

    return result

def compute_features_dataframe(process_id,input_directory,features_directory):

    try:
        #READ
        input_path = os.path.join(input_directory,'detections_%d.pickle' % (process_id) )
        det=pd.read_pickle(input_path)
        det.sigmapsf_corr = det.sigmapsf_corr.astype("float64")
        det.magpsf_corr = det.magpsf_corr.astype("float64")

        det = det[(det.sigmapsf_corr > 0) & (det.sigmapsf_corr < 1.0)]
        det = det[det.rb >= 0.55]

        input_path = os.path.join(input_directory,'non_detections_%d.pickle' % (process_id) )
        non_det=pd.read_pickle(input_path)

        #ITERATE
        oid = det['oid'].unique()

        det.set_index('oid',inplace=True,drop=False)
        non_det.set_index('oid',inplace=True,drop=False)

        results = []
        for idx in oid:

            det_oid        = det[ det.index == idx]
            non_det_oid    = non_det[ non_det.index == idx ]

            result = compute_features(det_oid,non_det_oid)
            result['oid'] = idx
            results.append(result)

        #MERGE
        df = pd.DataFrame(results)
        df.set_index('oid',inplace=True)

        #WRITE
        output_path = os.path.join(features_directory, 'features_%d.pickle' % (process_id) )
        df.to_pickle(output_path)
    except:
        print("Error dataframe = %d" %(process_id), file=sys.stderr )
        traceback.print_exc(file=sys.stderr)


if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    start = time.time()

    input_directory    = sys.argv[1]
    features_directory = sys.argv[2]

    process_id = rank
    compute_features_dataframe(process_id,input_directory,features_directory)

    end = time.time()
    total = end-start
    print("process %s=%s" % ( rank, total ) )
