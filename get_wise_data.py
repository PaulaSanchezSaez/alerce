import pandas as pd
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool

from alerce.api import AlerceAPI
alerce = AlerceAPI()

ncores=40 #number of cores used to compute the features
output = 'WISE_SDSS_data_without_wise_data.csv' #Output text file with calculated features.


df_oids = pd.read_csv('ra_dec_ZTF_20200110_without_wise_data.csv',index_col='oid') #list of objects with more than one detection
oids = df_oids.index.values
#oids = oids[0:10]

def get_wise(oid):

    try:
        a=alerce.catsHTM_crossmatch(oid, radius=10, catalog_name='all', format='pandas')
        try:
            W2 = a['WISE'].Mag_W2
            W1 = a['WISE'].Mag_W1
            W3 = a['WISE'].Mag_W3
            W4 = a['WISE'].Mag_W4
            W1_W2 = W1 - W2
            print(oid,W1_W2)

        except:
            W1,W2,W3,W4,W1_W2=-999,-999,-999,-999,-999
            print(oid,"not succesfull for WISE")

        try:
            u = a['SDSS/DR10'].modelMag_u
            g = a['SDSS/DR10'].modelMag_g
            r = a['SDSS/DR10'].modelMag_r
            i = a['SDSS/DR10'].modelMag_i
            z = a['SDSS/DR10'].modelMag_z
            u_g = u-g
            g_r = g-r
            r_i = r-i
            i_z = i-z

            print(oid,u_g,g_r,r_i)

        except:
            u,g,r,i,z=-999,-999,-999,-999,-999
            u_g = -999
            g_r = -999
            r_i = -999
            i_z = -999

            print(oid,"not succesfull for SDSS")

    except:

        W1,W2,W3,W4,W1_W2,u,g,r,i,z,u_g,g_r,r_i,i_z = -999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999,-999
        print(oid,"not succesfull")

    return([oid,W1,W2,W3,W4,W1_W2,u,g,r,i,z,u_g,g_r,r_i,i_z])

def run_parallele():


    proc_pool = Pool(ncpus=int(ncores),processes=ncores)
    results = proc_pool.map(get_wise,list(oids))

    proc_pool.close()
    proc_pool.join()


    head=['oid','W1','W2','W3','W4','W1-W2','u','g','r','i','z','u_g','g_r','r_i','i_z']


    df = pd.DataFrame(results)
    df.to_csv(output, header=head, index=None)
    print("File %s writen" % (output))
    return (results)


run_parallele()
