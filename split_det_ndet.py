import sys
import pandas
import math
import os
import random

det_path      = sys.argv[1]
non_det_path      = sys.argv[2]
output_dir_path = sys.argv[3]
n_partitions    = int(sys.argv[4])

#read detections
print("READ DETECTIONS")
#det = pandas.read_pickle(det_path)
det = pandas.read_csv(det_path)
det.index = det.oid

#read non detections
print("READ NON DETECTIONS")
#ndet = pandas.read_pickle(non_det_path)
ndet = pandas.read_csv(non_det_path)
ndet.index = ndet.oid
#get unique oid and shuffle
print("COMPUTE UNIQUE OID AND SHUFFLE")
oid = det.oid.unique()
random.shuffle(oid)

#compute oid partitions
print("COMPUTE OID PARTITIONS")
n = len(oid)
partitions = []
delta = math.floor(n / n_partitions)
for i in range(n_partitions-1):
	partitions.append( oid[i*delta:(i+1)*delta] )
partitions.append( oid[ (n_partitions-1)*delta: ] )

#split and save detections
print("SPLIT AND SAVE PARTITIONS")
det.set_index('oid',drop=False,inplace=True)
for i in range(n_partitions):
    print("PARTITION {}".format(i))
    print("DET")
    output_path = os.path.join( output_dir_path, 'detections_{}.pickle'.format(i) )
    det.loc[ partitions[i] ].reset_index(drop=True).to_pickle( output_path )
    print("NON DET")
    output_path = os.path.join( output_dir_path, 'non_detections_{}.pickle'.format(i) )
    ndet.loc[ partitions[i] ].reset_index(drop=True).to_pickle( output_path )
