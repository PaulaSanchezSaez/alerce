import sys
import pandas
import math
import os
import random

input_path      = sys.argv[1]
output_dir_path = sys.argv[2]
label           = sys.argv[3]
n_partitions    = int(sys.argv[4])

#read detections
print("READ DETECTIONS")
#df = pandas.read_pickle(input_path)
df = pandas.read_csv(input_path)

#get unique oid and shuffle
print("COMPUTE UNIQUE OID AND SHUFFLE")
oid = df.oid.unique()
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
df.set_index('oid',drop=False,inplace=True)
for i in range(n_partitions):
	print("PARTITION {}".format(i))
	output_path = os.path.join( output_dir_path, '{}_{}.pickle'.format(label,i) )
	df.loc[ partitions[i] ].reset_index(drop=True).to_pickle( output_path )
