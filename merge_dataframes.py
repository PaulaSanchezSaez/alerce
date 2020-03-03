import sys
import os
import pandas

input_directory = sys.argv[1]
output_path     = sys.argv[2]

names = os.listdir(input_directory)

array = []
for name in names:
	print(name)
	input_path = os.path.join( input_directory, name )
	df = pandas.read_pickle( input_path )
	array.append( df )

df = pandas.concat( array )
df.to_pickle( output_path)
