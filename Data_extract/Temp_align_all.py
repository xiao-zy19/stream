import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import time
from utils import patient_id_age, heart_rate, align_data, temp


start_time_all = time.time()

# os.chdir('/Users/xiao-zy19/Desktop/Johns Hopkins/Biomedical Data Design/EICU Database/eicu-collaborative-research-database-demo-2.0.1') 
os.chdir('/home/en580-zxia028/EICU/data')
folder_path = '/home/en580-zxia028/output/temp'

# get patient id
patient_id, patient_age, patient_offset = patient_id_age()

# add parser
parser = argparse.ArgumentParser(description='''
    Process EICU data in batches. 
    This python script allows you to process data from the EICU database in specified batch sizes, 
    or all at once.
''')
parser.add_argument('--batch', type=int, default=0, help='batch number to process')
parser.add_argument('--batch_size', default='all', help='size of each batch or "all"')

args = parser.parse_args()

batch = args.batch
batch_size = args.batch_size

# check if batch_size is 'all'
if batch_size.lower() == 'all':
    batch_size = len(patient_id)  # set batch_size to be the number of patients
    patient_batch = [patient_id[i:i + batch_size] for i in range(0, len(patient_id), batch_size)]
    # patient_batch = patient_id
    file_name = 'temp_all.csv'
    print('batch size: ', batch_size)
    print('now processing all data')
else:
    batch_size = int(batch_size)  # turn batch_size into integer
    patient_batch = [patient_id[i:i + batch_size] for i in range(0, len(patient_id), batch_size)]
    file_name = 'temp_full_' + str(batch) + '.csv'
    print('batch size: ', batch_size)
    print('now processing batch: ', batch, '/', len(patient_batch)-1)

print(patient_batch[batch].shape)
# extract heart rate data
Temp, _ = temp(patient_batch[batch])
data_full, data_full_index = align_data(patient_batch[batch], patient_offset, Temp, graph=False)

data_full.to_csv(os.path.join(folder_path, file_name), index=False)

end_time_all = time.time()
print("Execution time: ", end_time_all - start_time_all, " seconds")
