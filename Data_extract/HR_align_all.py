import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from utils import patient_id_age, heart_rate, align_data

# os.chdir('/Users/xiao-zy19/Desktop/Johns Hopkins/Biomedical Data Design/EICU Database/eicu-collaborative-research-database-demo-2.0.1') 
# os.chdir('/home/en580-zxia028/EICU_demo/data')
os.chdir('/home/en580-zxia028/EICU/data')

# get patient id
patient_id, patient_age, patient_offset = patient_id_age()

# add parser
parser = argparse.ArgumentParser(description='''
    Process EICU data in batches. 
    This python script allows you to process data from the EICU database in specified batch sizes, 
    or all at once.
''')
parser.add_argument('--batch', type=int, default=0, help='batch number to process')
parser.add_argument('--batch_size', default='1000', help='size of each batch or "all"')

args = parser.parse_args()

batch = args.batch
batch_size = args.batch_size

# check if batch_size is 'all'
if batch_size.lower() == 'all':
    batch_size = len(patient_id)  # set batch_size to be the number of patients
    patient_batch = [patient_id[i:i + batch_size] for i in range(0, len(patient_id), batch_size)]
    print('batch size: ', batch_size)
    print('now processing all data')
else:
    batch_size = int(batch_size)  # turn batch_size into integer
    patient_batch = [patient_id[i:i + batch_size] for i in range(0, len(patient_id), batch_size)]
    print('batch size: ', batch_size)
    print('now processing batch: ', batch, '/', len(patient_batch)-1)

# extract heart rate data
HR, _ = heart_rate(patient_batch[batch], drop_neg=True)
data_full, data_full_index = align_data(patient_batch[batch], patient_offset, HR, graph=False)

folder_path = '/home/en580-zxia028/output/hr/data_full'
file_name = 'HR_full_' + str(batch) + '.csv'
data_full.to_csv(os.path.join(folder_path, file_name), index=False)
