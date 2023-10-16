import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def heart_rate(patient_id, 
               file_name1='data/vitalPeriodic.csv',
               file_name2='data/nurseCharting.csv'
               ):
    
    """
    Function to extract heart rate values.

    Parameters:
        patient_id: the list of wanted patient id
        file_name1: the file name of vitalPeriodic.csv
        file_name2: the file name of nurseCharting.csv
    Return:
        HR: the dataframe of heart rate data, including patientunitstayid, observationoffset, heartrate
        HR_index: the series of the index of the first occurrence of each patient
    """
    # usage:
    # If we want the i th patient data (i starts from 0)
    # use HR.iloc[HR_index[i]:HR_index[i+1]]
    # i = 0 # the first patient
    # print(f'HeartRate data for patient {i+1}: \n{HR.iloc[HR_index[i]:HR_index[i+1]]}')
    
    print('Loading Heart Rate Data...')
    start_time = time.time()
    # Load data
    df_vitalPeriodic = pd.read_csv(file_name1)
    df_vitalPeriodic.sort_values(by=['patientunitstayid', 'observationoffset'], inplace=True)
    
    df_nurseCharting = pd.read_csv(file_name2)
    df_nurseCharting.sort_values(by=['patientunitstayid', 'nursingchartoffset'], inplace=True)
    
    # select wanted patient
    df_vitalPeriodic = df_vitalPeriodic[df_vitalPeriodic['patientunitstayid'].isin(patient_id)]
    df_nurseCharting = df_nurseCharting[df_nurseCharting['patientunitstayid'].isin(patient_id)]
    
    # nursecharting extract hr 
    df_nurseCharting = df_nurseCharting[df_nurseCharting['nursingchartcelltypevallabel']=='Heart Rate']
    
    # nursecharting change index
    df_nurseCharting = df_nurseCharting.rename(columns={'nursingchartoffset': 'observationoffset', 'nursingchartvalue':'heartrate'})
    
    # extract heart rate from df_vitalPeriodic & df_nurseCharting
    HR_v = df_vitalPeriodic[['patientunitstayid', 'observationoffset', 'heartrate']]
    HR_n = df_nurseCharting[['patientunitstayid', 'observationoffset', 'heartrate']]
    HR = pd.concat([HR_v, HR_n]).astype(float)
    HR.sort_values(by=['patientunitstayid', 'observationoffset'], inplace=True)
    
    # exclude abnormal heart rate values
    HR.loc[:, 'heartrate'] = HR['heartrate'].apply(normal_heartrate)
    value_position_dict = {}
    first_occurrences = []
    for idx, value in enumerate(HR['patientunitstayid']):
    # if the value is not in the dictionary, add it and create index
        if value not in value_position_dict:
            value_position_dict[value] = idx
            first_occurrences.append(idx)
            
    first_occurrences.append(len(HR))
    # create first occurrence index for every patient
    HR_index = pd.Series(first_occurrences)
    
    end_time = time.time()
    print(f'Heart Rate Data Loaded. Time: {end_time - start_time:.2f}s')

    return HR, HR_index
    
    
def normal_heartrate(num):
    """
    Function to normalize heart rate values.

    Parameters:
        num: the originial input value
    Return:
        num: the normalized output value
    """
    # Return null values direcly
    if pd.isna(num):
        return num
    # Remove values out of range
    elif num > 300 or num < 0:
        return np.nan
    # Return normal values directly
    else:
        return num



