import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import patient_id_age, heart_rate, create_index

# change to your folder path
# os.chdir('C:/Users/xiao-zy19/Desktop/Johns Hopkins/Biomedical Data Design/EICU Database/eicu-collaborative-research-database-demo-2.0.1')
# os.chdir('/Users/xiao-zy19/Desktop/Johns Hopkins/Biomedical Data Design/EICU Database/eicu-collaborative-research-database-demo-2.0.1') 
os.chdir('/home/en580-zxia028/EICU_demo/data')

# get patient id
patient_id, patient_age, patient_offset = patient_id_age()
# extract heart rate data
HR, _ = heart_rate(patient_id, drop_neg=True)
print(f'Heartrate data shape: {HR.shape}')
# print(f'HR_index shape: {HR_index.shape}')
print(patient_offset)

patient_hours = patient_offset.copy().reset_index(drop=True)
patient_hours['unitdischargeoffset'] = np.floor(patient_hours['unitdischargeoffset']/60).astype(int)
print(patient_hours)

unique_HR_patient_ids = HR['patientunitstayid'].unique()
patient_hours = patient_hours[patient_hours['patientunitstayid'].isin(unique_HR_patient_ids)].reset_index(drop=True)
print(patient_hours)

HR_hour_buf = HR.copy().reset_index(drop=True)
HR_hour_buf["observationoffset"] = np.floor(HR_hour_buf["observationoffset"]/60).astype(int)
HR_hour_buf = HR_hour_buf.groupby(["patientunitstayid", "observationoffset"], as_index=False)["heartrate"].mean()
HR_hour_buf.sort_values(by=["patientunitstayid", "observationoffset"], inplace=True)

HR_hour_cleaned = pd.merge(HR_hour_buf, patient_hours,on='patientunitstayid', how='left')
HR_hour_cleaned = HR_hour_cleaned[HR_hour_cleaned['observationoffset'] <= HR_hour_cleaned['unitdischargeoffset']]
HR_hour = HR_hour_cleaned.drop(['unitdischargeoffset'], axis=1)

max_offset_per_patient = HR_hour.groupby('patientunitstayid')['observationoffset'].max().reset_index()
# print(max_offset_per_patient)

complete_ranges = []
for index, row in max_offset_per_patient.iterrows():
    patient_id = row['patientunitstayid']
    max_offset = row['observationoffset']
    complete_range = pd.DataFrame({
        'patientunitstayid': patient_id,
        'observationoffset': range(int(max_offset) + 1)
    })
    complete_ranges.append(complete_range)

complete_df = pd.concat(complete_ranges, ignore_index=True)
HR_full = pd.merge(complete_df, HR_hour, on=['patientunitstayid', 'observationoffset'], how='left')

HR_full_index = create_index(HR_full)
# print(HR_full_index)

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

for i in range(len(HR_full_index)-1):
    if HR_full.iloc[HR_full_index[i]:HR_full_index[i+1]].isnull().values.any() and i<50: # test:  i < 10
        HR_data = HR_full.iloc[HR_full_index[i]:HR_full_index[i+1]][['observationoffset', 'heartrate']].to_numpy()
        HR_id = HR_full.iloc[HR_full_index[i]:HR_full_index[i+1]]['patientunitstayid'].unique()[0]
        # print(HR_id)
        # print(i)
        t = HR_data[:, 0]
        y = HR_data[:, 1]
        
        t_known = t[~np.isnan(y)]
        y_known = y[~np.isnan(y)]
        # print(t_known, y_known)
        
        # kernel
        kernel = C(1.0) * RBF(10) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e5))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1000, normalize_y=True)
        gp.fit(t_known.reshape(-1, 1), y_known)
        t_missing = t[np.isnan(y)]
        # print(t_missing)

        y_pred, sigma = gp.predict(t_missing.reshape(-1, 1), return_std=True)
        inter_data = pd.DataFrame({'patientunitstayid': HR_id, 'observationoffset': t_missing, 'heartrate': y_pred})
        
        for idx, row in inter_data.iterrows():
            mask = (HR_full['patientunitstayid'] == row['patientunitstayid']) & (HR_full['observationoffset'] == row['observationoffset']) & HR_full['heartrate'].isnull()
            HR_full.loc[mask, 'heartrate'] = row['heartrate']
        y_pred_all = gp.predict(t.reshape(-1, 1))
        print(f'Finish {i}th patient')
        
        save_path = f'.~/output/hr/demo/{HR_id}.png'
        plt.scatter(t_known, y_known, color='red', label='Known data')
        plt.scatter(t_missing, y_pred, color='blue', label='Interpolated data')
        plt.plot(t, y_pred_all)
    
        plt.fill_between(t_missing, y_pred - sigma, y_pred + sigma, alpha=0.2, color='blue')

        plt.title(f'Heart Rate Interpolation for Patient {HR_id}')
        plt.xlabel('Time Offset')
        plt.ylabel('Heart Rate')
        plt.legend()
        plt.show()
        plt.savefig(save_path)
        # print(HR_full.iloc[HR_full_index[i]:HR_full_index[i+1]])

print(f'GaussianProcess finished!')
print(HR_full)
