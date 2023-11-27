import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def create_index(df):
    """
    Create first occurrence index for every patient
    """
    # create a dictionary to store the first occurrence index
    value_position_dict = {}
    first_occurrences = []
    for idx, value in enumerate(df["patientunitstayid"]):
        # if the value is not in the dictionary, add it and create index
        if value not in value_position_dict:
            value_position_dict[value] = idx
            first_occurrences.append(idx)

    first_occurrences.append(len(df))
    # create first occurrence index for every patient
    df_index = pd.Series(first_occurrences)
    return df_index


def patient_id_age(file_name1="diagnosis.csv", file_name2="patient.csv"):
    """
    Function to extract patient id and age.

    Args:
        file_name1: the file path of diagnosis.csv
        file_name2: the file path of patient.csv
    Return:
        patient_id: the list of wanted patient id
        df_patient_age: the dataframe of patient age data, including patientunitstayid, age
    """
    # import diagnosis.csv
    df_diagnosis = pd.read_csv(file_name1)
    df_diagnosis.sort_values(by=["patientunitstayid", "diagnosisoffset"], inplace=True)

    # select cardiovascular patients
    df_cardiovascular = df_diagnosis[
        df_diagnosis["diagnosisstring"].str.contains("cardiovascular")
    ]

    # get shock patient
    shock_patient = df_cardiovascular[
        df_cardiovascular["diagnosisstring"].str.contains("shock")
    ]

    # get ventricular patient
    ventricular_patient = df_cardiovascular[
        df_cardiovascular["diagnosisstring"].str.contains("ventricular")
    ]

    # get chest pain patient
    chest_pain_patient = df_cardiovascular[
        df_cardiovascular["diagnosisstring"].str.contains("chest pain")
    ]

    # get arrhythmias patient
    arrhythmias_patient = df_cardiovascular[
        df_cardiovascular["diagnosisstring"].str.contains("arrhythmias")
    ]

    # put id together
    df_wanted = pd.concat(
        [shock_patient, ventricular_patient, chest_pain_patient, arrhythmias_patient]
    )

    # Get the patient ids from df_wanted & sort the patient id
    # patient_id_all contains multiple entry patient's stayid
    patient_id_all = df_wanted["patientunitstayid"].unique()
    patient_id_all.sort()

    ## exclude patients with unitvisitnumber>1
    # import patient.csv
    df_patient = pd.read_csv(file_name2)
    df_patient.sort_values(by=["patientunitstayid"], inplace=True)
    df_patient_buf = df_patient[df_patient["patientunitstayid"].isin(patient_id_all)]
    df_1time_patient = df_patient_buf[df_patient_buf["unitvisitnumber"] == 1]

    # select the patient id from df_1time_patient
    patient_id = df_1time_patient["patientunitstayid"].unique()

    # extract patient age
    df_patient_age = df_1time_patient[["patientunitstayid", "age"]]
    df_patient_age = df_patient_age.replace(["> 89", ">89"], "90")
    df_patient_age["age"].fillna(0, inplace=True)
    df_patient_age = df_patient_age.astype({"age": "int32"})
    
    df_patient_offset = df_1time_patient[["patientunitstayid", "unitdischargeoffset"]]

    return patient_id, df_patient_age, df_patient_offset


def heart_rate(
    patient_id, file_name1="vitalPeriodic.csv", file_name2="nurseCharting.csv",
    drop_neg=False
):
    """
    Function to extract heart rate values.

    Args:
        patient_id: the list of wanted patient id
        file_name1: the file path of vitalPeriodic.csv
        file_name2: the file path of nurseCharting.csv
    Returns:
        HR: the dataframe of heart rate data, including patientunitstayid, observationoffset, heartrate
        HR_index: the series of the index of the first occurrence of each patient
    Usage:
        If we want the i th patient data (i starts from 0)
        use HR.iloc[HR_index[i]:HR_index[i+1]]
    """

    print("Loading Heart Rate Data...")
    start_time = time.time()
    # Load data
    df_vitalPeriodic = pd.read_csv(file_name1)
    df_vitalPeriodic.sort_values(
        by=["patientunitstayid", "observationoffset"], inplace=True
    )

    df_nurseCharting = pd.read_csv(file_name2)
    df_nurseCharting.sort_values(
        by=["patientunitstayid", "nursingchartoffset"], inplace=True
    )

    # select wanted patient
    df_vitalPeriodic = df_vitalPeriodic[
        df_vitalPeriodic["patientunitstayid"].isin(patient_id)
    ]
    df_nurseCharting = df_nurseCharting[
        df_nurseCharting["patientunitstayid"].isin(patient_id)
    ]

    # nursecharting extract hr
    df_nurseCharting = df_nurseCharting[
        df_nurseCharting["nursingchartcelltypevallabel"] == "Heart Rate"
    ]

    # nursecharting change index
    df_nurseCharting = df_nurseCharting.rename(
        columns={
            "nursingchartoffset": "observationoffset",
            "nursingchartvalue": "heartrate",
        }
    )

    # extract heart rate from df_vitalPeriodic & df_nurseCharting
    HR_v = df_vitalPeriodic[["patientunitstayid", "observationoffset", "heartrate"]]
    HR_n = df_nurseCharting[["patientunitstayid", "observationoffset", "heartrate"]]
    HR = pd.concat([HR_v, HR_n]).astype(float)
    HR.sort_values(by=["patientunitstayid", "observationoffset"], inplace=True)
    
    # delete negative observationoffset
    if drop_neg:
        HR = HR.drop(HR[HR['observationoffset'] < 0].index)

    # exclude abnormal heart rate values
    HR.loc[:, "heartrate"] = HR["heartrate"].apply(normal_heartrate)

    # create index for HR
    HR_index = create_index(HR)

    end_time = time.time()
    print(f"Heart Rate Data Loaded. Time: {end_time - start_time:.2f}s")

    return HR, HR_index


def temp(
    patient_id, file_name1="vitalPeriodic.csv", file_name2="nurseCharting.csv"
):
    """
    Function to extract temperature values.

    Parameters:
        patient_id: the list of wanted patient id
        file_name1: the file path of vitalPeriodic.csv
        file_name2: the file path of nurseCharting.csv
    Return:
        Temp: the dataframe of temperature data, including patientunitstayid, observationoffset, temperature
        Temp_index: the series of the index of the first occurrence of each patient
    Usage:
        If we want the i th patient data (i starts from 0)
        use Temp.iloc[Temp_index[i]:Temp_index[i+1]]
    """
    print("Loading Temperature Data...")
    start_time = time.time()

    # import vitalPeriodic.csv & nurseCharting.csv
    df_vitalPeriodic = pd.read_csv(file_name1)
    df_vitalPeriodic.sort_values(
        by=["patientunitstayid", "observationoffset"], inplace=True
    )
    df_nurseCharting = pd.read_csv(file_name2)
    df_nurseCharting.sort_values(
        by=["patientunitstayid", "nursingchartoffset"], inplace=True
    )

    # select the wanted patient
    df_vitalPeriodic = df_vitalPeriodic[
        df_vitalPeriodic["patientunitstayid"].isin(patient_id)
    ]
    df_nurseCharting = df_nurseCharting[
        df_nurseCharting["patientunitstayid"].isin(patient_id)
    ]

    # nursingchartcelltypevallabel Temperature
    df_nurseCharting = df_nurseCharting[
        df_nurseCharting["nursingchartcelltypevalname"] == "Temperature (C)"
    ]
    df_nurseCharting = df_nurseCharting.rename(
        columns={
            "nursingchartoffset": "observationoffset",
            "nursingchartvalue": "temperature",
        }
    )

    # extract temperature from df_vitalPeriodic & df_nurseCharting
    Temp_v = df_vitalPeriodic[["patientunitstayid", "observationoffset", "temperature"]]
    Temp_n = df_nurseCharting[["patientunitstayid", "observationoffset", "temperature"]]

    # delete the rows with string values
    Temp_n = Temp_n[
        Temp_n["temperature"].apply(lambda x: str(x).replace(".", "", 1).isdigit())
    ]

    Temp = pd.concat([Temp_v, Temp_n]).astype(float)
    
    # drop null values
    Temp.dropna(inplace=True)
    
    Temp.sort_values(by=["patientunitstayid", "observationoffset"], inplace=True)

    # exclude abnormal temp values and convert Fahrenheit to Celsius
    Temp.loc[:, "temperature"] = Temp["temperature"].apply(normal_temperature)

    # create index for Temp
    Temp_index = create_index(Temp)

    end_time = time.time()
    print(f"Temperature Data Loaded. Time: {end_time - start_time:.2f}s")

    return Temp, Temp_index


def blood_pressure(
    patient_id,
    file_name1="vitalPeriodic.csv",
    file_name2="nurseCharting.csv",
    file_name3="vitalAperiodic.csv",
):
    """
    Function to extract blood pressure values.

    Args:
        patient_id: the list of wanted patient id
        file_name1: the file path of vitalPeriodic.csv
        file_name2: the file path of nurseCharting.csv
        file_name3: the file path of vitalAperiodic.csv
    Returns:
        systemicsystolic: the dataframe of systolic blood pressure data, including patientunitstayid, observationoffset, systemicsystolic
        
        systemicsystolic_index: the series of the index of the first occurrence of each patient
        
        non_invasive_BP_Systolic: the dataframe of non-invasive blood pressure data, including patientunitstayid, observationoffset, Non-Invasive BP Systolic
        
        non_invasive_BP_Systolic_index: the series of the index of the first occurrence of each patient
        
        invasive_BP_Systolic: the dataframe of invasive blood pressure data, including patientunitstayid, observationoffset, Invasive BP Systolic
        
        invasive_BP_Systolic_index: the series of the index of the first occurrence of each patient
        
        Noninvasivesystolic: the dataframe of non-invasive blood pressure data, including patientunitstayid, observationoffset, noninvasivesystolic
        
        Noninvasivesystolic_index: the series of the index of the first occurrence of each patient
    """
    print("Loading Blood Pressure Data...")
    start_time = time.time()
    # Load data
    df_vitalPeriodic = pd.read_csv(file_name1)
    df_vitalPeriodic.sort_values(
        by=["patientunitstayid", "observationoffset"], inplace=True
    )

    df_nurseCharting = pd.read_csv(file_name2)
    df_nurseCharting.sort_values(
        by=["patientunitstayid", "nursingchartoffset"], inplace=True
    )

    df_vitalAPeriodic = pd.read_csv(file_name3)
    df_vitalAPeriodic.sort_values(
        by=["patientunitstayid", "observationoffset"], inplace=True
    )

    # select wanted patient
    df_vitalPeriodic = df_vitalPeriodic[
        df_vitalPeriodic["patientunitstayid"].isin(patient_id)
    ]
    df_nurseCharting = df_nurseCharting[
        df_nurseCharting["patientunitstayid"].isin(patient_id)
    ]
    df_vitalAPeriodic = df_vitalAPeriodic[
        df_vitalAPeriodic["patientunitstayid"].isin(patient_id)
    ]

    # nursingchartcelltypevallabel Non-Invasive BP Systolic
    df_nurseCharting_noninvasive = df_nurseCharting[
        df_nurseCharting["nursingchartcelltypevalname"] == "Non-Invasive BP Systolic"
    ]
    df_nurseCharting_noninvasive = df_nurseCharting_noninvasive.rename(
        columns={
            "nursingchartoffset": "observationoffset",
            "nursingchartvalue": "Non-Invasive BP Systolic",
        }
    )

    # nursingchartcelltypevallabel Invasive BP Systolic
    df_nurseCharting_invasive = df_nurseCharting[
        df_nurseCharting["nursingchartcelltypevalname"] == "Invasive BP Systolic"
    ]
    df_nurseCharting_invasive = df_nurseCharting_invasive.rename(
        columns={
            "nursingchartoffset": "observationoffset",
            "nursingchartvalue": "Invasive BP Systolic",
        }
    )

    # extract systolics from vitalPeriodic, nurseCharting & vitalAPeriodic
    systemicsystolic = df_vitalPeriodic[
        ["patientunitstayid", "observationoffset", "systemicsystolic"]
    ]
    non_invasive_BP_Systolic = df_nurseCharting_noninvasive[
        ["patientunitstayid", "observationoffset", "Non-Invasive BP Systolic"]
    ]
    invasive_BP_Systolic = df_nurseCharting_invasive[
        ["patientunitstayid", "observationoffset", "Invasive BP Systolic"]
    ]
    Noninvasivesystolic = df_vitalAPeriodic[
        ["patientunitstayid", "observationoffset", "noninvasivesystolic"]
    ]

    # create index for each variable
    systemicsystolic_index = create_index(systemicsystolic)
    non_invasive_BP_Systolic_index = create_index(non_invasive_BP_Systolic)
    invasive_BP_Systolic_index = create_index(invasive_BP_Systolic)
    Noninvasivesystolic_index = create_index(Noninvasivesystolic)

    end_time = time.time()
    print(f"Blood Pressure Data Loaded. Time: {end_time - start_time:.2f}s")

    return (
        systemicsystolic,
        systemicsystolic_index,
        non_invasive_BP_Systolic,
        non_invasive_BP_Systolic_index,
        invasive_BP_Systolic,
        invasive_BP_Systolic_index,
        Noninvasivesystolic,
        Noninvasivesystolic_index,
    )


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


# define normal temp function
def normal_temperature(num):
    """
    Function to normalize temperature values.

    Parameters:
        num: the originial input value
    Return:
        num: the normalized output value
    """
    # Return null values direcly
    if pd.isna(num):
        return num
    # Convert Fahrenheit to Celsius
    # And apply the function again
    elif num > 50:
        return normal_temperature((num - 32) * 5 / 9)
    # Remove values out of range
    elif num < 15 or num > 45:
        return np.nan
    # Return normal values directly
    else:
        return num


def align_data(patient_id, patient_offset, data, kernel='C(1.0) * RBF(10) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e5))'):
    
    """
    Summary: align data and interpolate missing values
    
    Args:
        patient_id: the list of wanted patient id
        patient_offset: the dataframe of patient offset data, including patientunitstayid, unitdischargeoffset
        data: the dataframe of data, including patientunitstayid, observationoffset, value
        kernel: the self-defined kernel function for Gaussian Process Regressor

    Returns:
        data_full: the dataframe of aligned and interpolated data, including patientunitstayid, observationoffset, value
        data_full_index: the series of the index of the first occurrence of each patient
    """
    
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
    
    # turn kernel string to kernel function
    kernel = eval(kernel)
    
    column_names = data.columns.tolist()
    print(column_names)
    
    # transform patient_offset to hours
    patient_hours = patient_offset.copy().reset_index(drop=True)
    patient_hours['unitdischargeoffset'] = np.floor(patient_hours['unitdischargeoffset']/60).astype(int)
    
    # get unique patient ids in data
    unique_data_patient_ids = data['patientunitstayid'].unique()
    patient_hours = patient_hours[patient_hours['patientunitstayid'].isin(unique_data_patient_ids)].reset_index(drop=True)
    
    # 
    data_hour_buf = data.copy().reset_index(drop=True)
    data_hour_buf["observationoffset"] = np.floor(data_hour_buf["observationoffset"]/60).astype(int)
    data_hour_buf = data_hour_buf.groupby([column_names[0], column_names[1]], as_index=False)[column_names[2]].mean()
    data_hour_buf.sort_values(by=[column_names[0], column_names[1]], inplace=True)
    
    data_hour_cleaned = pd.merge(data_hour_buf, patient_hours, on=column_names[0], how='left')
    data_hour_cleaned = data_hour_cleaned[data_hour_cleaned[column_names[1]] <= data_hour_cleaned['unitdischargeoffset']]
    data_hour = data_hour_cleaned.drop(['unitdischargeoffset'], axis=1)
    # print(data_hour)
    
    max_offset_per_patient = data_hour.groupby(column_names[0]).max().reset_index()
    
    complete_ranges = []
    for index, row in max_offset_per_patient.iterrows():
        patient_id = row[column_names[0]]
        max_offset = row[column_names[1]]
        complete_range = pd.DataFrame({
            column_names[0]: patient_id,
            column_names[1]: range(int(max_offset)+1)
        })
        complete_ranges.append(complete_range)
    
    complete_ranges = pd.concat(complete_ranges, ignore_index=True)
    data_full = pd.merge(complete_ranges, data_hour, on=[column_names[0], column_names[1]], how='left')
    data_full_index = create_index(data_full)
    # print(data_full_index)
    
    for i in range(len(data_full_index)-1):
        if data_full.iloc[data_full_index[i]:data_full_index[i+1]].isnull().values.any() and i<50: # test
            data_data = data_full.iloc[data_full_index[i]:data_full_index[i+1]][[column_names[1], column_names[2]]].to_numpy()
            data_id = data_full.iloc[data_full_index[i]:data_full_index[i+1]][column_names[0]].unique()[0]
            
            t = data_data[:, 0]
            y = data_data[:, 1]
            t_known = t[~np.isnan(y)]
            y_known = y[~np.isnan(y)]
            t_missing = t[np.isnan(y)]
            
            # kernel
            # kernel = C(1.0) * RBF(10) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e5))
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1000, normalize_y=True)
            gp.fit(t_known.reshape(-1, 1), y_known) #fit
            y_pred, sigma = gp.predict(t_missing.reshape(-1, 1), return_std=True)    
            
            inter_data = pd.DataFrame({
                column_names[0]: data_id,
                column_names[1]: t_missing,
                column_names[2]: y_pred
            })
            for idx, row in inter_data.iterrows():
                mask = (data_full[column_names[0]] == row[column_names[0]]) & (data_full[column_names[1]] == row[column_names[1]]) & data_full[column_names[2]].isnull()
                data_full.loc[mask, column_names[2]] = row[column_names[2]]
            print(f'finished {i}th patient, patient_id: {data_id}')
            
    return data_full, data_full_index