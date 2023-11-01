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
    for idx, value in enumerate(df['patientunitstayid']):
        # if the value is not in the dictionary, add it and create index
        if value not in value_position_dict:
            value_position_dict[value] = idx
            first_occurrences.append(idx)

    first_occurrences.append(len(df))
    # create first occurrence index for every patient
    df_index = pd.Series(first_occurrences)
    return df_index


def patient_id_age(file_name1="data/diagnosis.csv", file_name2="data/patient.csv"):
    """
    Function to extract patient id and age.

    Parameters:
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

    return patient_id, df_patient_age


def heart_rate(
    patient_id, file_name1="data/vitalPeriodic.csv", file_name2="data/nurseCharting.csv"
):
    """
    Function to extract heart rate values.

    Parameters:
        patient_id: the list of wanted patient id
        file_name1: the file path of vitalPeriodic.csv
        file_name2: the file path of nurseCharting.csv
    Return:
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

    # exclude abnormal heart rate values
    HR.loc[:, "heartrate"] = HR["heartrate"].apply(normal_heartrate)
    
    # create index for HR
    HR_index = create_index(HR)

    end_time = time.time()
    print(f"Heart Rate Data Loaded. Time: {end_time - start_time:.2f}s")

    return HR, HR_index


def temp(
    patient_id, file_name1="data/vitalPeriodic.csv", file_name2="data/nurseCharting.csv"
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
    Temp.sort_values(by=["patientunitstayid", "observationoffset"], inplace=True)

    # exclude abnormal temp values and convert Fahrenheit to Celsius
    Temp.loc[:, "temperature"] = Temp["temperature"].apply(normal_temperature)

    # create index for Temp
    Temp_index = create_index(Temp)
    
    end_time = time.time()
    print(f"Temperature Data Loaded. Time: {end_time - start_time:.2f}s")

    return Temp, Temp_index


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
