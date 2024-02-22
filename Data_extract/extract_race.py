import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def convert_to_float(df):
    """
    Convert the last column of dataframe to float64

    Args:
        df: the dataframe to be converted

    Returns:
        df: the converted dataframe
    """
    df.iloc[:, -1] = pd.to_numeric(df.iloc[:, -1], errors="coerce").astype("float64")
    return df


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


def patient_id_race(file_name1="diagnosis.csv", file_name2="patient.csv"):
    """
    Function to extract patient id and age.

    Args:
        file_name1: the file path of diagnosis.csv
        file_name2: the file path of patient.csv
    Return:
        patient_id: the list of wanted patient id
        df_patient_race: the dataframe of patient race, including patientunitstayid, race
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

    # exclude patients with unitvisitnumber>1
    # import patient.csv
    df_patient = pd.read_csv(file_name2)
    df_patient.sort_values(by=["patientunitstayid"], inplace=True)
    df_patient_buf = df_patient[df_patient["patientunitstayid"].isin(patient_id_all)]
    df_1time_patient = df_patient_buf[df_patient_buf["unitvisitnumber"] == 1]

    # select the patient id from df_1time_patient
    patient_id = df_1time_patient["patientunitstayid"].unique()

    # extract patient race
    df_patient_race = df_1time_patient[["patientunitstayid", "ethnicity"]]

    return patient_id, df_patient_race


if __name__ == '__main__':
    patient_id, df_patient_race = patient_id_race(file_name1="diagnosis.csv", file_name2="patient.csv")
    #print(patient_id, df_patient_race)

    df_patient_race.to_csv('patient_race_data.csv', index=False)
