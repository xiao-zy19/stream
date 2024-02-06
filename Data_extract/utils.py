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
    patient_id,
    file_name1="vitalPeriodic.csv",
    file_name2="nurseCharting.csv",
    drop_neg=False,
):
    """
    Function to extract heart rate values.

    Args:
        patient_id: the list of wanted patient id
        file_name1: the file path of vitalPeriodic.csv
        file_name2: the file path of nurseCharting.csv
        drop_neg: whether to drop negative observationoffset
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
    cols_to_read_v = ['patientunitstayid', 'observationoffset', 'heartrate']
    df_vitalPeriodic = pd.read_csv(file_name1, usecols=cols_to_read_v)
    df_vitalPeriodic.sort_values(
        by=["patientunitstayid", "observationoffset"], inplace=True
    )
    
    # select wanted patient
    df_vitalPeriodic = df_vitalPeriodic[
        df_vitalPeriodic["patientunitstayid"].isin(patient_id)
    ]
    HR_v = df_vitalPeriodic[["patientunitstayid", "observationoffset", "heartrate"]]

    # memory deallocation
    del df_vitalPeriodic

    cols_to_read_n = ['patientunitstayid', 'nursingchartoffset', 'nursingchartcelltypevallabel', 'nursingchartvalue']
    df_nurseCharting = pd.read_csv(file_name2, usecols=cols_to_read_n)
    df_nurseCharting.sort_values(
        by=["patientunitstayid", "nursingchartoffset"], inplace=True
    )

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

    # extract heart rate
    HR_n = df_nurseCharting[["patientunitstayid", "observationoffset", "heartrate"]]
    HR = pd.concat([HR_v, HR_n]).astype(float)

    # memory deallocation
    del df_nurseCharting, HR_v, HR_n

    HR.sort_values(by=["patientunitstayid", "observationoffset"], inplace=True)

    # delete negative observationoffset
    if drop_neg:
        HR = HR.drop(HR[HR["observationoffset"] < 0].index)

    # exclude abnormal heart rate values
    HR.loc[:, "heartrate"] = HR["heartrate"].apply(normal_heartrate)

    # create index for HR
    HR_index = create_index(HR)

    end_time = time.time()
    print(f"Heart Rate Data Loaded. Time: {end_time - start_time:.2f}s")

    return HR, HR_index


def temp(patient_id, file_name1="vitalPeriodic.csv", file_name2="nurseCharting.csv"):
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
    cols_to_read_v = ['patientunitstayid', 'observationoffset', 'temperature']
    df_vitalPeriodic = pd.read_csv(file_name1, usecols=cols_to_read_v)
    df_vitalPeriodic.sort_values(
        by=["patientunitstayid", "observationoffset"], inplace=True
    )
    df_vitalPeriodic = df_vitalPeriodic[
        df_vitalPeriodic["patientunitstayid"].isin(patient_id)
    ]
    Temp_v = df_vitalPeriodic[["patientunitstayid", "observationoffset", "temperature"]]
    
    # memory deallocation
    del df_vitalPeriodic
    
    cols_to_read_n = ['patientunitstayid', 'nursingchartoffset', 'nursingchartcelltypevalname', 'nursingchartvalue']
    df_nurseCharting = pd.read_csv(file_name2, usecols=cols_to_read_n)
    df_nurseCharting.sort_values(
        by=["patientunitstayid", "nursingchartoffset"], inplace=True
    )   
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
    Temp_n = df_nurseCharting[["patientunitstayid", "observationoffset", "temperature"]]
    
    # memory deallocation
    del df_nurseCharting

    # delete the rows with string values
    Temp_n = Temp_n[
        Temp_n["temperature"].apply(lambda x: str(x).replace(".", "", 1).isdigit())
    ]

    Temp = pd.concat([Temp_v, Temp_n]).astype(float)
    
    # memory deallocation
    del Temp_v, Temp_n

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


# def blood_pressure(
#     patient_id,
#     file_name1="vitalPeriodic.csv",
#     file_name2="nurseCharting.csv",
#     file_name3="vitalAperiodic.csv",
# ):
#     """
#     Function to extract blood pressure values.

#     Args:
#         patient_id: the list of wanted patient id
#         file_name1: the file path of vitalPeriodic.csv
#         file_name2: the file path of nurseCharting.csv
#         file_name3: the file path of vitalAperiodic.csv
#     Returns:
#         systemicsystolic: the dataframe of systolic blood pressure data, including patientunitstayid, observationoffset, systemicsystolic

#         systemicsystolic_index: the series of the index of the first occurrence of each patient

#         non_invasive_BP_Systolic: the dataframe of non-invasive blood pressure data, including patientunitstayid, observationoffset, Non-Invasive BP Systolic

#         non_invasive_BP_Systolic_index: the series of the index of the first occurrence of each patient

#         invasive_BP_Systolic: the dataframe of invasive blood pressure data, including patientunitstayid, observationoffset, Invasive BP Systolic

#         invasive_BP_Systolic_index: the series of the index of the first occurrence of each patient

#         Noninvasivesystolic: the dataframe of non-invasive blood pressure data, including patientunitstayid, observationoffset, noninvasivesystolic

#         Noninvasivesystolic_index: the series of the index of the first occurrence of each patient
#     """
#     print("Loading Blood Pressure Data...")
#     start_time = time.time()
#     # Load data
#     df_vitalPeriodic = pd.read_csv(file_name1)
#     df_vitalPeriodic.sort_values(
#         by=["patientunitstayid", "observationoffset"], inplace=True
#     )

#     df_nurseCharting = pd.read_csv(file_name2)
#     df_nurseCharting.sort_values(
#         by=["patientunitstayid", "nursingchartoffset"], inplace=True
#     )

#     df_vitalAPeriodic = pd.read_csv(file_name3)
#     df_vitalAPeriodic.sort_values(
#         by=["patientunitstayid", "observationoffset"], inplace=True
#     )

#     # select wanted patient
#     df_vitalPeriodic = df_vitalPeriodic[
#         df_vitalPeriodic["patientunitstayid"].isin(patient_id)
#     ]
#     df_nurseCharting = df_nurseCharting[
#         df_nurseCharting["patientunitstayid"].isin(patient_id)
#     ]
#     df_vitalAPeriodic = df_vitalAPeriodic[
#         df_vitalAPeriodic["patientunitstayid"].isin(patient_id)
#     ]

#     # nursingchartcelltypevallabel Non-Invasive BP Systolic
#     df_nurseCharting_noninvasive = df_nurseCharting[
#         df_nurseCharting["nursingchartcelltypevalname"] == "Non-Invasive BP Systolic"
#     ]
#     df_nurseCharting_noninvasive = df_nurseCharting_noninvasive.rename(
#         columns={
#             "nursingchartoffset": "observationoffset",
#             "nursingchartvalue": "Non-Invasive BP Systolic",
#         }
#     )

#     # nursingchartcelltypevallabel Invasive BP Systolic
#     df_nurseCharting_invasive = df_nurseCharting[
#         df_nurseCharting["nursingchartcelltypevalname"] == "Invasive BP Systolic"
#     ]
#     df_nurseCharting_invasive = df_nurseCharting_invasive.rename(
#         columns={
#             "nursingchartoffset": "observationoffset",
#             "nursingchartvalue": "Invasive BP Systolic",
#         }
#     )

#     # extract systolics from vitalPeriodic, nurseCharting & vitalAPeriodic
#     systemicsystolic = df_vitalPeriodic[
#         ["patientunitstayid", "observationoffset", "systemicsystolic"]
#     ]
#     non_invasive_BP_Systolic = df_nurseCharting_noninvasive[
#         ["patientunitstayid", "observationoffset", "Non-Invasive BP Systolic"]
#     ]
#     invasive_BP_Systolic = df_nurseCharting_invasive[
#         ["patientunitstayid", "observationoffset", "Invasive BP Systolic"]
#     ]
#     Noninvasivesystolic = df_vitalAPeriodic[
#         ["patientunitstayid", "observationoffset", "noninvasivesystolic"]
#     ]
    
#     non_invasive_BP_Systolic.sort_values(
#         by=["patientunitstayid", "observationoffset"], inplace=True
#     )
    
#     invasive_BP_Systolic.sort_values(
#         by=["patientunitstayid", "observationoffset"], inplace=True
#     )
    
#     Noninvasivesystolic.sort_values(
#         by=["patientunitstayid", "observationoffset"], inplace=True
#     )
    
#     systemicsystolic["systemicsystolic"] = systemicsystolic["systemicsystolic"].astype('float64')
#     non_invasive_BP_Systolic["Non-Invasive BP Systolic"] = non_invasive_BP_Systolic["Non-Invasive BP Systolic"].astype('float64')
#     invasive_BP_Systolic["Invasive BP Systolic"] = invasive_BP_Systolic["Invasive BP Systolic"].astype('float64')
#     Noninvasivesystolic["noninvasivesystolic"] = Noninvasivesystolic["noninvasivesystolic"].astype('float64')
    
#     systemicsystolic_u = systemicsystolic.rename(
#         columns={
#             "systemicsystolic": "BP",
#         }
#     )
#     non_invasive_BP_Systolic_u = non_invasive_BP_Systolic.rename(
#         columns={
#             "Non-Invasive BP Systolic": "BP",
#         }
#     )
#     invasive_BP_Systolic_u = invasive_BP_Systolic.rename(
#         columns={
#             "Invasive BP Systolic": "BP",
#         }
#     )
#     Noninvasivesystolic_u = Noninvasivesystolic.rename(
#         columns={
#             "noninvasivesystolic": "BP",
#         }
#     )
    
#     merged_df = pd.merge(systemicsystolic_u, non_invasive_BP_Systolic_u,how='outer')
#     merged_df = pd.merge(merged_df, invasive_BP_Systolic_u,how='outer')
#     blood_pressure = pd.merge(merged_df, Noninvasivesystolic_u,how='outer')
#     blood_pressure.sort_values(
#         by=["patientunitstayid", "observationoffset"], inplace=True
#     )
#     # create index for each variable
#     systemicsystolic_index = create_index(systemicsystolic)
#     non_invasive_BP_Systolic_index = create_index(non_invasive_BP_Systolic)
#     invasive_BP_Systolic_index = create_index(invasive_BP_Systolic)
#     Noninvasivesystolic_index = create_index(Noninvasivesystolic)
#     blood_pressure_index = create_index(blood_pressure)
#     end_time = time.time()
#     print(f"Blood Pressure Data Loaded. Time: {end_time - start_time:.2f}s")

#     return (
#         systemicsystolic,
#         systemicsystolic_index,
#         non_invasive_BP_Systolic,
#         non_invasive_BP_Systolic_index,
#         invasive_BP_Systolic,
#         invasive_BP_Systolic_index,
#         Noninvasivesystolic,
#         Noninvasivesystolic_index,
#         blood_pressure,
#         blood_pressure_index
#     )
    
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
    cols_to_read_v1 = ['patientunitstayid', 'observationoffset', 'systemicsystolic']
    df_vitalPeriodic = pd.read_csv(file_name1, usecols=cols_to_read_v1)
    df_vitalPeriodic.sort_values(
        by=["patientunitstayid", "observationoffset"], inplace=True
    )
    df_vitalPeriodic = df_vitalPeriodic[
        df_vitalPeriodic["patientunitstayid"].isin(patient_id)
    ]
    systemicsystolic = df_vitalPeriodic[
        ["patientunitstayid", "observationoffset", "systemicsystolic"]
    ].copy()
    # memory deallocation
    del df_vitalPeriodic

    cols_to_read_n = ['patientunitstayid', 'nursingchartoffset', 'nursingchartcelltypevalname', 'nursingchartvalue']
    df_nurseCharting = pd.read_csv(file_name2, usecols=cols_to_read_n)
    df_nurseCharting.sort_values(
        by=["patientunitstayid", "nursingchartoffset"], inplace=True
    )
    df_nurseCharting = df_nurseCharting[
        df_nurseCharting["patientunitstayid"].isin(patient_id)
    ]
    df_nurseCharting_noninvasive = df_nurseCharting[
        df_nurseCharting["nursingchartcelltypevalname"] == "Non-Invasive BP Systolic"
    ].copy()
    df_nurseCharting_noninvasive = df_nurseCharting_noninvasive.rename(
        columns={
            "nursingchartoffset": "observationoffset",
            "nursingchartvalue": "Non-Invasive BP Systolic",
        }
    )
    df_nurseCharting_invasive = df_nurseCharting[
        df_nurseCharting["nursingchartcelltypevalname"] == "Invasive BP Systolic"
    ].copy()

    # memory deallocation
    del df_nurseCharting

    non_invasive_BP_Systolic = df_nurseCharting_noninvasive[
        ["patientunitstayid", "observationoffset", "Non-Invasive BP Systolic"]
    ].copy()

    df_nurseCharting_invasive = df_nurseCharting_invasive.rename(
        columns={
            "nursingchartoffset": "observationoffset",
            "nursingchartvalue": "Invasive BP Systolic",
        }
    )
    cols_to_read_v3 = ['patientunitstayid', 'observationoffset', 'noninvasivesystolic']
    df_vitalAPeriodic = pd.read_csv(file_name3, usecols=cols_to_read_v3)
    df_vitalAPeriodic.sort_values(
        by=["patientunitstayid", "observationoffset"], inplace=True
    )
    df_vitalAPeriodic = df_vitalAPeriodic[
        df_vitalAPeriodic["patientunitstayid"].isin(patient_id)
    ]
    Noninvasivesystolic = df_vitalAPeriodic[
        ["patientunitstayid", "observationoffset", "noninvasivesystolic"]
    ].copy()
    Noninvasivesystolic.sort_values(
        by=["patientunitstayid", "observationoffset"], inplace=True
    )
    Noninvasivesystolic["noninvasivesystolic"] = Noninvasivesystolic["noninvasivesystolic"].astype('float64')
    Noninvasivesystolic_u = Noninvasivesystolic.rename(
        columns={
            "noninvasivesystolic": "BP",
        }
    ).copy()

    # memory deallocation
    del df_vitalAPeriodic

    # extract systolics from vitalPeriodic, nurseCharting & vitalAPeriodic
    
    invasive_BP_Systolic = df_nurseCharting_invasive[
        ["patientunitstayid", "observationoffset", "Invasive BP Systolic"]
    ].copy()
    
    non_invasive_BP_Systolic.sort_values(
        by=["patientunitstayid", "observationoffset"], inplace=True
    )
    
    invasive_BP_Systolic.sort_values(
        by=["patientunitstayid", "observationoffset"], inplace=True
    )
    
    systemicsystolic["systemicsystolic"] = systemicsystolic["systemicsystolic"].astype('float64')
    non_invasive_BP_Systolic["Non-Invasive BP Systolic"] = non_invasive_BP_Systolic["Non-Invasive BP Systolic"].astype('float64')
    invasive_BP_Systolic["Invasive BP Systolic"] = invasive_BP_Systolic["Invasive BP Systolic"].astype('float64')
    
    
    systemicsystolic_u = systemicsystolic.rename(
        columns={
            "systemicsystolic": "BP",
        }
    ).copy()
    non_invasive_BP_Systolic_u = non_invasive_BP_Systolic.rename(
        columns={
            "Non-Invasive BP Systolic": "BP",
        }
    ).copy()
    invasive_BP_Systolic_u = invasive_BP_Systolic.rename(
        columns={
            "Invasive BP Systolic": "BP",
        }
    ).copy()
    
    merged_df = pd.merge(systemicsystolic_u, non_invasive_BP_Systolic_u,how='outer')
    merged_df = pd.merge(merged_df, invasive_BP_Systolic_u,how='outer')
    blood_pressure = pd.merge(merged_df, Noninvasivesystolic_u,how='outer')
    blood_pressure.sort_values(
        by=["patientunitstayid", "observationoffset"], inplace=True
    )
    # create index for each variable
    systemicsystolic_index = create_index(systemicsystolic)
    non_invasive_BP_Systolic_index = create_index(non_invasive_BP_Systolic)
    invasive_BP_Systolic_index = create_index(invasive_BP_Systolic)
    Noninvasivesystolic_index = create_index(Noninvasivesystolic)
    blood_pressure_index = create_index(blood_pressure)
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
        blood_pressure,
        blood_pressure_index
    )


def glasgow(patient_id, file_name1="nurseCharting.csv"):
    """
    Function to extract urine values.
    Parameters:
        patient_id: the list of wanted patient id
        file_name1: the file path of nurseCharting.csv
    Return:
        Glasgow: the dataframe of glasgow data, including patientunitstayid, observationoffset, glasgow score
        Glasgow_index: the series of the index of the first occurrence of each patient
    Usage:
        If we want the i th patient data (i starts from 0)
        use glasgow.iloc[glasgow_index[i]:glasgow_index[i+1]]
    """
    print("Loading Glasgow Data...")
    start_time = time.time()
    
    cols_to_read = ['patientunitstayid', 'nursingchartoffset', 'nursingchartcelltypevallabel', 'nursingchartvalue']
    df_nurseCharting = pd.read_csv(file_name1, usecols=cols_to_read)
    df_nurseCharting.sort_values(
        by=["patientunitstayid", "nursingchartoffset"], inplace=True
    )

    # select the wanted patient
    df_nurseCharting = df_nurseCharting[
        df_nurseCharting["patientunitstayid"].isin(patient_id)
    ]
    df_nurseCharting = df_nurseCharting[
        df_nurseCharting["nursingchartcelltypevallabel"] == "Glasgow coma score"
    ]
    df_nurseCharting = df_nurseCharting.rename(
        columns={
            "nursingchartoffset": "observationoffset",
            "nursingchartvalue": "Glasgow score",
        }
    )
    Glasgow = df_nurseCharting[
        ["patientunitstayid", "observationoffset", "Glasgow score"]
    ].copy()
    
    # memory deallocation
    del df_nurseCharting
    
    Glasgow.sort_values(by=["patientunitstayid", "observationoffset"], inplace=True)
    Glasgow["Glasgow score"] = pd.to_numeric(Glasgow["Glasgow score"], errors="coerce")
    Glasgow_index = create_index(Glasgow)
    end_time = time.time()
    print(f"Glasgow Data Loaded. Time: {end_time - start_time:.2f}s")
    return (
        Glasgow,
        Glasgow_index,
    )


def urine(patient_id, file_name1="intakeOutput.csv"):
    """
    Function to extract urine values.
    Parameters:
        patient_id: the list of wanted patient id
        file_name1: the file path of intakeOutput.csv
    Return:
        Urine: the dataframe of urine output data, including patientunitstayid, observationoffset, urine output
        Urine_index: the series of the index of the first occurrence of each patient
    Usage:
        If we want the i th patient data (i starts from 0)
        use urine.iloc[urine_index[i]:urine_index[i+1]]
    """
    print("Loading Urine Data...")
    start_time = time.time()
    
    cols_to_read = ['patientunitstayid', 'intakeoutputoffset', 'celllabel', 'cellvaluenumeric']
    df_intakeOutput = pd.read_csv(file_name1, usecols=cols_to_read)
    df_intakeOutput.sort_values(
        by=["patientunitstayid", "intakeoutputoffset"], inplace=True
    )
    df_intakeOutput = df_intakeOutput[
        df_intakeOutput["patientunitstayid"].isin(patient_id)
    ]
    
    # extract Urine data from intakeOutput.csv
    df_UrineOutput = df_intakeOutput[df_intakeOutput["celllabel"] == "Urine"]
    
    # memory deallocation
    del df_intakeOutput
    
    df_UrineOutput = df_UrineOutput.rename(columns={"cellvaluenumeric": "UrineOutput"})
    df_UrineOutput = df_UrineOutput.rename(
        columns={"intakeoutputoffset": "observationoffset"}
    )
    Urine = df_UrineOutput[
        ["patientunitstayid", "observationoffset", "UrineOutput"]
    ].copy()
    Urine_24 = df_UrineOutput[
        ["patientunitstayid", "observationoffset", "UrineOutput"]
    ].copy()
    Urine.sort_values(by=["patientunitstayid", "observationoffset"], inplace=True)
    
    # create first occurrence index for every patient
    Urine_index = create_index(Urine)
    end_time = time.time()
    print(f"Urine Data Loaded. Time: {end_time - start_time:.2f}s")
    return (
        Urine,
        Urine_index,
    )


def pao2fio2(
    patient_id,
    file_name1="nurseCharting.csv",
    file_name2="lab.csv",
    file_name3="respiratoryCharting.csv",
):
    """
    Function to extract fio2 and pao2 values.

    Parameters:
        patient_id: the list of wanted patient id
        file_name1: the file path of nurseCharting.csv
        file_name2: the file path of lab.csv
        file_name3: the file path of respiratoryCharting.csv
    Return:
        fio2: the dataframe of fio2 output data, including patientunitstayid, observationoffset, fio2, the unit is %
        fio2_index: the series of the index of the first occurrence of each patient
        pao2: the dataframe of fio2 output data, including patientunitstayid, observationoffset, pao2, the unit is mmHg
        pao2_index: the series of the index of the first occurrence of each patient
    Usage:
        If we want the i th patient data (i starts from 0)
        use urine.iloc[urine_index[i]:urine_index[i+1]]
    """
    print("Loading pao2/fio2 Data...")
    start_time = time.time()
    
    nurse_cols = ['patientunitstayid', 'nursingchartoffset', 'nursingchartcelltypevallabel', 'nursingchartvalue']
    lab_cols = ['patientunitstayid', 'labresultoffset', 'labname', 'labresult']
    resp_cols = ['patientunitstayid', 'respchartoffset', 'respchartvaluelabel', 'respchartvalue']
    
    df_nurseCharting = pd.read_csv(file_name1, usecols=nurse_cols)
    df_nurseCharting.sort_values(
        by=["patientunitstayid", "nursingchartoffset"], inplace=True
    )
    
    df_lab = pd.read_csv(file_name2, usecols=lab_cols)
    df_lab.sort_values(by=["patientunitstayid", "labresultoffset"], inplace=True)
    
    df_respiratoryCharting = pd.read_csv(file_name3, usecols=resp_cols)
    df_respiratoryCharting = pd.read_csv(file_name3)
    df_respiratoryCharting.sort_values(
        by=["patientunitstayid", "respchartoffset"], inplace=True
    )

    # select the wanted patient
    df_nurseCharting = df_nurseCharting[
        df_nurseCharting["patientunitstayid"].isin(patient_id)
    ]
    df_lab = df_lab[df_lab["patientunitstayid"].isin(patient_id)]
    df_respiratoryCharting = df_respiratoryCharting[
        df_respiratoryCharting["patientunitstayid"].isin(patient_id)
    ]

    df_nurseCharting_SVO = df_nurseCharting[
        df_nurseCharting["nursingchartcelltypevallabel"] == "SVO2"
    ]
    df_nurseCharting_SVO = df_nurseCharting_SVO.rename(
        columns={"nursingchartoffset": "observationoffset", "nursingchartvalue": "SVO2"}
    )

    # nursingchartcelltypevallabel: O2 Saturation
    df_nurseCharting_O2 = df_nurseCharting[
        df_nurseCharting["nursingchartcelltypevallabel"] == "O2 Saturation"
    ]
    df_nurseCharting_O2 = df_nurseCharting_O2.rename(
        columns={
            "nursingchartoffset": "observationoffset",
            "nursingchartvalue": "O2 Saturation",
        }
    )

    # labname: FiO2
    df_lab_FiO2 = df_lab[df_lab["labname"] == "FiO2"]
    df_lab_FiO2 = df_lab_FiO2.rename(
        columns={"labresultoffset": "observationoffset", "labresult": "FiO2"}
    )

    # labname: paO2
    df_lab_paO2 = df_lab[df_lab["labname"] == "paO2"]
    df_lab_paO2 = df_lab_paO2.rename(
        columns={"labresultoffset": "observationoffset", "labresult": "paO2"}
    )

    # respchartvaluelabel: FiO2
    df_respiratoryCharting_FiO2 = df_respiratoryCharting[
        df_respiratoryCharting["respchartvaluelabel"] == "FiO2"
    ]
    df_respiratoryCharting_FiO2 = df_respiratoryCharting_FiO2.rename(
        columns={"respchartoffset": "observationoffset", "respchartvalue": "FiO2"}
    )

    # respchartvaluelabel: FIO2 (%)
    df_respiratoryCharting_FIO2_percent = df_respiratoryCharting[
        df_respiratoryCharting["respchartvaluelabel"] == "FIO2 (%)"
    ]
    df_respiratoryCharting_FIO2_percent = df_respiratoryCharting_FIO2_percent.rename(
        columns={"respchartoffset": "observationoffset", "respchartvalue": "FiO2 (%)"}
    )

    nurse_SVO2 = (
        df_nurseCharting_SVO[["patientunitstayid", "observationoffset", "SVO2"]]
        .copy()
        .astype("float64")
    )
    nurse_O2 = (
        df_nurseCharting_O2[["patientunitstayid", "observationoffset", "O2 Saturation"]]
        .copy()
        .astype("float64")
    )
    nurse_SVO2.sort_values(by=["patientunitstayid", "observationoffset"], inplace=True)
    nurse_O2.sort_values(by=["patientunitstayid", "observationoffset"], inplace=True)

    # extract data from df_respiratoryCharting and sort by patientunitstayid and observationoffset
    lab_FiO2 = (
        df_lab_FiO2[["patientunitstayid", "observationoffset", "FiO2"]]
        .copy()
        .astype("float64")
    )
    lab_paO2 = (
        df_lab_paO2[["patientunitstayid", "observationoffset", "paO2"]]
        .copy()
        .astype("float64")
    )
    lab_FiO2.sort_values(by=["patientunitstayid", "observationoffset"], inplace=True)
    lab_paO2.sort_values(by=["patientunitstayid", "observationoffset"], inplace=True)

    # extract data from df_respiratoryCharting and sort by patientunitstayid and observationoffset
    resp_FiO2 = df_respiratoryCharting_FiO2[
        ["patientunitstayid", "observationoffset", "FiO2"]
    ].copy()
    resp_FiO2_percent = df_respiratoryCharting_FIO2_percent[
        ["patientunitstayid", "observationoffset", "FiO2 (%)"]
    ].copy()

    # 将'FiO2'列中的百分数转换为浮点数
    resp_FiO2["FiO2"] = resp_FiO2["FiO2"].apply(lambda x: percentage_to_float(x))
    resp_FiO2_percent["FiO2 (%)"] = resp_FiO2_percent["FiO2 (%)"].apply(
        lambda x: percentage_to_float(x)
    )
    resp_FiO2.sort_values(by=["patientunitstayid", "observationoffset"], inplace=True)
    resp_FiO2_percent.sort_values(
        by=["patientunitstayid", "observationoffset"], inplace=True
    )
    f_FIO2 = pd.merge(
        resp_FiO2,
        lab_FiO2,
        how="outer",
        on=["patientunitstayid", "observationoffset", "FiO2"],
    )
    f_FIO2.sort_values(by=["patientunitstayid", "observationoffset"], inplace=True)
    fio2_index = create_index(f_FIO2)
    pao2_index = create_index(lab_paO2)
    end_time = time.time()
    print(f"pao2/fio2 Data Loaded. Time: {end_time - start_time:.2f}s")
    return (f_FIO2, fio2_index, lab_paO2, pao2_index)

def lab_result(
    patient_id,
    file_name1="lab.csv"
):
    """
    Function to extract fio2 and pao2 values.

    Parameters:
        patient_id: the list of wanted patient id
        file_name1: the file path of lab.csv
    Return:
        BUN: the dataframe of Serum urea nitrogen level output data, including patientunitstayid, observationoffset, Serum urea nitrogen level, the unit is mg/dL
        BUN_index: the series of the index of the first occurrence of each patient
        WBC: the dataframe of White blood cells count output data, including patientunitstayid, observationoffset, White blood cells count, the unit is K/mcL
        WBC_index: the series of the index of the first occurrence of each patient
        bicarbonate: the dataframe of Serum bicarbonate level output data, including patientunitstayid, observationoffset, Serum bicarbonate level, the unit is mmol/L
        bicarbonate_index: the series of the index of the first occurrence of each patient
        sodium: the dataframe of Sodium level output data, including patientunitstayid, observationoffset, Sodium level, the unit is mmol/L
        sodium_index: the series of the index of the first occurrence of each patient
        potassium: the dataframe of Potassium level output data, including patientunitstayid, observationoffset, Potassium level, the unit is mmol/L
        potassium_index: the series of the index of the first occurrence of each patient
        total bilirubin: the dataframe of Bilirubin level output data, including patientunitstayid, observationoffset, Bilirubin level, the unit is mg/dL
        total bilirubin_index: the series of the index of the first occurrence of each patient
    Usage:
        If we want the i th patient data (i starts from 0)
        use lab_result.iloc[lab_result_index[i]:lab_result_index[i+1]]
    """
    print("Loading lab Data...")
    start_time = time.time()
    
    cols_to_read = ['patientunitstayid', 'labresultoffset', 'labname', 'labresult']
    df_lab = pd.read_csv(file_name1, usecols=cols_to_read)
    df_lab.sort_values(by=['patientunitstayid', 'labresultoffset'], inplace=True)

# select the wanted patient
    df_lab = df_lab[df_lab['patientunitstayid'].isin(patient_id)]
    df_lab_BUN = df_lab[df_lab['labname']=='BUN']
    df_lab_BUN = df_lab_BUN.rename(columns={'labresultoffset': 'observationoffset', 'labresult':'BUN'})

# lab WBC x 1000
    df_lab_WBC = df_lab[df_lab['labname']=='WBC x 1000']
    df_lab_WBC = df_lab_WBC.rename(columns={'labresultoffset': 'observationoffset', 'labresult':'WBC x 1000'})

# lab bicarbonate
    df_lab_bicarbonate = df_lab[df_lab['labname']=='bicarbonate']
    df_lab_bicarbonate = df_lab_bicarbonate.rename(columns={'labresultoffset': 'observationoffset', 'labresult':'bicarbonate'})

# lab sodium
    df_lab_sodium = df_lab[df_lab['labname']=='sodium']
    df_lab_sodium = df_lab_sodium.rename(columns={'labresultoffset': 'observationoffset', 'labresult':'sodium'})

# lab potassium
    df_lab_potassium = df_lab[df_lab['labname']=='potassium']
    df_lab_potassium = df_lab_potassium.rename(columns={'labresultoffset': 'observationoffset', 'labresult':'potassium'})

# lab total bilirubin
    df_lab_bilirubin = df_lab[df_lab['labname']=='total bilirubin']
    df_lab_bilirubin = df_lab_bilirubin.rename(columns={'labresultoffset': 'observationoffset', 'labresult':'total bilirubin'})
    
    BUN = df_lab_BUN[['patientunitstayid', 'observationoffset', 'BUN']].copy()
    WBC = df_lab_WBC[['patientunitstayid', 'observationoffset', 'WBC x 1000']].copy()
    bicarbonate = df_lab_bicarbonate[['patientunitstayid', 'observationoffset', 'bicarbonate']].copy()
    sodium = df_lab_sodium[['patientunitstayid', 'observationoffset', 'sodium']].copy()
    potassium = df_lab_potassium[['patientunitstayid', 'observationoffset', 'potassium']].copy()
    bilirubin = df_lab_bilirubin[['patientunitstayid', 'observationoffset', 'total bilirubin']].copy()

# sort the lab results by patient id and observationoffset
    BUN.sort_values(by=['patientunitstayid', 'observationoffset'], inplace=True)
    WBC.sort_values(by=['patientunitstayid', 'observationoffset'], inplace=True)
    bicarbonate.sort_values(by=['patientunitstayid', 'observationoffset'], inplace=True)
    sodium.sort_values(by=['patientunitstayid', 'observationoffset'], inplace=True)
    potassium.sort_values(by=['patientunitstayid', 'observationoffset'], inplace=True)
    bilirubin.sort_values(by=['patientunitstayid', 'observationoffset'], inplace=True)

# create index for each variable
    BUN_index = create_index(BUN)
    WBC_index = create_index(WBC)
    bicarbonate_index = create_index(bicarbonate)
    sodium_index = create_index(sodium)
    potassium_index = create_index(potassium)
    bilirubin_index = create_index(bilirubin)
    end_time = time.time()
    print(f"lab_result Data Loaded. Time: {end_time - start_time:.2f}s")
    return(
        BUN,
        BUN_index,
        WBC,
        WBC_index,
        bicarbonate,
        bicarbonate_index,
        sodium,
        sodium_index,
        potassium,
        potassium_index,
        bilirubin,
        bilirubin_index
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


def percentage_to_float(value):
    # Check if the value is a string and contains a percentage sign
    if isinstance(value, str) and "%" in value:
        try:
            # Remove the percentage sign and convert to float
            return float(value.rstrip("%"))
        except ValueError:
            # Return None if conversion fails
            return None
    # If the value is already a number (float or int), return it directly
    elif isinstance(value, (float, int)):
        return value
    # Return None for other types
    else:
        return None


def align_data(
    patient_batch,
    patient_offset,
    data,
    kernel="C(1.0) * RBF(10) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e5))",
    graph=False
):
    # TODO
    # add save func / save manually after align
    # output patient id with no known samples
    """
    Summary: align data and interpolate missing values

    Args:
        patient_batch: the list of wanted patient id, used to split data
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
    print(f"column names: {column_names}")

    # select the wanted patient
    data = data[data[column_names[0]].isin(patient_batch)]
    patient_offset = patient_offset[patient_offset[column_names[0]].isin(patient_batch)]

    # transform patient_offset to hours
    patient_hours = patient_offset.copy().reset_index(drop=True)
    patient_hours["unitdischargeoffset"] = np.floor(
        patient_hours["unitdischargeoffset"] / 60
    ).astype(int)

    # get unique patient ids in data
    unique_data_patient_ids = data["patientunitstayid"].unique()
    patient_hours = patient_hours[
        patient_hours["patientunitstayid"].isin(unique_data_patient_ids)
    ].reset_index(drop=True)

    # change observationoffset to hours
    data_hour_buf = data.copy().reset_index(drop=True)
    # data_hour_buf["observationoffset"] = np.floor(data_hour_buf["observationoffset"]/60).astype(int)
    data_hour_buf[column_names[1]] = np.floor(
        data_hour_buf[column_names[1]] / 60
    ).astype(int)
    data_hour_buf = data_hour_buf.groupby(
        [column_names[0], column_names[1]], as_index=False
    )[column_names[2]].mean()
    data_hour_buf.sort_values(by=[column_names[0], column_names[1]], inplace=True)

    data_hour_cleaned = pd.merge(
        data_hour_buf, patient_hours, on=column_names[0], how="left"
    )
    data_hour_cleaned = data_hour_cleaned[
        data_hour_cleaned[column_names[1]] <= data_hour_cleaned["unitdischargeoffset"]
    ]
    # data_hour = data_hour_cleaned.drop(["unitdischargeoffset"], axis=1)
    # print(data_hour)
    data_hour = data_hour_cleaned.copy()

    max_offset_per_patient = data_hour.groupby(column_names[0]).max().reset_index()

    complete_ranges = []
    for index, row in max_offset_per_patient.iterrows():
        patient_id = row[column_names[0]]
        # max_offset = row[column_names[1]]
        max_offset = row["unitdischargeoffset"]
        complete_range = pd.DataFrame(
            {column_names[0]: patient_id, column_names[1]: range(int(max_offset) + 1)}
        )
        complete_ranges.append(complete_range)

    complete_ranges = pd.concat(complete_ranges, ignore_index=True)
    data_full = pd.merge(
        complete_ranges, data_hour, on=[column_names[0], column_names[1]], how="left"
    )
    data_full.drop(["unitdischargeoffset"], axis=1, inplace=True)
    data_full_index = create_index(data_full)
    # print(data_full_index)
    
    

    for i in range(len(data_full_index) - 1):
        if (
            data_full.iloc[data_full_index[i] : data_full_index[i + 1]]
            .isnull()
            .values.any()
        ):  # test and i < 50
            data_data = data_full.iloc[data_full_index[i] : data_full_index[i + 1]][
                [column_names[1], column_names[2]]
            ].to_numpy()
            data_id = data_full.iloc[data_full_index[i] : data_full_index[i + 1]][
                column_names[0]
            ].unique()[0]

            t = data_data[:, 0].astype('float64')
            y = data_data[:, 1].astype('float64')
            t_known = t[~np.isnan(y)]
            # skip if there is no known data
            if t_known.size == 0:
                continue
            y_known = y[~np.isnan(y)]
            t_missing = t[np.isnan(y)]

            # kernel
            # kernel = C(1.0) * RBF(10) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e5))
            gp = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=10, normalize_y=True
            )
            gp.fit(t_known.reshape(-1, 1), y_known)  # fit
            y_pred, sigma = gp.predict(t_missing.reshape(-1, 1), return_std=True)

            inter_data = pd.DataFrame(
                {
                    column_names[0]: data_id,
                    column_names[1]: t_missing,
                    column_names[2]: y_pred,
                }
            )
            for idx, row in inter_data.iterrows():
                mask = (
                    (data_full[column_names[0]] == row[column_names[0]])
                    & (data_full[column_names[1]] == row[column_names[1]])
                    & data_full[column_names[2]].isnull()
                )
                data_full.loc[mask, column_names[2]] = row[column_names[2]]
            print(f"finished {i}th patient, patient_id: {data_id}")
            if graph:
                y_pred_all = gp.predict(t.reshape(-1, 1))
                plt.figure()
                plt.scatter(t_known, y_known, color="red", label="Known data")
                plt.scatter(t_missing, y_pred, color="blue", label="Interpolated data")
                plt.plot(t, y_pred_all)
                plt.fill_between(
                    t_missing, y_pred - sigma, y_pred + sigma, alpha=0.2, color="blue"
                )
                plt.title(f"Interpolation for Patient {data_id}")
                plt.xlabel("Time Offset")
                plt.ylabel(column_names[2])
                plt.legend()
                plt.show()

    print(f"Gaussian Process Finished!")

    return data_full, data_full_index
