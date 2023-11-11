"""
Prototype functions that are useful for data exploration
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, roc_curve, precision_recall_curve, brier_score_loss


def find_patient(fname, patientunitstayid, cs=100000, verbose=True):
    """
    Retrieve single patient info. Also sorts by offsets

    Parameters:
        patientunitstayid: the patient whose data to pull
        cs = the size of the chunk to iterate through
        verbose = information on how many iterations through
    Returns:
        df: a dataframe (sorted by nursingchartoffset) with data from that patient
    """

    iteration = 0
    for chunk in pd.read_csv(fname, chunksize=cs):
        # for keeping track of where we are
        if verbose:
            if iteration % 100 == 0:
                print("iter {0}".format(iteration))
            iteration += 1

        # pull patient information out
        df = chunk.loc[chunk["patientunitstayid"] == patientunitstayid]

        # when we find our dude
        if df.empty == False:
            print("Found patient")
            break

    if df.empty == True:
        print("Error: no patient was found by that id")
        return None

    # sort for convenience
    df = df.sort_values(by=["nursingchartoffset"])

    return df


def plot_vitals(vitals, patient):
    """
    Given a list of vitals and patient, form a nice timeseries plot

    Parameters:
        vitals: list of list in the form:
                 [nursingchartcelltypevallabel, nursingchartcelltypevalname]
        patient: dataframe from a single patient's data
    """

    plt.figure(figsize=[12, 8])
    for v in vitals:
        idx = (patient["nursingchartcelltypevallabel"] == v[0]) & (
            patient["nursingchartcelltypevalname"] == v[1]
        )
        df_plot = patient.loc[idx, :]

        if "Systolic" in v[1]:
            marker = "^-"
        elif "Diastolic" in v[1]:
            marker = "v-"
        else:
            marker = "o-"
        plt.plot(
            df_plot["nursingchartoffset"],
            pd.to_numeric(df_plot["nursingchartvalue"], errors="coerce"),
            marker,
            markersize=8,
            lw=2,
            label=v,
        )

    plt.xlabel("Time since ICU admission (minutes)")
    plt.ylabel("Measurement value")
    plt.legend(loc="upper right")
    plt.show()


def multi_patient_feature_plot(df, vitals):
    """
    For a dataframe of a given size, plot the time series feature of all patients

    Parameters:
        df = the pd dataframe to extract info from (don't use a big one!)
        vitals = list of list in form [[nursingchartcelltypevallabel, nursingchartcelltypevalname]]
    """

    # sort df if not already done and find unique patients
    df = df.sort_values(by=["nursingchartoffset"])
    unique_patients = df.patientunitstayid.unique()

    plt.figure()
    # check out all unique patientunitstayid's
    for patient_id in unique_patients:
        patient = df.loc[df["patientunitstayid"] == patient_id]
        idx = (patient["nursingchartcelltypevallabel"] == vitals[0][0]) & (
            patient["nursingchartcelltypevalname"] == vitals[0][1]
        )
        df_plot = patient.loc[idx, :]

        # style choice
        marker = "o-"

        plt.plot(
            df_plot["nursingchartoffset"],
            pd.to_numeric(df_plot["nursingchartvalue"], errors="coerce"),
            marker,
            markersize=8,
            lw=2,
            label=vitals,
        )

    plt.xlabel("Time Since ICU Admission (minutes)")
    plt.ylabel("GCS Value")
    plt.show()


def get_next_patient(fname, idx_start=0, num_rows=1000000):
    """
    Function to read one patient at a time. This function is intended to be chained together
    with repeated calls.
    Note: this isn't the fastest way to do this for sure, but it works

    Parameters:
        fname: the filename to draw data from
        idx_start: the row to skip to during data collection
        num_rows: the amount of rows to read at a time
    Returns:
        patient: the dataframe belonging solely to the patient of interest
                 the dataframe is returned SORTED by nursingchartoffset
        next_idx: the row to skip to (aka where the next patient begins)
    """

    # extract headers
    temp = pd.read_csv(fname, nrows=1)
    header = temp.columns

    # read in a chunk of the dataframe starting at a location
    df = pd.read_csv(fname, skiprows=idx_start, nrows=num_rows)
    df.columns = header

    # get patientunitstayid - will be the patient of interest
    patient_id = df["patientunitstayid"].iloc[0]

    # extract all data of patient with this id
    # and sort according to offsets
    patient = df.loc[df["patientunitstayid"] == patient_id]
    patient = patient.sort_values(by=["nursingchartoffset"])

    # keep track of next patient index (i.e. how many rows to skip)
    length, __ = patient.shape
    next_idx = length + idx_start

    return patient, next_idx


# Caution! Potential problems with corrupted index!
def shear_next_patient(df, next_idx):
    """
    Function to generate a dataframe for next patient and trim the original.

    Parameters:
        df: the original dataframe that contains all data
        next_idx: the index to check
    Returns:
        df: the trimmed dataframe without the selected patient's data
        p_df: the patient dataframe that was sheared from the original
        p_id: the patient id
        next_idx: the index to skip to (aka where the next patient begins)
    """
    # ASSUMES THE DF HAS BEEN SORTED BY PATIENTUNITSTAYID ALREADY

    # Trim the dataframe
    df = df.loc[next_idx:]

    # Select the patient of interest
    p_id = df["patientunitstayid"].loc[next_idx]
    p_df = df.loc[df["patientunitstayid"] == p_id]

    # Sort by the offset
    p_df = p_df.sort_values(by=["nursingchartoffset"])

    # Determine the index to skip to
    length, __ = p_df.shape
    next_idx = length + next_idx

    return df, p_df, p_id, next_idx


def lift_next_patient(df, next_idx):
    """
    Function to generate a dataframe for next patient.

    Parameters:
        df: the original dataframe that contains all data
        next_idx: the index to check
    Returns:
        p_df: the patient dataframe that was lifted
        p_id: the patient id
        next_idx: the index to skip to (aka where the next patient begins)
    """
    # ASSUMES THE DF HAS BEEN SORTED BY PATIENTUNITSTAYID ALREADY

    # index of patient
    p_id = df["patientunitstayid"].iloc[next_idx]

    # extract all data of patient with this id and sort according to offsets
    p_df = df.loc[df["patientunitstayid"] == p_id]
    p_df = p_df.sort_values(by=["nursingchartoffset"])

    # keep track of next patient index (i.e. how many rows to skip)
    length, __ = p_df.shape
    next_idx = length + next_idx

    return p_df, p_id, next_idx


def filter_patients(df, patients):
    """
    Function to remove all rows that is not one of targeted patients

    Parameters:
        df: the original dataframe that contains all data
        patients: the patient list interested
    Return:
        filtered df of target patients
    """

    return df[df["patientunitstayid"].isin(patients["patientunitstayid"].to_list())]


def apply_exclusion_criteria(df, diagnoses, criteria):
    """
    Function to apply exclusion criteria.

    Parameters:
        df: the original dataframe that contains all data
        diagnoses: the diagnosis dataframe
        criteria: the criteria list
    Return:
        excluded_df: filtered df that excluded diagnosis criteria
    """

    # Apply a mask to find ids of patients to exclude
    mask = diagnoses.diagnosisstring.apply(
        lambda x: any(item for item in criteria if item in x)
    )
    temp_df = diagnoses[mask]

    # recover just the ids
    exclusion_ids = temp_df["patientunitstayid"]

    # apply those ids
    excluded_df = df[~df["patientunitstayid"].isin(exclusion_ids)]
    return excluded_df


def normal_temperature(num):
    """
    Function to normalize temperature values.

    Parameters:
        num: the originial input value
    Return:
        num: the normalized output value
    """
    # Return null values direcly
    if num == np.nan:
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


def normal_sao2(num):
    """
    Function to normalize O2 saturation values.

    Parameters:
        num: the originial input value
    Return:
        num: the normalized output value
    """
    # Return null values direcly
    if num == np.nan:
        return num
    # Remove values out of range
    elif num < 50 or num > 100:
        return np.nan
    # Return normal values directly
    else:
        return num


def normal_heartrate(num):
    """
    Function to normalize heart rate values.

    Parameters:
        num: the originial input value
    Return:
        num: the normalized output value
    """
    # Return null values direcly
    if num == np.nan:
        return num
    # Remove values out of range
    elif num > 300 or num < 0:
        return np.nan
    # Return normal values directly
    else:
        return num


def normal_respiration(num):
    """
    Function to normalize respiratory rate values.

    Parameters:
        num: the originial input value
    Return:
        num: the normalized output value
    """
    # Return null values direcly
    if num == np.nan:
        return num
    # Remove values out of range
    elif num > 100 or num < 0:
        return np.nan
    # Return normal values directly
    else:
        return num


def normal_cvp(num):
    """
    Function to normalize central venous pressure values.

    Parameters:
        num: the originial input value
    Return:
        num: the normalized output value
    """
    # Return null values direcly
    if num == np.nan:
        return num
    # Remove values out of range
    elif num < -10 or num > 50:
        return np.nan
    # Return normal values directly
    else:
        return num


def normal_etco2(num):
    """
    Function to normalize end tidal CO2 values.

    Parameters:
        num: the originial input value
    Return:
        num: the normalized output value
    """
    # Return null values direcly
    if num == np.nan:
        return num
    # Remove values out of range
    elif num < 0 or num > 100:
        return np.nan
    # Return normal values directly
    else:
        return num


def normal_systemic(systolic, diastolic, mean_p):
    """
    Function to normalize systemic blood pressure values.

    Parameters:
        num: the originial input value
    Return:
        num: the normalized output value
    """
    # Return null values direcly
    if systolic == np.nan or diastolic == np.nan or mean_p == np.nan:
        return np.nan, np.nan, np.nan
    # Remove values out of range
    elif systolic < 0 or systolic > 300:
        return np.nan, np.nan, np.nan
    elif diastolic < 0 or diastolic > 200:
        return np.nan, np.nan, np.nan
    elif mean_p < 0 or mean_p > 190:
        return np.nan, np.nan, np.nan
    elif diastolic >= mean_p:
        return np.nan, np.nan, np.nan
    elif systolic < mean_p:
        return np.nan, np.nan, np.nan
    elif systolic - diastolic <= 4:
        return np.nan, np.nan, np.nan
    # Return normal values directly
    else:
        return systolic, diastolic, mean_p


def normal_pa(systolic, diastolic, mean_p):
    """
    Function to normalize pulmonary artery blood pressure values.

    Parameters:
        num: the originial input value
    Return:
        num: the normalized output value
    """
    # Return null values direcly
    if systolic == np.nan or diastolic == np.nan or mean_p == np.nan:
        return np.nan, np.nan, np.nan
    # Remove values out of range
    elif systolic < 0 or systolic > 300:
        return np.nan, np.nan, np.nan
    elif diastolic < 0 or diastolic > 200:
        return np.nan, np.nan, np.nan
    elif mean_p < 0 or mean_p > 190:
        return np.nan, np.nan, np.nan
    elif diastolic >= mean_p:
        return np.nan, np.nan, np.nan
    elif systolic < mean_p:
        return np.nan, np.nan, np.nan
    elif systolic - diastolic <= 4:
        return np.nan, np.nan, np.nan
    # Return normal values directly
    else:
        return systolic, diastolic, mean_p


def normal_lab(labname, num):
    """
    Function to normalize lab values.

    Parameters:
        labname: the label name of lab test
        num: the originial input value
    Return:
        num: the normalized output value
    """
    labmin = {
        "BUN": 0,
        "Hct": 0,
        "Hgb": 1,
        "MCH": 10,
        "MCHC": 15,
        "MCV": 40,
        "MPV": 3,
        "RBC": 2,
        "RDW": 5,
        "WBC x 1000": 0,
        "anion gap": 0,
        "bicarbonate": 1,
        "calcium": 3,
        "chloride": 40,
        "creatinine": 0.1,
        "glucose": 0,
        "platelets x 1000": 0,
        "potassium": 1,
        "sodium": 80,
    }

    labmax = {
        "BUN": 200,
        "Hct": 70,
        "Hgb": 30,
        "MCH": 50,
        "MCHC": 60,
        "MCV": 150,
        "MPV": 20,
        "RBC": 10,
        "RDW": 25,
        "WBC x 1000": 100,
        "anion gap": 40,
        "bicarbonate": 50,
        "calcium": 20,
        "chloride": 160,
        "creatinine": 40,
        "glucose": 1000,
        "platelets x 1000": 2000,
        "potassium": 10,
        "sodium": 200,
    }

    rangemin = labmin[labname]
    rangemax = labmax[labname]

    if num == np.nan:
        return num
    # Remove values out of range
    elif num > rangemax or num < rangemin:
        return np.nan
    # Return normal values directly
    else:
        return num


def plot_roc(clf_list, x_train, y_train, x_test, y_test, name_list):
    """
    Function to plot the ROC curve for classifiers
    and list their AUC on the graph.

    Parameters:
        clf_list: the list of classifiers
        x_train: the training dataset
        y_train: the training labels
        x_test: the testing dataset
        y_test: the testing labels
        name_list: the list of classifier names
    """
    plt.figure(figsize=(15, 15))
    for index, clf in enumerate(clf_list):
        clf.fit(x_train, y_train)

        # make prediction
        y_actual = y_test.copy()
        y_pred = clf.predict_proba(x_test)[:, 1]
        fprs, tprs, thresholds = roc_curve(y_actual, y_pred)
        curve_auc = auc(fprs, tprs)
        plt.plot(fprs, tprs, label="" + name_list[index] + ": AUC = %0.2f" % curve_auc)

    plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
    plt.legend(loc="lower right", fontsize=20)
    plt.title("ROC Curve", fontsize=20)
    plt.xlabel("False Positive Rate", fontsize=20)
    plt.ylabel("True Positive Rate", fontsize=20)
    plt.show()


def plot_pr(clf_list, x_train, y_train, x_test, y_test, name_list):
    """
    Function to plot the precision-recall curve for classifiers
    and list their AUC on the graph.

    Parameters:
        clf_list: the list of classifiers
        x_train: the training dataset
        y_train: the training labels
        x_test: the testing dataset
        y_test: the testing labels
        name_list: the list of classifier names
    """
    plt.figure(figsize=(15, 15))
    for index, clf in enumerate(clf_list):
        clf.fit(x_train, y_train)

        # make prediction
        y_actual = y_test.copy()
        y_pred = clf.predict_proba(x_test)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_actual, y_pred)
        curve_auc = auc(recall, precision)
        plt.plot(
            recall, precision, label="" + name_list[index] + ": AUC = %0.2f" % curve_auc
        )
    plt.legend(loc="upper right", fontsize=20)
    plt.title("Precision-Recall Curve", fontsize=20)
    plt.xlabel("Recall", fontsize=20)
    plt.ylabel("Precision", fontsize=20)
    plt.show()


def plot_cc(clf_list, x_train, y_train, x_test, y_test, name_list):
    """
    Function to plot the calibration curve for classifiers
    and list their brier scores on the graph.

    Parameters:
        clf_list: the list of classifiers
        x_train: the training dataset
        y_train: the training labels
        x_test: the testing dataset
        y_test: the testing labels
        name_list: the list of classifier names
    """
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in zip(clf_list, name_list):
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(x_test)[:, 1]
        else:
            # use decision function
            prob_pos = clf.decision_function(x_test)
            prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        clf_score = brier_score_loss(y_test, prob_pos)
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, prob_pos, n_bins=10
        )

        ax.plot(
            mean_predicted_value,
            fraction_of_positives,
            "s-",
            label="%s: Brier = (%1.3f)" % (name, clf_score),
        )

    ax.set_xlabel("Mean predicted value", size=20)
    ax.set_ylabel("Fraction of positives", size=20)
    ax.set_ylim([-0.05, 1.05])
    ax.legend(loc="lower right", fontsize=20)
    ax.set_title("Calibration plots (reliability curve)", size=20)

    plt.tight_layout()


def k_fold_roc(clf, x, y, folds):
    """
    Function to run a cross-validation model and produce the mean ROC values

    Parameters:
        clf_list: the list of classifiers
        x: the dataset
        y: the labels
        folds: the number of folds to use
    Returns:
        mean_auc: the mean AUC
        std_auc: the standard deviation of AUC
        mean_tpr: the mean interpolated true positive rate
    """
    cv = StratifiedKFold(n_splits=folds)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for i, (train, test) in enumerate(cv.split(x, y)):
        clf.fit(x[train], y[train])
        y_pred = clf.predict_proba(x[test])[:, 1]
        fpr, tpr, thresholds = roc_curve(y[test], y_pred)
        curve_auc = auc(fpr, tpr)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(curve_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    return mean_auc, std_auc, mean_tpr


def plot_k_fold_roc(clf_list, x, y, folds, name_list):
    """
    Function to plot the k-fold ROC curves for classifiers
    and list their AUC on the graph.

    Parameters:
        clf_list: the list of classifiers
        x: the dataset
        y: the labels
        folds: the number of folds to use
        name_list: the list of classifier names
    """
    mean_aucs = []
    std_aucs = []
    mean_tprs = []
    mean_fpr = np.linspace(0, 1, 100)

    for index, clf in enumerate(clf_list):
        mean_auc, std_auc, mean_tpr = k_fold_roc(clf, x, y, folds)
        mean_aucs.append(mean_auc)
        std_aucs.append(std_auc)
        mean_tprs.append(mean_tpr)

    # plot means
    fig, ax = plt.subplots(figsize=(15, 15))
    for i in range(len(mean_aucs)):
        ax.plot(
            mean_fpr,
            mean_tprs[i],
            label=r"%s (AUC = %0.2f $\pm$ %0.2f)"
            % (name_list[i], mean_aucs[i], std_aucs[i]),
            markersize=20,
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        lw=2,
        color="r",
        label="Chance",
        alpha=0.8,
        markersize=20,
    )
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="ROC Curves")
    ax.legend(loc="lower right", fontsize=20)
    plt.xlabel("False Positive Rate", fontsize=20)
    plt.ylabel("True Positive Rate", fontsize=20)
    plt.show()
