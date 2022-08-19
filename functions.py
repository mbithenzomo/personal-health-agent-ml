import logging
import numpy as np
import pandas as pd

from joblib import dump
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

logging.getLogger().setLevel(logging.INFO)

column_names = ['I_rms', 'I_mean_RR', 'I_mean_Peaks', 'I_median_RR', 'I_median_Peaks', 'I_std_RR', 'I_std_Peaks', 'I_var_RR', 'I_var_Peaks', 'I_skew_RR', 'I_skew_Peaks', 'I_kurt_RR', 'I_kurt_Peaks',
                'II_rms', 'II_mean_RR', 'II_mean_Peaks', 'II_median_RR', 'II_median_Peaks', 'II_std_RR', 'II_std_Peaks', 'II_var_RR', 'II_var_Peaks', 'II_skew_RR', 'II_skew_Peaks', 'II_kurt_RR', 'II_kurt_Peaks',
                'III_rms', 'III_mean_RR', 'III_mean_Peaks', 'III_median_RR', 'III_median_Peaks', 'III_std_RR', 'III_std_Peaks', 'III_var_RR', 'III_var_Peaks', 'III_skew_RR', 'III_skew_Peaks', 'III_kurt_RR', 'III_kurt_Peaks',
                'AVR_rms', 'AVR_mean_RR', 'AVR_mean_Peaks', 'AVR_median_RR', 'AVR_median_Peaks', 'AVR_std_RR', 'AVR_std_Peaks', 'AVR_var_RR', 'AVR_var_Peaks', 'AVR_skew_RR', 'AVR_skew_Peaks', 'AVR_kurt_RR', 'AVR_kurt_Peaks',
                'AVL_rms', 'AVL_mean_RR', 'AVL_mean_Peaks', 'AVL_median_RR', 'AVL_median_Peaks', 'AVL_std_RR', 'AVL_std_Peaks', 'AVL_var_RR', 'AVL_var_Peaks', 'AVL_skew_RR', 'AVL_skew_Peaks', 'AVL_kurt_RR', 'AVL_kurt_Peaks',
                'AVF_rms', 'AVF_mean_RR', 'AVF_mean_Peaks', 'AVF_median_RR', 'AVF_median_Peaks', 'AVF_std_RR', 'AVF_std_Peaks', 'AVF_var_RR', 'AVF_var_Peaks', 'AVF_skew_RR', 'AVF_skew_Peaks', 'AVF_kurt_RR', 'AVF_kurt_Peaks',
                'V1_rms', 'V1_mean_RR', 'V1_mean_Peaks', 'V1_median_RR', 'V1_median_Peaks', 'V1_std_RR', 'V1_std_Peaks', 'V1_var_RR', 'V1_var_Peaks', 'V1_skew_RR', 'V1_skew_Peaks', 'V1_kurt_RR', 'V1_kurt_Peaks',
                'V2_rms', 'V2_mean_RR', 'V2_mean_Peaks', 'V2_median_RR', 'V2_median_Peaks', 'V2_std_RR', 'V2_std_Peaks', 'V2_var_RR', 'V2_var_Peaks', 'V2_skew_RR', 'V2_skew_Peaks', 'V2_kurt_RR', 'V2_kurt_Peaks',
                'V3_rms', 'V3_mean_RR', 'V3_mean_Peaks', 'V3_median_RR', 'V3_median_Peaks', 'V3_std_RR', 'V3_std_Peaks', 'V3_var_RR', 'V3_var_Peaks', 'V3_skew_RR', 'V3_skew_Peaks', 'V3_kurt_RR', 'V3_kurt_Peaks',
                'V4_rms', 'V4_mean_RR', 'V4_mean_Peaks', 'V4_median_RR', 'V4_median_Peaks', 'V4_std_RR', 'V4_std_Peaks', 'V4_var_RR', 'V4_var_Peaks', 'V4_skew_RR', 'V4_skew_Peaks', 'V4_kurt_RR', 'V4_kurt_Peaks',
                'V5_rms', 'V5_mean_RR', 'V5_mean_Peaks', 'V5_median_RR', 'V5_median_Peaks', 'V5_std_RR', 'V5_std_Peaks', 'V5_var_RR', 'V5_var_Peaks', 'V5_skew_RR', 'V5_skew_Peaks', 'V5_kurt_RR', 'V5_kurt_Peaks',
                'V6_rms', 'V6_mean_RR', 'V6_mean_Peaks', 'V6_median_RR', 'V6_median_Peaks', 'V6_std_RR', 'V6_std_Peaks', 'V6_var_RR', 'V6_var_Peaks', 'V6_skew_RR', 'V6_skew_Peaks', 'V6_kurt_RR', 'V6_kurt_Peaks']


def load_dataset(dataset_num):
    """
    Parameters:
    dataset_num: which dataset to load
        - 1: AF and None
        - 2: AF, Other and None
    """
    if dataset_num not in [1, 2]:
        raise ValueError('Invalid dataset selected: ', str(dataset_num))

    dataset = pd.read_csv('data/dataset_chapman.csv')
    assert dataset.shape == (5340, 159)
    logging.info('Data loaded successfully')

    if dataset_num == 1:
        logging.info('Dropping the "other" class')
        dataset = dataset[dataset['label'] != 2]
        assert dataset.shape == (3560, 159)

    return dataset

def normalise_features(features):
    logging.info('Normalising the features')
    scaler = StandardScaler()
    scaled_features = features.iloc[:, :-2]
    scaled_features = scaler.fit_transform(scaled_features)
    scaled_features = pd.DataFrame(scaled_features)
    scaled_features.columns = column_names # restore column names
    scaled_features.index = features.index
    scaled_features[['age', 'sex']] = features[['age', 'sex']]
    return scaled_features

def get_features_and_labels(which_lead, demographics, normalised, dataset_num):
    """
    Generate features (X) and labels (y)

    Parameters:
    which_lead: which lead(s) to include
    demographics: whether to include demographic features (age and sex) or not
    normalised: whether to normalise the features or not
    """
    if which_lead not in ['all', 'six', 1, 2]:
        raise ValueError('Invalid selection of leads: ', str(which_lead))

    if not isinstance(demographics, (bool)):
        raise TypeError('Should be either True or False')

    if not isinstance(normalised, (bool)):
        raise TypeError('Should be either True or False')

    dataset = load_dataset(dataset_num)
    y = dataset['label']
    X = dataset.drop(columns=['label'])

    if normalised:
        X = normalise_features(X)

    if which_lead != 'all':
        if which_lead == 'six':
            start = 0
            stop = 78
        else:
            start = (which_lead-1)*13
            stop = which_lead*13

    if demographics:
        if which_lead != 'all':
            X = X.iloc[:, np.r_[start:stop, 156, 157]]
    else:
        if which_lead == 'all':
            X = X.drop(columns=['age', 'sex'])
        else:
            X = X.iloc[:, np.r_[start:stop]]

    datasets = {'X': X, 'y': y}
    return datasets

def evaluation_and_fitting(model, which_lead, demographics, normalised, dataset_num):
    """
    Evaluate a model using cross-validation
    Get classification reports and confusion matrices
    Fit and save the model
    """

    datasets = get_features_and_labels(which_lead, demographics, normalised, dataset_num)
    X = datasets['X']
    y = datasets['y']
    y_list = []
    y_pred_list = []

    def class_report_with_accuracy(y, y_pred):
        y_list.extend(y)
        y_pred_list.extend(y_pred)
        return accuracy_score(y, y_pred)

    logging.info('Getting cross-validation scores')
    cross_val_score(model, X, y, scoring=make_scorer(
        class_report_with_accuracy), cv=10)
    logging.info('Done')

    if dataset_num == 1:
        labels = [1, 0] # AF, None
    else:
        labels = [1, 2, 0] # AF, Other, None

    logging.info('Getting classification report')
    class_report = classification_report(y_list, y_pred_list, digits=4,
                                         labels=labels)
    logging.info('Done')

    logging.info('Getting confusion matrix')
    conf_matrix = confusion_matrix(y_list, y_pred_list,
                                   labels=labels)
    logging.info('Done')

    logging.info('Fitting and saving model')
    model.fit(X, y)
    filename = str(model)[0:3] + '_lead_' + str(which_lead) + \
                '_demo_' + str(demographics)[0] + \
                '_norm_' + str(normalised)[0] + \
                '_ds_' + str(dataset_num)
    path = 'saved_models/' + filename + '.joblib'
    dump(model, path)
    logging.info('Done')

    report = {}
    report['class_report'] = str(class_report)
    report['conf_matrix'] = str(conf_matrix)

    return report
