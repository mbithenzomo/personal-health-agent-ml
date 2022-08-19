from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from functions import evaluation_and_fitting

# parameters obtained by hyperparameter tuning
# see hyperparameter_tuning.py and reports/hyp_tuning.txt
gb = GradientBoostingClassifier(criterion='friedman_mse', loss='log_loss',
                                 max_depth=3, n_estimators=800)
mlp = MLPClassifier(hidden_layer_sizes=(158, 100, 50), activation='tanh',
                    learning_rate='adaptive', max_iter=500)
rf = RandomForestClassifier(n_estimators=800, max_features='sqrt',
                            max_depth=None, criterion='entropy')
svm = SVC(C=1.5, kernel='rbf', gamma='auto', decision_function_shape='ovo')

models = [gb, mlp, rf, svm]
leads_list = [1, 2]
demographics_list = [True, False]
normalised_list = [True, False]
datasets_list = [1, 2]

# all leads
for model in models:
    for dem in demographics_list:
        for norm in normalised_list:
            for dataset in datasets_list:
                report = evaluation_and_fitting(model,
                                                which_lead = 'all',
                                                demographics = dem,
                                                normalised = norm,
                                                dataset_num = dataset)
                if dataset == 1:
                    classes = 'AF, None'
                else:
                    classes = 'AF, Other, None'
                output = ['######## ' + str(model),
                          'Leads: All',
                          'Demographics: ' + str(dem),
                          'Normalised data: ' + str(norm),
                          'Classes: '  + classes,
                          'Classification report:', report['class_report'],
                          'Confusion matrix:', report['conf_matrix']]
                path = 'reports/classification/' + str(model)[0:3] + '/all_leads.txt'
                with open(path, 'a',
                          encoding='utf-8') as file:
                    for item in output:
                        file.write(item)
                        file.write('\n')
                    file.write('\n')

# 6 leads
for model in models:
    print('Model:', str(model))
    dem = True # all models perform better when demographic data is included
    if str(model)[0:3] == 'Ran' or str(model)[0:3] == 'Gra':
        norm = False # RF and GB perform slightly better on non-normalised data
    else:
        norm = True # SVM and MLP perform much better on normalised data
    for dataset in datasets_list:
        print('Dataset:', str(dataset))
        report = evaluation_and_fitting(model,
                                        which_lead = 'six',
                                        demographics = dem,
                                        normalised = norm,
                                        dataset_num = dataset)
        if dataset == 1:
            classes = 'AF, None'
        else:
            classes = 'AF, Other, None'
        output = ['######## ' + str(model),
                  'Leads: Six',
                  'Demographics: ' + str(dem),
                  'Normalised data: ' + str(norm),
                  'Classes: '  + classes,
                  'Classification report:', report['class_report'],
                  'Confusion matrix:', report['conf_matrix']]
        path = 'reports/classification/' + str(model)[0:3] + '/six_leads.txt'
        with open(path, 'a',
                  encoding='utf-8') as file:
            for item in output:
                file.write(item)
                file.write('\n')
            file.write('\n')

# single leads
for model in models:
    dem = True # all models perform better when demographic data is included
    if str(model)[0:3] == 'Ran' or str(model)[0:3] == 'Gra':
        norm = False # RF and GB perform slightly better on non-normalised data
    else:
        norm = True # SVM and MLP perform much better on normalised data
    for lead in leads_list:
        for dataset in datasets_list:
            report = evaluation_and_fitting(model,
                                            which_lead = lead,
                                            demographics = dem,
                                            normalised = norm,
                                            dataset_num = dataset)
            if dataset == 1:
                classes = 'AF, None'
            else:
                classes = 'AF, Other, None'
            output = ['######## ' + str(model),
                      'Lead: ' + str(lead),
                      'Demographics: ' + str(dem),
                      'Normalised data: ' + str(norm),
                      'Classes: '  + classes,
                      'Classification report:', report['class_report'],
                      'Confusion matrix:', report['conf_matrix']]
            path = 'reports/classification/' + str(model)[0:3] + '/lead_'+ str(lead) + '.txt'
            with open(path, 'a',
                      encoding='utf-8') as file:
                for item in output:
                    file.write(item)
                    file.write('\n')
                file.write('\n')
