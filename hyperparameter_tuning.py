import logging

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from functions import get_features_and_labels

logging.getLogger().setLevel(logging.INFO)


def gridsearch(model, parameters):
    clf = GridSearchCV(model, parameters)
    return clf.fit(X, y)


models = []

# Random Forest
rf = RandomForestClassifier()
rf_params = dict(n_estimators=[300, 500, 800],
                 criterion=['gini', 'entropy', 'log_loss'],
                 max_depth=[50, 100, None],
                 max_features=['sqrt', 'log2', None])
models.append({'name': 'Random Forest', 'model': rf,
               'params': rf_params})

# Support Vector Machine
svm = SVC()
svm_params = dict(C=[0.5, 1, 1.5],
                  kernel=['poly', 'rbf', 'sigmoid'],
                  gamma=['scale', 'auto'],
                  decision_function_shape=['ovo', 'ovr'])
models.append({'name': 'Support Vector Machine', 'model': svm,
               'params': svm_params})

# Gradient Boosting
gb = GradientBoostingClassifier()
gb_params = dict(n_estimators=[300, 500, 800],
                 criterion=['friedman_mse', 'squared_error', 'mse'],
                 loss=['log_loss', 'exponential'],
                 max_depth=[1, 3, 10])
models.append({'name': 'Gradient Boosting', 'model': gb,
               'params': gb_params})

# Multilayer Perceptron
mlp = MLPClassifier(hidden_layer_sizes=(158, 100, 50))
mlp_params = dict(activation=['identity', 'logistic', 'tanh', 'relu'],
                  batch_size=['auto', 64, 100],
                  solver=['lbfgs', 'sgd', 'adam'],
                  learning_rate=['constant', 'invscaling', 'adaptive'],
                  max_iter=[200, 500, 1000, 2000])
models.append({'name': 'Multilayer Perceptron', 'model': mlp,
               'params': mlp_params})


which_lead = 'all'
demographics = True
normalised = True
dataset_num = 2

datasets = get_features_and_labels(which_lead, demographics, normalised, dataset_num)
X = datasets['X']
y = datasets['y']

for model in models:
    logging.info('Hyperparameter tuning for %s', model['name'])
    search = gridsearch(model['model'], model['params'])
    output = [model['name'],
              'Leads: ' + which_lead,
              'Demographics:  ' + str(demographics),
              'Normalised data: ' + str(normalised),
              'Dataset: '  + str(dataset_num),
              str(search.best_params_)]
    with open('reports/hyp_tuning.txt', 'a', encoding='utf-8') as file:
        file.write('###############\n')
        for item in output:
            file.write(item)
            file.write('\n')
        file.write('\n')
