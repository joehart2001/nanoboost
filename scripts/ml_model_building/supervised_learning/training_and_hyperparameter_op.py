import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from skopt import BayesSearchCV
from scipy.stats import randint, expon

def hyperparam_op(model, type, X, y, no_train_test = False):
    """
    Conducts hyperparameter optimization for different types of classifiers using specified search strategies.

    Args:
        model (str): Classifier type to use ('DT' for Decision Tree, 'RF' for Random Forest, 'XG' for XGBoost, 'SVM' for Support Vector Machine).
        type (str): Type of search to perform ('random', 'grid', 'sobol', 'bayes').
        X (array-like): Feature array.
        y (array-like): Label array.
        no_train_test (bool, optional): If True, skips splitting the dataset into training and testing subsets. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - X_train (array-like): Training data subset or the entire dataset if no_train_test is True.
            - X_test (array-like or None): Testing data subset or None if no_train_test is True.
            - y_train (array-like): Training labels or the entire label set if no_train_test is True.
            - y_test (array-like or None): Testing labels or None if no_train_test is True.
            - y_pred (array-like): Predicted labels for the training set.
            - search (estimator): Trained model.
            - best_params (dict): Best parameters found during the optimization.
    """
    
    if no_train_test:
        # already have the train test split
        X_train, X_test, y_train, y_test = X, None, y, None
    else:
        # stratified split to ensure equal number of each class in train and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # initialise k fold cross validation so that its the same splits for every test --> fair test and accuracy comparison
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # which model
    if model == "DT":
        clf_pipeline = Pipeline([
            ('smote', SMOTE(random_state=42)),
            ('scaler', RobustScaler()),
            ('DT', DecisionTreeClassifier(random_state=42))
        ])
        # model parameter space
        params = {"DT__max_depth": [3, None],
              "DT__max_features": randint(1, 9),
              "DT__min_samples_leaf": randint(1, 9),
              "DT__criterion": ["gini", "entropy"]}

    elif model == "RF":
        clf_pipeline = Pipeline([
            ('smote', SMOTE(random_state=42)),
            ('scaler', RobustScaler()),
            ('RF', RandomForestClassifier(random_state=42))
        ])
        
        params = {
            'RF__bootstrap': [True, False],
            'RF__n_estimators': [10, 17, 25, 33, 41, 48, 56, 64, 72, 80],
            'RF__max_depth': randint(3, 20),
            'RF__min_samples_split': randint(2, 11),
            'RF__min_samples_leaf': randint(1, 11),
            'RF__max_features': ['sqrt', None]
        }
        
    elif model == "XG":
        clf_pipeline = Pipeline([
            ('smote', SMOTE(random_state=42)),
            ('scaler', RobustScaler()),
            ('XG', XGBClassifier(random_state=42))
        ])
        
        params = {
            'XG__n_estimators': randint(100, 1000),
            'XG__max_depth': randint(3, 20),
            'XG__learning_rate': expon(scale=0.1),
            'XG__subsample': [0.5, 0.75, 1.0],
            'XG__min_child_weight' : [ 1, 3, 5, 7 ],
            'XG__gamma': [ 0.0, 0.05, 0.1, 0.15, 0.2 , 0.3, 0.4 ],
            'XG__colsample_bytree': [ 0.3, 0.4, 0.5 , 0.6, 0.7 , 0.8, 0.9, 1.0 ]} #lambda and alpha for regularisation
        
    elif model == "SVM":
        clf_pipeline = Pipeline([
            ('smote', SMOTE(random_state=42)),
            ('scaler', RobustScaler()),
            ('SVM', SVC(random_state=42))
        ])
        
        params = {
            'SVM__C': expon(scale=100),
            'SVM__kernel': ['linear', 'rbf'],
            'SVM__gamma': ['scale', 'auto']}
        
    # which optimisaiton technique
    if type == "random":
        search = RandomizedSearchCV(
            clf_pipeline,
            params,
            n_iter=60,
            cv=kf,
            scoring='accuracy',
            random_state=42
        )
    elif type == "grid":
        search = GridSearchCV(
            clf_pipeline,
            params,
            cv=kf,
            scoring='accuracy',
        )
    elif type == "sobol":
        search = BayesSearchCV( 
            clf_pipeline,
            params,
            cv=kf,
            scoring='accuracy',
            n_iter=60,
            random_state=42, 
            n_initial_points=10 # in scikit-optimize, the Sobol sequence is used by default when initial points specified
        )
    elif type == "bayes":
        search = BayesSearchCV(
            clf_pipeline,
            params,
            cv=kf,
            scoring='accuracy',
            n_iter=60,
            random_state=42,
            n_jobs=-1
        )
    
    # fit training data
    search.fit(X_train, y_train)
    accuracies = search.best_score_
    y_pred = cross_val_predict(search.best_estimator_, X_train, y_train, cv=kf, method = "predict")

    average_accuracy = search.cv_results_['mean_test_score'][search.best_index_]

    best_params = search.best_params_
    print(f"{model} best params using {type}:", best_params)
    print("Average CV accuracy:",average_accuracy, "\u00B1", np.std(accuracies))
    
    return X_train, X_test, y_train, y_test, y_pred, search, best_params