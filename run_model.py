import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
# import cupy as cp
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, accuracy_score, f1_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.utils import compute_sample_weight
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import SGDRegressor, RidgeCV, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, StackingRegressor, StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectFromModel, RFE
import shap
import logging
import argparse


# Setup logging configuration
logging.basicConfig(filename='results.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')


# Custom function to log and print
def log_and_print(message, printed=True):
    """Logs a message and prints it to the console."""
    logging.info(message)
    if printed:
        print(message)


def model_crossvalidation(args, params, train_data):

    n = 5000
    model = xgb.cv(
    params=params,
    dtrain=train_data,
    num_boost_round=n,
    nfold=5,
    early_stopping_rounds=50
    )

    print(model.head())


def model_shap(args, model, X_train):

    if args.mode == "boosting":
        explainer = shap.TreeExplainer(model)
    elif args.mode == "stacking":
        explainer = shap.KernelExplainer(model.predict, X_train)

    shap_values = explainer.shap_values(X_train)
    shap.plots.waterfall(shap_values[0])


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", default="boosting", choices=["stacking", "boosting"],
                        help="Sets the model type")
    parser.add_argument("-r", "--regression", default=False, action="store_true",
                        help="Set the model to the regression task")
    parser.add_argument("-cv","--cross_validation", action='store_true', default=None,
                        help="Runs the cross-validation on the model")
    parser.add_argument("-e","--eta", type=float, default=0.05,
                        help="Sets the learning rate")
    parser.add_argument("-md", "--max_depth", type=int, default=3,
                        help="Sets the max depth of each tree")
    parser.add_argument("-n","--n_rounds", type=int, default=1000,
                        help="Sets the number of boost rounds")
    parser.add_argument("-d","--device", default="cpu", choices=["cpu", "cuda"],
                        help="Allows to enable GPU when running the code")
    parser.add_argument("-rfe", "--rfe", default=False, action="store_true",
                        help="Enables the RFE (Recursive Feature Elimitation)")
    parser.add_argument("-s","--shap", default=False, action="store_true",
                        help="Enables the SHAP stats for feature performance")
    parser.add_argument("-g", "--grid_search", default=False, action="store_true",
                        help="Enables Grid-search of hyperparameters before running the model")

    args = parser.parse_args()
    return args


def get_features(df, args):

    # Convert strings to numbers
    df['words'] = df['words'].astype('Int64')
    df['chapters'] = df['chapters'].astype('Int64')

    # Convert strings to dates
    df['pubDate'] = pd.to_datetime(df['pubDate'])
    df['modDate'] = pd.to_datetime(df['modDate'])
    df['packDate'] = pd.to_datetime(df['packDate'])

    # Extract useful date features (Year, Month, Day, etc.)
    df['pub_year'] = df['pubDate'].dt.year
    #df['pub_month'] = df['pubDate'].dt.month
    df['mod_year'] = df['modDate'].dt.year
    df['pack_year'] = df['packDate'].dt.year
    #df['pack_month'] = df['packDate'].dt.month

    df = df.drop(columns=['pubDate', 'modDate', 'packDate'])

    # Encode the strings to categorical values
    df['pubStat'] = LabelEncoder().fit_transform(df['pubStat'])
    df['rating'] = LabelEncoder().fit_transform(df['rating'])

    if args.regression:
        y = df['kudos']
    else:
        y = df['kudos_categorical']

    x = df.drop(columns=['kudos', 'story', 'title', 'kudos_label', 'kudos_categorical'])
    return x, y


def grid_search(model, args, pipeline_str=""):

    param_grid = {
            f'{pipeline_str}learning_rate': [0.005, 0.01, 0.05, 0.1],
            f'{pipeline_str}max_depth': [1, 3, 5, 7, 10, 12],
            f'{pipeline_str}gamma': [0, 0.1, 0.3, 0.5], 
            f'{pipeline_str}subsample': [0.5, 0.8, 1.0],
            f'{pipeline_str}reg_alpha':[0, 0.001, 0.005, 0.01, 0.05],
            f'{pipeline_str}colsample_bytree': [0.5, 0.7, 0.8, 0.9, 1.0],
            f'{pipeline_str}n_estimators': [50, 75, 100],
        }

    # remove unnecessary for regression parameter
    if args.mode == "regression":
        del param_grid[f'{pipeline_str}n_estimators']

    model = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=1)
    log_and_print(f"Params Grid: {param_grid}")

    return model


def main():
    args = create_arg_parser()

    # open train and test data
    with open("sparql_data_train_3.csv") as f1,\
         open("sparql_data_test_3.csv") as f2:
        train = pd.read_csv(f1)
        test = pd.read_csv(f2)

    X_train, y_train = get_features(train, args)
    X_test, y_test = get_features(test, args)

    params = {
        'learning_rate': args.eta,
        'colsample_bytree': 1.0,
        'max_depth': 10,
        'n_estimators': 50,
        'subsample': 0.8,
        }

    dtrain = xgb.DMatrix(X_train, y_train, enable_categorical=True)
    dtest = xgb.DMatrix(X_test, y_test, enable_categorical=True)

    if args.mode == "stacking":
        if args.regression:
            estimators = [
                ('lr', RidgeCV()),
                ('sgd', SGDRegressor(max_iter=1000, tol=1e-3, alpha=0.01))
            ]
            model = StackingRegressor(
                estimators=estimators,
                final_estimator=RandomForestRegressor(n_estimators=100,
                                                      max_depth=10,
                                                      min_samples_split=3,
                                                      min_samples_leaf=3,
                                                      max_leaf_nodes=2)
            )

            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            if args.shap:
                #model_best = model.best_estimator_
                model_shap(args, model,X_train)

            # Calculate the RMSE on the test set
            score = root_mean_squared_error(y_test, preds)
            print(f"RMSE of the model: {score:.3f}")
        else:
            estimators = [
                ('rf', RandomForestClassifier(n_estimators=100)),
                ('svr', make_pipeline(StandardScaler(), LinearSVC(C=0.01, penalty='l2', loss='squared_hinge')))
            ]
            model = StackingClassifier(
                estimators=estimators, final_estimator=LogisticRegression()
            )
            model.fit(X_train, y_train)
                            # Analyze the feature performance with SHAP
            if args.shap:
                #model_best = model.best_estimator_
                model_shap(args, model,X_train)

            preds = model.predict(X_test)
            print(classification_report(y_test, preds))
    else:
        # Run regression 
        if args.regression:

            params = {'colsample_bytree': 0.7,
                    'gamma': 0,
                    'learning_rate': 0.05,
                    'max_depth': 3,
                    'reg_alpha': 0,
                    'subsample': 1.0}

            params['objective'] = 'reg:squarederror'
            params['eta'] = params.pop('learning_rate')

            model = xgb.XGBRegressor(device=args.device)
            model.set_params(**params)

            if args.cross_validation:
                model_crossvalidation(args, params, dtrain)
            else:
                if args.grid_search:
                    model = grid_search(model, args)
                    model.fit(X_train, y_train)
                    log_and_print(f"Best parameters: {model.best_params_}")
                else:
                    model.fit(X_train, y_train)

                # Analyze the feature performance with SHAP
                if args.shap:
                    #model_best = model.best_estimator_
                    model_shap(args, model,X_train)

                preds = model.predict(X_test)
                # Calculate the RMSE on the test set
                score = root_mean_squared_error(y_test, preds)
                print(f"RMSE of the model: {score:.3f}")
        # Run classification
        else:
            params['num_class'] = 3

            if args.cross_validation:
                model_crossvalidation(args, params, dtrain)
            else:
                sample_weight = compute_sample_weight({0: 1, 1: 5, 2: 1}, y_train)

                model = xgb.XGBClassifier(
                    # num_class=3,
                    device=args.device,
                )

                model.set_params(**params)

                # Run model with RFE
                if args.rfe:
                    pipeline = Pipeline([
                        ('feature_selection', RFE(estimator=model, n_features_to_select=4)),
                        ('classifier', model)])

                    if args.grit_search:
                        # Perform the grid search for the hyperparameter tuning
                        model = grid_search(pipeline, args, "classifier__")

                    model.fit(X_train, y_train, 
                            classifier__sample_weight=sample_weight
                            )
                # Run grid_search only
                elif args.grid_search:
                    model = grid_search(model, args)
                    model.fit(X_train, y_train, 
                                sample_weight=sample_weight
                            )
                    log_and_print(f"Best parameters: {model.best_params_}")
                # Run base model
                else:
                    model.fit(X_train, y_train, 
                                sample_weight=sample_weight
                            )

                # Analyze the feature performance with SHAP
                if args.shap:
                    #model_best = model.best_estimator_
                    model_shap(args, model,X_train)

                # Predict the labels
                preds = model.predict(X_test)

                print(classification_report(y_test, preds))
                #log_and_print(classification_report(y_test, preds))

if __name__ == "__main__":
    main()