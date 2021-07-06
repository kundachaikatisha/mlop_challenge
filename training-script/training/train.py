import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import lightgbm


def split_data(data_df):
    """Split a dataframe into training and validation datasets"""

    features = data_df.drop(['target', 'id'], axis=1)
    labels = np.array(data_df['target'])
    features_train, features_valid, labels_train, labels_valid = \
        train_test_split(features, labels, test_size=0.2, random_state=0)

    train_data = lightgbm.Dataset(features_train, label=labels_train)
    valid_data = lightgbm.Dataset(
        features_valid,
        label=labels_valid,
        free_raw_data=False
        )

    return (train_data, valid_data)
#commit test
z=1+2+3

def train_model(train_data, valid_data, parameters):
    """Train a model with the given datasets and parameters"""
    # The object returned by split_data is a tuple.
    # Access train_data with data[0] and valid_data with data[1]

    model = lightgbm.train(parameters,
                           train_data,
                           valid_sets=valid_data,
                           num_boost_round=500,
                           early_stopping_rounds=20)

    return model


#def get_model_metrics(model, train_data, valid_data):
 #   """Construct a dictionary of metrics for the model"""
#
#    predictions = model.predict(valid_data.data)
#    fpr, tpr, thresholds = metrics.roc_curve(valid_data.label, predictions)
#    model_metrics = {"auc": (metrics.auc(fpr, tpr))}
#
#    return model_metrics

def get_model_metrics(model, train_data, valid_data):
    predictions = model.predict(valid_data.data)
    fpr, tpr, thresholds = metrics.roc_curve(valid_data.label, predictions)
    classified_perdection = predictions

    for index in range(len(predictions)):
        if(predictions[index] > 0.25):
            classified_perdection[index] = 0
        else: 
            classified_perdection[index] = 1

    #model_metrics = {"auc": (metrics.auc(fpr, tpr))}
    model_metrics = {"auc": (metrics.auc(fpr, tpr)), "f1_score":(metrics.f1_score(valid_data.label, classified_perdection))}
    #metrics.f1_score(valid_data.label, predictions)
    return model_metrics


    

def main():
    """This method invokes the training functions for development purposes"""

    # Read data from a file
    data_df = pd.read_csv('porto_seguro_safe_driver_prediction_input.csv')

    # Hard code the parameters for training the model
    parameters = {
        'learning_rate': 0.02,
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'sub_feature': 0.7,
        'num_leaves': 60,
        'min_data': 100,
        'min_hessian': 1,
        'verbose': 2
    }

    # split data
    train_data, valid_data = split_data(data_df)

    # train model
    model = train_model(train_data, valid_data, parameters)

    # validate
    metric = get_model_metrics(model, train_data, valid_data)

    print(metric)

    # Print the resulting metrics for the model


if __name__ == '__main__':
    main()
