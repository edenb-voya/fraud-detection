import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline   

def score_classification(y_train, y_train_pred, y_test, y_test_pred):
    
    scores = pd.DataFrame(data = np.array([[accuracy_score(y_train, y_train_pred), accuracy_score(y_test, y_test_pred)],
                                          [balanced_accuracy_score(y_train, y_train_pred), balanced_accuracy_score(y_test, y_test_pred)],
                                          [precision_score(y_train, y_train_pred), precision_score(y_test, y_test_pred)],
                                          [recall_score(y_train, y_train_pred), recall_score(y_test, y_test_pred)],
                                          [f1_score(y_train, y_train_pred), f1_score(y_test, y_test_pred)],
                                          [roc_auc_score(y_train, y_train_pred), roc_auc_score(y_test, y_test_pred)],
                                          [brier_score_loss(y_train, y_train_pred), brier_score_loss(y_test, y_test_pred)],
                                          [log_loss(y_train, y_train_pred), log_loss(y_test, y_test_pred)],
                                          [jaccard_score(y_train, y_train_pred), jaccard_score(y_test, y_test_pred)]]),
                          index = ['Accuracy', 
                                   'Balanced_Accuracy', 
                                   'Precision', 
                                   'Recall', 
                                   'f1',
                                   'ROC_AUC',
                                   'Brier_Loss',
                                   'Log_Loss',
                                   'Jaccard'], 
                          columns = ['Train', 'Test'])
    print(scores)
    print(confusion_matrix(y_test, y_test_pred))
    
def downsample(df, target):

    is_0 =  df[target]==0 
    is_1 =  df[target]==1

    if is_0.sum() > is_1.sum():
        df_majority = df[is_0]
        df_minority = df[is_1]
    else:
        df_majority = df[is_1]
        df_minority = df[is_0]

    df_majority_downsampled = resample(df_majority, 
                                       replace=False,   
                                       n_samples=df_minority.shape[0],    
                                       random_state=42)
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])

    return df_downsampled
 
def scaled_model_search(scalers, models, X_train, y_train, X_test, y_test):

    best_score = 0
    
    for scaler in scalers:
        for model in models:
            pipe = Pipeline(steps=[('scaler', scaler),
                              ('classifier', model)])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            score = accuracy_score(y_test, y_pred)

            if score > best_score:
                best_score = score
                best_model = model
                best_scaler = scaler
    print("The best model is {}, scaled by {}, with a test (accuracy) score of {}.".format(best_model, best_scaler, best_score)) 