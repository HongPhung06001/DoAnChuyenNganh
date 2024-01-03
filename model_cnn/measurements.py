from sklearn.metrics import classification_report, precision_recall_fscore_support
import numpy as np
import pandas as pd

import keras

def print_measurements(y_pred, y_test, labels, name, dir):
    y_pred_argmax = [np.argmax(prob) for prob in y_pred]
    y_pred_binary = [[1 if i == n else 0 for i in range(len(labels))] for n in y_pred_argmax]

    report = classification_report(y_test, y_pred_binary, target_names=labels, zero_division=0)

    print(report)

    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred_binary)

    report_df = pd.DataFrame(index=labels)
    report_df['precision'] = precision
    report_df['recall'] = recall
    report_df['f1-score'] = fscore
    report_df['support'] = support

    report_df.to_csv(f'{dir}/classification_report_{name}.csv')