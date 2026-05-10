'''
Concrete Evaluate class for a specific evaluation metrics
Stage 2: Accuracy + F1 / Precision / Recall in weighted, macro, and micro variants.

Why not binary F1?
  Binary F1 is only valid for two-class problems where one class is the "positive" class.
  For multiclass we aggregate per-class scores:
    - weighted: accounts for class imbalance (weights each class by its support)
    - macro: treats every class equally regardless of support
    - micro: computes globally by counting total TP/FP/FN across all classes
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import (
    accuracy_score,
    f1_score, precision_score, recall_score, classification_report,
)
import numpy as np

'''copy pasted from stage 2'''

class Evaluate_Accuracy(evaluate):
    data = None
    
    def evaluate(self):
        print('evaluating performance...')

        true_y = np.array(self.data['true_y'])
        pred_y = np.array(self.data['pred_y'])

        metrics = {
            'accuracy': accuracy_score(true_y, pred_y),

            'f1_weighted': f1_score(true_y, pred_y, average='weighted', zero_division=0),
            'f1_micro': f1_score(true_y, pred_y, average='micro', zero_division=0),
            'f1_macro': f1_score(true_y, pred_y, average='macro', zero_division=0),

            'precision_weighted': precision_score(true_y, pred_y, average='weighted', zero_division=0),
            'precision_micro': precision_score(true_y, pred_y, average='micro', zero_division=0),
            'precision_macro': precision_score(true_y, pred_y, average='macro', zero_division=0),

            'recall_weighted': recall_score(true_y, pred_y, average='weighted', zero_division=0),
            'recall_micro': recall_score(true_y, pred_y, average='micro', zero_division=0),
            'recall_macro': recall_score(true_y, pred_y, average='macro', zero_division=0),
        }
        return metrics

    def print_report(self):
        true_y = np.array(self.data['true_y'])
        pred_y = np.array(self.data['pred_y'])
        print('\n Classification Report')
        print(classification_report(true_y, pred_y, zero_division=0))
        