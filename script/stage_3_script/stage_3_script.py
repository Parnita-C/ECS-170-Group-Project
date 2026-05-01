'''
Main experiment script for CNN on MNIST, CIFAR, and ORL datasets.
'''

import pickle
import numpy as np
import local_code.stage_3_code.Method_CNN as m
print(m.__file__)

from local_code.stage_3_code.Method_CNN import Method_CNN
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy

np.random.seed(2)

# ---- select dataset here ----
DATASET_NAME = 'MNIST'   # 'MNIST', 'CIFAR', or 'ORL'

DATA_PATH = f'../../data/stage_3_data/{DATASET_NAME}'

# ---- load dataset ----
print(f'Loading {DATASET_NAME} dataset...')
with open(DATA_PATH, 'rb') as f:
    loaded_data = pickle.load(f)
print(f'  Train: {len(loaded_data["train"])} samples')
print(f'  Test : {len(loaded_data["test"])} samples')

# ---- method ----
method_obj = Method_CNN('CNN', '', dataset_name=DATASET_NAME)
method_obj.data = loaded_data   # list-of-dicts format: [{'image':…, 'label':…}, …]

# ---- result saver ----
result_obj = Result_Saver('saver', '')
result_obj.result_destination_folder_path = '../../result/stage_3_result/'
result_obj.result_destination_file_name = f'CNN_{DATASET_NAME}_prediction_result'
result_obj.fold_count = 1

# ---- evaluation ----
evaluate_obj = Evaluate_Accuracy('accuracy', '')

# ---- run ----
print('************ Start ************')
learned_result = method_obj.run()

# ---- save ----
result_obj.data = learned_result
result_obj.save()

# ---- evaluate ----
evaluate_obj.data = learned_result
metrics = evaluate_obj.evaluate()

print('************ Overall Performance ************')
print(f'Accuracy         : {metrics["accuracy"]:.4f}')
print(f'F1  (weighted)   : {metrics["f1_weighted"]:.4f}')
print(f'F1  (macro)      : {metrics["f1_macro"]:.4f}')
print(f'F1  (micro)      : {metrics["f1_micro"]:.4f}')
print(f'Prec (weighted)  : {metrics["precision_weighted"]:.4f}')
print(f'Recall (weighted): {metrics["recall_weighted"]:.4f}')
evaluate_obj.print_report()
print('************ Finish ************')