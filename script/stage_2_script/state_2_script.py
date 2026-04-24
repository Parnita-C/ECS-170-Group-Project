'''
Main experiment script for MLP on MNIST dataset
'''

from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Setting_No_Split import Setting_No_Split
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np

np.random.seed(2)

# ---- dataset ----
data_obj = Dataset_Loader('MNIST', '')
data_obj.dataset_source_folder_path = r'C:\Users\chpar\ECS 170\data\stage_2_data\\'   # <-- folder where train.csv and test.csv live
data_obj.dataset_source_train_file_name = 'train.csv'
data_obj.dataset_source_test_file_name = 'test.csv'

# ---- method ----
method_obj = Method_MLP('MLP', '')
# 784 input features (28x28 pixels), 10 output classes (digits 0-9)
# Note: update the Linear layer sizes inside Method_MLP.py as well:
#   self.fc_layer_1 = nn.Linear(784, 784)
#   self.fc_layer_2 = nn.Linear(784, 10)

# ---- result saver ----
result_obj = Result_Saver('saver', '')
result_obj.result_destination_folder_path = r'C:\Users\chpar\ECS 170\result\stage_2_result\MLP_'
result_obj.result_destination_file_name = 'prediction_result'
result_obj.fold_count = 1

# ---- evaluation ----
evaluate_obj = Evaluate_Accuracy('accuracy', '')

# ---- setting ----
setting_obj = Setting_No_Split('no split', '')
setting_obj.dataset = data_obj
setting_obj.method = method_obj
setting_obj.result = result_obj
setting_obj.evaluate = evaluate_obj

print('************ Start ************')
metrics, _ = setting_obj.load_run_save_evaluate()
print('************ Overall Performance ************')
print('Accuracy:', metrics['accuracy'])
print('F1 (weighted):', metrics['f1_weighted'])
print('************ Finish ************')