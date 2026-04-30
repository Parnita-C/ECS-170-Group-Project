'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting
import numpy as np

'''copy pasted from stage 2 below'''

class Setting_No_Split(setting):
    train_ratio = 0.8

    def load_run_save_evaluate(self):
        loaded_data = self.dataset.load()

        X_train = np.array(loaded_data['train']['X'])
        y_train = np.array(loaded_data['train']['y'])
        X_test = np.array(loaded_data['test']['X'])
        y_test = np.array(loaded_data['test']['y'])
        print(f' Pre-split: {len(X_train)} train / {len(X_test)} test')

        #run method
        self.method.data = {
            'train': {'X': X_train, 'y': y_train},
            'test': {'X': X_test, 'y': y_test},
        }
        learned_result = self.method.run()

        #save raw result
        self.result.data = learned_result
        self.result.save()

        #evaluate
        self.evaluate.data = learned_result
        metrics = self.evaluate.evaluate()

        print(f" Accuracy          : {metrics['accuracy']:.4f}")
        print(f" F1  (weighted)    : {metrics['f1_weighted']:.4f}")
        print(f" F1  (macro)       : {metrics['f1_macro']:.4f}")
        print(f" F1  (micro)       : {metrics['f1_micro']:.4f}")
        print(f" Prec (weighted)   : {metrics['precision_weighted']:.4f}")
        print(f" Prec (macro)      : {metrics['precision_macro']:.4f}")
        print(f" Prec (micro)      : {metrics['precision_micro']:.4f}")
        print(f" Recall (weighted) : {metrics['recall_weighted']:.4f}")
        print(f" Recall (macro)    : {metrics['recall_macro']:.4f}")
        print(f" Recall (micro)    : {metrics['recall_micro']:.4f}")

        self.evaluate.print_report()

        return metrics, None
