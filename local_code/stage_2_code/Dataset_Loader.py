'''
Concrete IO class for a specific dataset
'''
import csv

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_train_file_name = None
    dataset_source_test_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load_file(self, file_path):
        print('loading data...')
        X = []
        y = []
        f = open(file_path, 'r')
        reader = csv.reader(f)
        for row in reader:
            label = int(row[0])
            features = [int(value) for value in row[1:]]
            y.append(label)
            X.append(features)
        return X, y

    def load(self):
        print('loading data...')

        train_path = self.dataset_source_folder_path + self.dataset_source_train_file_name
        test_file_path = self.dataset_source_folder_path + self.dataset_source_test_file_name

        X_train, y_train = self.load_file(train_path)
        print(f' Train: {len(X_train)} samples loaded')

        X_test, y_test = self.load_file(test_file_path)
        print(f' Test: {len(X_test)} samples loaded')

        print(f' Features per sample : {len(X_train[0])}')
        print(f' Classes : {sorted(set(y_train))}')

        return{
            'train' : {'X' : X_train, 'y' : y_train},
            'test' : {'X' : X_test, 'y' : y_test}
        }


