'''
Concrete MethodModule class for a CNN model supporting MNIST, CIFAR-10, and ORL datasets.
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class Method_CNN(method, nn.Module):
    data = None
    # dataset name controls which CNN architecture is built: 'MNIST', 'CIFAR', or 'ORL'
    dataset_name = 'MNIST'

    max_epoch = 5
    learning_rate = 1e-3
    batch_size = 64

    # ---------- architecture builders ----------

    def _build_mnist(self):
        """
        MNIST: grayscale 28x28 (1 channel), 10 classes.
        Two conv blocks followed by a small FC head.
        """
        self.num_classes = 10
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # -> 32 x 28 x 28
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                               # -> 32 x 14 x 14

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # -> 64 x 14 x 14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                               # -> 64 x 7 x 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, self.num_classes),
        )

    def _build_cifar(self):
        """
        CIFAR-10: color 32x32x3 (3 channels), 10 classes.
        Three conv blocks to capture richer color features.
        """
        self.num_classes = 10
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # -> 32 x 32 x 32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                               # -> 32 x 16 x 16

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # -> 64 x 16 x 16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                               # -> 64 x 8 x 8

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # -> 128 x 8 x 8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),                               # -> 128 x 4 x 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes),
        )

    def _build_orl(self):
        """
        ORL: grayscale 112x92 (use R channel only = 1 channel), 40 classes.
        Three conv blocks to handle the larger spatial resolution.
        """
        self.num_classes = 40
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # -> 32 x 112 x 92
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                               # -> 32 x 56 x 46

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # -> 64 x 56 x 46
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                               # -> 64 x 28 x 23

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # -> 128 x 28 x 23
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),                               # -> 128 x 14 x 11
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 14 * 11, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes),
        )

    # ---------- init ----------

    def __init__(self, mName, mDescription, dataset_name='MNIST'):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.dataset_name = dataset_name

        # Build both sub-networks immediately so every instance attribute
        # is defined inside __init__ (avoids "attribute defined outside
        # __init__" inspection warnings).
        self.num_classes: int
        self.features: nn.Sequential
        self.classifier: nn.Sequential
        self._build_network()

    def _build_network(self):
        name = self.dataset_name.upper()
        if name == 'MNIST':
            self._build_mnist()
        elif name == 'CIFAR':
            self._build_cifar()
        elif name == 'ORL':
            self._build_orl()
        else:
            raise ValueError(f"Unknown dataset_name '{self.dataset_name}'. "
                             "Expected 'MNIST', 'CIFAR', or 'ORL'.")

    # ---------- forward ----------

    def forward(self, x):
        '''Forward propagation through conv feature extractor then FC classifier.'''
        return self.classifier(self.features(x))

    # ---------- data preprocessing ----------

    def _preprocess(self, instances):
        """
        Convert a list of {'image': …, 'label': …} dicts into
        (X_tensor, y_tensor) ready for the CNN.

        Image shapes handled:
          MNIST  -> (28, 28)       -> tensor (N, 1, 28, 28), float32 / 255
          CIFAR  -> (32, 32, 3)   -> tensor (N, 3, 32, 32), float32 / 255
          ORL    -> (112, 92, 3)  -> tensor (N, 1, 112, 92), R-channel, float32 / 255
        """
        X_list, y_list = [], []
        name = self.dataset_name.upper()

        for item in instances:
            img = np.array(item['image'], dtype=np.float32) / 255.0
            label = int(item['label'])

            if name == 'MNIST':
                # shape (28, 28) -> (1, 28, 28)
                img = img[np.newaxis, :, :]
            elif name == 'CIFAR':
                # shape (32, 32, 3) -> (3, 32, 32)
                img = np.transpose(img, (2, 0, 1))
            elif name == 'ORL':
                # shape (112, 92, 3) — all channels identical (grayscale)
                # use only the R channel -> (1, 112, 92)
                img = img[:, :, 0][np.newaxis, :, :]

            X_list.append(img)
            y_list.append(label)

        X_tensor = torch.FloatTensor(np.array(X_list))
        y_tensor = torch.LongTensor(y_list)

        # ORL labels are 1-indexed; shift to 0-indexed for CrossEntropyLoss
        if name == 'ORL':
            y_tensor = y_tensor - 1

        return X_tensor, y_tensor

    # ---------- fit (renamed from train to avoid nn.Module.train conflict) ----------

    def fit(self, train_instances):
        X_tensor, y_tensor = self._preprocess(train_instances)
        n_samples = len(X_tensor)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        self.loss_history = []

        for epoch in range(self.max_epoch):
            # Shuffle each epoch
            perm = torch.randperm(n_samples)
            X_tensor = X_tensor[perm]
            y_tensor = y_tensor[perm]

            self.train()  # set to training mode (nn.Module.train)
            epoch_losses = []
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_tensor[i: i + self.batch_size]
                y_batch = y_tensor[i: i + self.batch_size]

                y_pred = self.forward(X_batch)
                loss = loss_function(y_pred, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())

            avg_loss = float(np.mean(epoch_losses))
            self.loss_history.append(avg_loss)
            scheduler.step(avg_loss)

            # Evaluate on full training set
            self.eval()
            with torch.no_grad():
                y_pred_full = self.forward(X_tensor)
            self.train()  # back to training mode

            accuracy_evaluator.data = {
                'true_y': y_tensor,
                'pred_y': y_pred_full.max(1)[1],
            }
            metrics = accuracy_evaluator.evaluate()
            print(f'Epoch: {epoch:>3}  '
                  f'Accuracy: {metrics["accuracy"]:.4f}  '
                  f'Loss: {avg_loss:.4f}')

        self.save_loss_plot()

    # ---------- predict (renamed from test to keep naming consistent) ----------

    def predict(self, test_instances):
        X_tensor, y_tensor = self._preprocess(test_instances)
        self.eval()
        with torch.no_grad():
            y_pred = self.forward(X_tensor).max(1)[1]
        return y_pred, y_tensor

    # ---------- loss plot ----------

    def save_loss_plot(self,
                       output_path='../../result/stage_3_result/training_loss_convergence.png'):
        """Plot and save the training loss curve recorded during training."""
        epochs = np.arange(1, len(self.loss_history) + 1)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(epochs, self.loss_history,
                color='#378ADD', linewidth=1.8, linestyle='-')

        ax.set_xlabel('Training Epoch', fontsize=12)
        ax.set_ylabel('Cross-Entropy Loss', fontsize=12)
        ax.set_title(f'CNN Training Loss Convergence ({self.dataset_name})\n'
                     r'Adam optimizer, $\alpha$=1e-3, CrossEntropyLoss',
                     fontsize=13, pad=12)

        ax.set_xlim(1, len(self.loss_history))
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
        ax.spines[['top', 'right']].set_visible(False)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'Loss plot saved to: {output_path}')
        plt.close()

    # ---------- run ----------

    def run(self, trainData=None, trainLabel=None, testData=None):
        print('method running...')
        print(f'-- dataset: {self.dataset_name}')
        print('-- start training...')
        self.fit(self.data['train'])

        print('-- start testing...')
        pred_y, true_y = self.predict(self.data['test'])

        return {'pred_y': pred_y, 'true_y': true_y}