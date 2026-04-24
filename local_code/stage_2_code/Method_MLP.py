'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class Method_MLP(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 10
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3
    # mini-batch size: much faster than full-batch and helps the model generalize better
    batch_size = 256

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # deeper network: 784 -> 512 -> 256 -> 10
        # BatchNorm stabilizes training; Dropout prevents overfitting
        self.network = nn.Sequential(
            nn.Linear(784, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 10),
            # NOTE: no Softmax here — nn.CrossEntropyLoss applies it internally
        )

    def forward(self, x):
        '''Forward propagation'''
        return self.network(x)

    def train(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # ReduceLROnPlateau lowers the learning rate if loss stops improving
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        X_tensor = torch.FloatTensor(np.array(X))
        y_tensor = torch.LongTensor(np.array(y))
        n_samples = len(X_tensor)

        self.loss_history = []

        for epoch in range(self.max_epoch):
            # shuffle data each epoch
            perm = torch.randperm(n_samples)
            X_tensor = X_tensor[perm]
            y_tensor = y_tensor[perm]

            epoch_losses = []
            # mini-batch training
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_tensor[i: i + self.batch_size]
                y_batch = y_tensor[i: i + self.batch_size]

                y_pred = self.forward(X_batch)
                loss = loss_function(y_pred, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())

            # average loss across all mini-batches this epoch
            avg_loss = np.mean(epoch_losses)
            self.loss_history.append(avg_loss)
            scheduler.step(avg_loss)

            # evaluate on full training set every epoch
            self.network.eval()
            with torch.no_grad():
                y_pred_full = self.forward(X_tensor)
            self.network.train()

            accuracy_evaluator.data = {'true_y': y_tensor, 'pred_y': y_pred_full.max(1)[1]}
            metrics = accuracy_evaluator.evaluate()
            print(f'Epoch: {epoch}  Accuracy: {metrics["accuracy"]:.4f}  Loss: {avg_loss:.4f}')

        self.save_loss_plot()

    def save_loss_plot(self, output_path='training_loss_convergence.png'):
        """Plot and save the training loss curve recorded during training."""
        epochs = np.arange(1, len(self.loss_history) + 1)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(epochs, self.loss_history, color='#378ADD', linewidth=1.8, linestyle='-')

        ax.set_xlabel('Training Epoch', fontsize=12)
        ax.set_ylabel('Cross-Entropy Loss', fontsize=12)
        ax.set_title('MLP Training Loss Convergence\n'
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

    def test(self, X):
        self.network.eval()
        with torch.no_grad():
            y_pred = self.forward(torch.FloatTensor(np.array(X)))
        return y_pred.max(1)[1]

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}