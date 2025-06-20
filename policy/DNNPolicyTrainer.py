import torch
import torch.nn as nn
import torch.optim
from copy import deepcopy
import math

from torch.utils.data import Dataset, DataLoader

class DNNPolicyTrainer:

    def __init__(self, configs):
        self.configs = configs
        self.model_configs = configs['model_cfgs']
        self.dnn_train_configs = self.configs['dnn_train_cfgs']
        self._intitalize_model()

    def _intitalize_model(self):
        # setup DNN architecture
        if self.model_configs['model'] == 'dnn-mlp':
            self.model = MLPPolicy(self.model_configs['input_dim'], self.model_configs['output_dim'], self.model_configs['hidden_dims'], self.model_configs['activation'])
        else:
            raise ValueError('Invalid model type')
        # setup device
        self.device = self.model_configs['device']
        
    def _initialize_train(self):
        # setup optimizer
        if self.dnn_train_configs['optimizer'] == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.dnn_train_configs['learning_rate'])
        elif self.dnn_train_configs['optimizer'] == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.dnn_train_configs['learning_rate'], momentum=self.dnn_train_configs['momentum'])
        else:
            raise ValueError('Invalid optimizer')
        
        # setup training parameters
        self.batch_size = self.dnn_train_configs['batch_size']
        self.epochs = self.dnn_train_configs['epochs']

    def _compute_loss(self, y_pred, y_true):
        if self.dnn_train_configs['loss'] == 'mse':
            return nn.MSELoss()(y_pred, y_true)
        elif self.dnn_train_configs['loss'] == 'l1':
            return nn.L1Loss()(y_pred, y_true)
        elif self.dnn_train_configs['loss'] == 'corrcoef':
            _m = torch.cat([y_pred.reshape(1, -1), y_true.reshape(1, -1)], dim=0)
            return torch.mean(torch.abs(1.0 - torch.corrcoef(_m)[0, 1]))
        elif self.dnn_train_configs['loss'] == 'corrcoef+mse':
            corrcoef_loss_w, mse_loss_w = self.dnn_train_configs['loss_weights']['corrcoef'], self.dnn_train_configs['loss_weights']['mse']
            _m = torch.cat([y_pred.reshape(1, -1), y_true.reshape(1, -1)], dim=0)
            return (corrcoef_loss_w*torch.mean(torch.abs(1.0 - torch.corrcoef(_m)[0, 1]))) + (mse_loss_w*nn.MSELoss()(y_pred, y_true))
        elif self.dnn_train_configs['loss'] == 'corrcoef+l1':
            corrcoef_loss_w, l1_loss_w = self.dnn_train_configs['loss_weights']['corrcoef'], self.dnn_train_configs['loss_weights']['l1']
            _m = torch.cat([y_pred.reshape(1, -1), y_true.reshape(1, -1)], dim=0)
            return (corrcoef_loss_w*torch.mean(torch.abs(1.0 - torch.corrcoef(_m)[0, 1]))) + (l1_loss_w*nn.L1Loss()(y_pred, y_true))
        else:
            raise ValueError('Invalid loss function')

    def _train(self):
        best_model = deepcopy(self.model)
        best_loss = math.inf
        best_model_epoch = 0
        self.model = self.model.to(self.device)
        for epoch in range(self.epochs):
            # Train
            all_train_loss = []
            for x_batch, y_batch in self.train_dataloader:
                x_batch = x_batch.float().to(self.device)
                y_batch = y_batch.float().to(self.device)
                y_pred = self.model(x_batch)
                loss = self._compute_loss(y_pred, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                all_train_loss.append(loss.cpu().detach().item())
            avg_train_loss = sum(all_train_loss)/len(all_train_loss)
            all_val_loss = []
            # Val
            for x_batch, y_batch in self.val_dataloader:
                x_batch = x_batch.float().to(self.device)
                y_batch = y_batch.float().to(self.device)
                with torch.no_grad():
                    y_pred = self.model(x_batch)
                loss = self._compute_loss(y_pred, y_batch)
                all_val_loss.append(loss.cpu().detach().item())
            avg_val_loss = sum(all_val_loss)/len(all_val_loss)
            print('Step: {} | Train loss: {:.4f} | Val loss: {:.4f}'.format(epoch, avg_train_loss, avg_val_loss), end='\r')
            # Update model
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_model = deepcopy(self.model)
                best_model_epoch = epoch
        self.model = best_model.to('cpu')
        print(f"[Decision DNN Offline Training] Best Val loss: {best_loss:.4f}")
        return best_loss
    
    def _adapt(self, train_dataloader, val_dataloader, epochs, batch_size, early_stop_epochs=None):
        best_model = deepcopy(self.model)
        best_loss = math.inf
        best_model_epoch = 0
        self.model = self.model.to(self.device)
        early_stop_cnt = 0
        for epoch in range(epochs):
            # Train
            all_train_loss = []
            for x_batch, y_batch in train_dataloader:
                # Avoid batch size too small: lead to NaN loss after calculating corrcoef loss
                if x_batch.shape[0] < batch_size:
                    continue
                x_batch = x_batch.float().to(self.device)
                y_batch = y_batch.float().to(self.device)
                y_pred = self.model(x_batch)
                loss = self._compute_loss(y_pred, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                all_train_loss.append(loss.cpu().detach().item())
            avg_train_loss = sum(all_train_loss)/len(all_train_loss) if len(all_train_loss) > 0 else 0
            all_val_loss = []
            # Val
            for x_batch, y_batch in val_dataloader:
                if x_batch.shape[0] < batch_size:
                    continue
                x_batch = x_batch.float().to(self.device)
                y_batch = y_batch.float().to(self.device)
                with torch.no_grad():
                    y_pred = self.model(x_batch)
                loss = self._compute_loss(y_pred, y_batch)
                all_val_loss.append(loss.cpu().detach().item())
            avg_val_loss = sum(all_val_loss)/len(all_val_loss)
            # Update model
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_model = deepcopy(self.model)
                best_model_epoch = epoch
                early_stop_cnt = 0
            else:
                early_stop_cnt += 1
                if early_stop_epochs is not None and early_stop_cnt >= early_stop_epochs:
                    break
        self.model = best_model
        print(f"[Decision DNN Info] Decision DNN updated")
        return best_loss

    ############################################################################################################
    # Public methods
    ############################################################################################################
    def save_model(self, model_path):
        return

    def load_model(self, model_path):
        return

    def fit(self, x_train, y_train, x_val, y_val):
        # offline training the model
        # training setups
        self._initialize_train()
        # dataloader
        self.train_dataloader = DataLoader(OnlineStatsDataset(x_train, y_train), self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(OnlineStatsDataset(x_val, y_val), self.batch_size, shuffle=False)
        best_loss = self._train()
        return best_loss

    def predict(self, x_stats):
        with torch.no_grad():
            x_stats_t = torch.tensor(x_stats).float().to(self.device)
            self.model = self.model.to(self.device)
            pred = self.model(x_stats_t).cpu().detach().numpy()
        return pred
    
    def adapt(self, x_train, y_train, x_val, y_val, batch_size, epochs, early_stop_epochs=None):
        # online adaptation: training setups
        self._initialize_train()
        # dataloader
        train_dataloader = DataLoader(OnlineStatsDataset(x_train, y_train), batch_size, shuffle=True)
        val_dataloader = DataLoader(OnlineStatsDataset(x_val, y_val), batch_size, shuffle=False)
        best_loss = self._adapt(train_dataloader, val_dataloader, epochs, batch_size, early_stop_epochs)
        return best_loss
    
class OnlineStatsDataset(Dataset):
    def __init__(self, x, y):
        assert x.shape[0] == y.shape[0]
        self.x = x # (n_samples, n_features)
        self.y = y # (n_samples, n_targets)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
class MLPPolicy(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dims, activation='relu'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.layers = self._build_layers()

    def _get_activation(self):
        if self.activation == 'relu':
            return nn.ReLU()
        elif self.activation == 'tanh':
            return nn.Tanh()
        elif self.activation == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError('Invalid activation function')

    def _build_layers(self):
        layers = []
        prev_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim, bias=True))
            layers.append(self._get_activation())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, self.output_dim))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)