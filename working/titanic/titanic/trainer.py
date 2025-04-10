import joblib
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, TensorDataset

from .dataset import get_input_data, save_as_csv


class trainer:
    def __init__(
            self, 
            model_class, 
            epochs: int, 
            batch_size: int, 
            lr: float, 
            train_file: str,
            save_dir: str
        ):
        self.model_class = model_class
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.train_file = train_file
        self.save_dir = save_dir

        self.model_save_path = os.path.join(save_dir, "mlp_model.pth")
        self.imputer_save_path = os.path.join(save_dir, "imputer.pkl")
        self.scaler_save_path = os.path.join(save_dir, "scaler.pkl")

        self.model = None
        self.device = None
        self.imputer = None
        self.scaler = None
        self.input_size = None
        self.train_loader = None
        self.test_loader = None
        self.criterion = None
        self.optimizer = None
    
    def set_device(self, device=None):
        self.device = device if device is not None else (
                      torch.device("cuda" if torch.cuda.is_available() else "cpu") )
        if not torch.cuda.is_available():
            self.device = "cpu"
        print(f"Using device: {self.device}")
    
    def get_input_size(self, X: np.array):
        self.input_size = X.shape[1]
        return self.input_size

    def process_dataset(self):
        X, y, self.imputer, self.scaler = get_input_data(os.path.basename(self.train_file),
                                                         is_train=True)
        self.get_input_size(X)

        # 3. 分割数据集并转换为PyTorch张量
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
    
    def init_model(self):
        self.model = self.model_class(self.input_size).to(self.device)
        print(f"Set model to {self.device}")

        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def _train_epoch(self, epoch: int):
        total_loss = 0
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # 前向传播
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels.unsqueeze(1).float())
            
            # 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {total_loss/len(self.train_loader):.4f}')

    def save_model(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        # 保存模型参数
        torch.save(self.model.state_dict(), self.model_save_path)
        
        # 保存预处理工具
        joblib.dump(self.imputer, self.imputer_save_path)
        joblib.dump(self.scaler, self.scaler_save_path)
        
        print(f"Model and preprocessing tools saved to {self.save_dir}\n")
    
    def load_tools(self):
        # 加载预处理工具
        self.imputer = joblib.load(self.imputer_save_path)
        self.scaler = joblib.load(self.scaler_save_path)

    def load_model(self):
        # 初始化模型
        self.model = self.model_class(self.input_size)
        self.model.load_state_dict(torch.load(self.model_save_path))
        
        self.model = self.model.to(self.device)
        print(f"Set model to {self.device}")
    
    def train(self):
        self.set_device()
        self.process_dataset()
        self.init_model()
        
        print("\nTraining model...")
        self.model.train()
        for epoch in range(self.epochs):
            self._train_epoch(epoch)
        
        self.save_model()
    
    def evaluate(self):
        print("Model Evaluation on test set:")

        self.model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                predicted = (outputs > 0.5).float()
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy().flatten())
        
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
    
    def test(self, test_file: str, output_file_path: str):
        print("Loading model and making predictions...\n")

        self.set_device()

        self.load_tools()
        ID, X, _, _ = get_input_data(os.path.basename(test_file), is_train=False, 
                                     imputer=self.imputer, scaler=self.scaler)
        self.get_input_size(X)
        self.load_model()
        
        self.model.eval()
        X = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            outputs = self.model(X)
            probabilities = outputs.cpu().numpy().flatten()
            predictions = (probabilities > 0.5).astype(int)
        
        predictions = np.column_stack((ID, predictions))
        #print(predictions)

        save_as_csv(predictions, output_file_path)