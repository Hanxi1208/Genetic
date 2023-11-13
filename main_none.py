import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import torch.nn as nn
import random
import math
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont
import os

SEED = 4321
torch.manual_seed(SEED)
random.seed(SEED)

class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size1,
        hidden_size2,
        hidden_size3,
        hidden_size4,
        hidden_size5,
        hidden_size6,
        hidden_size7,
        hidden_size8,
        hidden_size9,
        output_size,
    ):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)  
        self.relu1 = nn.ReLU()
        # self.relu1 = nn.LeakyReLU()
        # self.dropout1 = nn.Dropout(0.5)  

        self.fc2 = nn.Linear(hidden_size1, hidden_size2)  
        self.relu2 = nn.ReLU()
        # self.relu2 = nn.LeakyReLU()
        # self.dropout2 = nn.Dropout(0.5)  

        self.fc3 = nn.Linear(hidden_size2, hidden_size3)  
        self.relu3 = nn.ReLU()
        # self.relu3 = nn.LeakyReLU()
        # self.dropout3 = nn.Dropout(0.5)  

        self.fc4 = nn.Linear(hidden_size3, hidden_size4) 
        self.relu4 = nn.ReLU()
        # self.relu4 = nn.LeakyReLU()
        # self.dropout4 = nn.Dropout(0.5)  

        self.fc5 = nn.Linear(hidden_size4, hidden_size5)  
        self.relu5 = nn.ReLU()
        # self.relu5 = nn.LeakyReLU()
        # self.dropout5 = nn.Dropout(0.5)  

        self.fc6 = nn.Linear(hidden_size5, hidden_size6)  
        self.relu6 = nn.ReLU()
        # self.relu6 = nn.LeakyReLU()

        self.fc7 = nn.Linear(hidden_size6, hidden_size7)
        self.relu7 = nn.ReLU()
        # self.relu7 = nn.LeakyReLU()


        self.fc8 = nn.Linear(hidden_size7, hidden_size8)
        self.relu8 = nn.ReLU()
        # self.relu8 = nn.LeakyReLU()

        self.fc9 = nn.Linear(hidden_size8, hidden_size9)
        self.relu9 = nn.ReLU()
        # self.relu9 = nn.LeakyReLU()

        # self.fc6= nn.Linear(hidden_size5, output_size)
        # self.fc7= nn.Linear(hidden_size6, output_size)
        # self.fc8= nn.Linear(hidden_size7, output_size)
        # self.fc9= nn.Linear(hidden_size8, output_size)
        self.fc10= nn.Linear(hidden_size9, output_size)  

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        # out = self.dropout1(out)

        out = self.fc2(out)
        out = self.relu2(out)
        # out = self.dropout2(out)

        out = self.fc3(out)
        out = self.relu3(out)
        # out = self.dropout3(out)

        out = self.fc4(out)
        out = self.relu4(out)
        # out = self.dropout4(out)

        out = self.fc5(out)
        out = self.relu5(out)
        # out = self.dropout5(out)

        out = self.fc6(out)
        out = self.relu6(out)

        out = self.fc7(out)
        out = self.relu7(out)

        out = self.fc8(out)
        out = self.relu8(out)

        out = self.fc9(out)
        out = self.relu9(out)

        # out = self.fc6(out)
        # out = self.fc7(out)
        # out = self.fc8(out)
        # out = self.fc9(out)
        out = self.fc10(out)
        return out


def main():
    df = pd.read_csv("testpara.csv")
    df = df.to_numpy()

    df = np.delete(df, 6, axis=1)

    X = df[:, np.arange(df.shape[1]) != 6]
    y = df[:, 6]

    # split into train, validation and test set (70%, 15%, 15%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=20
    )  # 15% validation set, 15% test set
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=30
    )  # 15% validation set, 15% test set

    scaler1 = MinMaxScaler()
    X_train = scaler1.fit_transform(X_train)
    X_val = scaler1.transform(X_val)
    X_test = scaler1.transform(X_test)

    scaler2 = MinMaxScaler()
    # scaler2 = StandardScaler()
    y_train = scaler2.fit_transform(y_train.reshape(-1, 1))
    y_val = scaler2.transform(y_val.reshape(-1, 1))
    y_test = scaler2.transform(y_test.reshape(-1, 1))

    input_size = 6
    hidden_size1 = 64
    hidden_size2 = 64
    hidden_size3 = 128
    hidden_size4 = 128
    hidden_size5 = 128
    hidden_size6 = 128
    hidden_size7 = 128
    hidden_size8 = 64
    hidden_size9 = 32
    output_size = 1
    learningrate = 5e-4

    MinTrainLoss = 1e10  # 保存最小的训练误差对应的模型参数

    model = MLP(
        input_size,
        hidden_size1,
        hidden_size2,
        hidden_size3,
        hidden_size4,
        hidden_size5,
        hidden_size6,
        hidden_size7,
        hidden_size8,
        hidden_size9,
        output_size,
    )
    print(model)

    nn.L1Loss
    criterion = nn.L1Loss()  
    # criterion = nn.MSELoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=learningrate)  

    # transfer numpy array to PyTorch tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    val_loss1 = []
    train_loss = []
    
    num_epochs = 400
    for epoch in range(num_epochs):
        model.train()  # model.train()

        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
      
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()  
    
        train_loss.append(loss.item())

        with torch.no_grad():
            model.eval() 
            val_outputs = model(X_val_tensor)

            val_loss = criterion(val_outputs, y_val_tensor)
            # val_mse = np.mean((val_outputs - y_val_tensor) ** 2)

            y_test = val_outputs.detach().cpu().numpy()

            val_loss1.append(val_loss.item())

        if val_loss1[-1] < MinTrainLoss:
            torch.save(model.state_dict(), "model.pth")  
            MinTrainLoss = val_loss1[-1]
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Val Loss: {val_loss.item():.4f}")

    # plt_loss = loss.detach().cpu().numpy()
    plt.plot(range(num_epochs), val_loss1, label="Validation Loss")
    plt.plot(range(num_epochs), train_loss, label="Train Loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()

    # filename_png = f"none_ReLu_{input_size}_{hidden_size1}_{hidden_size2}_{hidden_size3}_{hidden_size4}_{hidden_size5}.png"
    # filename_png = f"none_ReLu_{input_size}_{hidden_size1}_{hidden_size2}_{hidden_size3}_{hidden_size4}_{hidden_size5}_{hidden_size6}.png"
    # filename_png = f"none_ReLu_{input_size}_{hidden_size1}_{hidden_size2}_{hidden_size3}_{hidden_size4}_{hidden_size5}_{hidden_size6}_{hidden_size7}.png"
    # filename_png = f"none_ReLu_{input_size}_{hidden_size1}_{hidden_size2}_{hidden_size3}_{hidden_size4}_{hidden_size5}_{hidden_size6}_{hidden_size7}_{hidden_size8}.png"
    filename_png = f"none_ReLu_{input_size}_{hidden_size1}_{hidden_size2}_{hidden_size3}_{hidden_size4}_{hidden_size5}_{hidden_size6}_{hidden_size7}_{hidden_size8}_{hidden_size9}.png"
    folder_path = "None"
    os.makedirs(folder_path, exist_ok=True)
    plt.savefig(os.path.join(folder_path, filename_png), bbox_inches="tight")

    min_train_loss = np.min(train_loss)
    min_val_loss = np.min(val_loss1)

    
    filename_txt = f"loss.txt"
    folder_path = "None"
    os.makedirs(folder_path, exist_ok=True)

    file_path = os.path.join(folder_path, filename_txt)
    with open(file_path, 'a+') as f:
        f.write(f"none_ReLu_{input_size}_{hidden_size1}_{hidden_size2}_{hidden_size3}_{hidden_size4}_{hidden_size5}\n")
        # f.write(f"none_ReLu_{input_size}_{hidden_size1}_{hidden_size2}_{hidden_size3}_{hidden_size4}_{hidden_size5}_{hidden_size6}\n")
        # f.write(f"none_ReLu_{input_size}_{hidden_size1}_{hidden_size2}_{hidden_size3}_{hidden_size4}_{hidden_size5}_{hidden_size6}_{hidden_size7}\n")
        # f.write(f"none_ReLu_{input_size}_{hidden_size1}_{hidden_size2}_{hidden_size3}_{hidden_size4}_{hidden_size5}_{hidden_size6}_{hidden_size7}_{hidden_size8}\n")
        f.write(f"none_LeakyReLu_{input_size}_{hidden_size1}_{hidden_size2}_{hidden_size3}_{hidden_size4}_{hidden_size5}_{hidden_size6}_{hidden_size7}_{hidden_size8}_{hidden_size9}\n")
        f.write(f"min_train_loss:{min_train_loss:.4f}\nmin_val_loss:{min_val_loss:.4f}")
        f.write('\n')


    # text = f"min_train_loss:{min_train_loss:.4f}\nmin_val_loss:{min_val_loss:.4f}"
    # plt.text(0.5, -0.07, text, horizontalalignment='left', verticalalignment='center', color='red')

    # file_path = os.path.join(folder_path, filename_png)
    # plt.savefig(os.path.join(folder_path, filename_png), bbox_inches="tight")
   
if __name__ == "__main__":
    main()
