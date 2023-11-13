import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import torch.nn as nn
import random
from main import MLP
import math
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


import plotly.express as px
import plotly.graph_objects as go
import os


SEED = 4321
torch.manual_seed(SEED)
random.seed(SEED)

def positional_encoding(X, L):
    """
    X: input data features
    L: number of encoding layers
    """
    X_min = np.min(X)
    X_max = np.max(X)
    x = 2 * ((X - X_min) / (X_max - X_min)) - 1
    temp = []
    for l in range(L):
        value = 2**l
        sinx = np.sin(value * np.pi * x)
        cosx = np.cos(value * np.pi * x)
        temp.extend([sinx, cosx])
    temp = np.array(temp)
    temp = temp.T
    return temp


df = pd.read_csv("testpara.csv")
df = df.to_numpy()

df = np.delete(df, 6, axis=1)
X = df[:, np.arange(df.shape[1]) != 6]
y = df[:, 6]

# InnerRadius_index = 0
# Length_index = 1
# OpeningAngle_index = 2
# InnerRadius2_index = 3
# Length2_index = 3
# OpeningAngle2_index = 2

# X_InnerRadius = X[:, InnerRadius_index]
# X_Length = X[:, 1]
# X_OpeningAngle = X[:, 2]
# X_InnerRadius2 = X[:, 3]
# X_length2 = X[:, 4]
# X_OpeningAngle2 = X[:, 2]

# InnerRadius_pe = positional_encoding(X_InnerRadius, InnerRadius_index)
# Length_pe = positional_encoding(X_Length, Length_index)
# OpeningAngle_pe = positional_encoding(X_OpeningAngle, OpeningAngle_index)
# InnerRadius2_pe = positional_encoding(X_InnerRadius2, InnerRadius2_index)
# Length2_pe = positional_encoding(X_length2, Length2_index)
# OpeningAngle2_pe = positional_encoding(X_OpeningAngle2, OpeningAngle2_index)


# X = np.concatenate((X, Length2_pe), axis=1)
# X = np.concatenate((X, OpeningAngle_pe), axis=1)
# X = np.concatenate((X, OpeningAngle2_pe), axis=1)

X_InnerRadius = X[:, 0]
X_Length = X[:, 1]
X_OpeningAngle = X[:, 2]
X_InnerRadius2 = X[:, 3]
X_length2 = X[:, 4]
X_OpeningAngle2 = X[:, 5]

# InnerRadius_index = 1
Length_index = 2
OpeningAngle_index = 2
# InnerRadius2_index = 1
Length2_index = 3
OpeningAngle2_index = 2

# InnerRadius_pe = positional_encoding(X_InnerRadius, InnerRadius_index)
Length_pe = positional_encoding(X_Length, Length_index)
OpeningAngle_pe = positional_encoding(X_OpeningAngle, OpeningAngle_index)
# InnerRadius2_pe = positional_encoding(X_InnerRadius2, InnerRadius2_index)
Length2_pe = positional_encoding(X_length2, Length2_index)
OpeningAngle2_pe = positional_encoding(X_OpeningAngle2, OpeningAngle2_index)

# X = np.concatenate((X, InnerRadius_pe), axis=1)
# X = np.concatenate((X, InnerRadius2_pe), axis=1)
# X = np.concatenate((X, Length_pe), axis=1)
X = np.concatenate((X, Length2_pe), axis=1)
X = np.concatenate((X, OpeningAngle_pe), axis=1)
X = np.concatenate((X, OpeningAngle2_pe), axis=1)

# split into train, validation and test set (70%, 15%, 15%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=20
)  # 70% train set

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=30
)  # 15% validation set, 15% test set

# print("y_test",y_test)
target1 = y_test
target = torch.tensor(y_test, dtype=torch.float32)

scaler1 = MinMaxScaler()  # zscore标准化
X_train = scaler1.fit_transform(X_train)
X_val = scaler1.transform(X_val)
X_test = scaler1.transform(X_test)

scaler2 = MinMaxScaler()  # zscore标准化
# scaler2 = StandardScaler()
y_train = scaler2.fit_transform(y_train.reshape(-1,1))
y_val = scaler2.transform(y_val.reshape(-1,1))
y_test = scaler2.transform(y_test.reshape(-1,1))
# print("y_test",y_test)

# 转换NumPy数组为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
# X_train_tensor = X_train_tensor + pos_encoding_train
# y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
# X_val_tensor = X_val_tensor + pos_encoding_val
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
# X_test_tensor = X_test_tensor + pos_encoding_test
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# print("y_test",y_test_tensor)

#  创建MLP模型实例
# input_size = 6
# hidden_size1 = 64
# hidden_size2 = 64
# hidden_size3 = 128
# hidden_size4 = 128
# hidden_size5 = 32
# num_classes = 1

input_size = X.shape[1]
hidden_size1 = 32
hidden_size2 = 64
hidden_size3 = 128
hidden_size4 = 64
hidden_size5 = 32
hidden_size6 = 128
hidden_size7 = 128
hidden_size8 = 64
hidden_size9 = 32
output_size = 1


device = torch.device("cpu")
Mlp = MLP(
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
# Mlp = MLP(input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, num_classes)
# Mlp = MLP(input_size, hidden_size1, hidden_size2, num_classes)
# Mlp = MLP(input_size, hidden_size1, hidden_size2, hidden_size3, num_classes)
Mlp.load_state_dict(torch.load("model.pth", map_location=device))  # pytoch 导入模型

Mlp.eval()
with torch.no_grad():
    criterion = nn.L1Loss()  
    test_outputs = Mlp(X_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor)
    y_output = scaler2.inverse_transform(test_outputs)
    predicted1 = y_output
    predicted = y_output.flatten()
    # print("predicted",predicted)
    # print("target",target)
    test_mse = np.mean((predicted1- target1) ** 2)
    # print("test_mse",test_mse)


    x = np.linspace(3.8, 5, 100)
    y = x
    fig = go.Figure(data=go.Scatter(x=target, y=predicted, mode='markers', name = 'test data'))
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='y=x'))

    xlabel = "groundtruth"
    ylabel = "output"
    fig.update_layout(
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        font=dict(
            size=18,
            color="black"
        )
    )

    # fig.update_layout(title=f'MSE loss:{test_mse}', xaxis_title='groundtruth', yaxis_title='output')
    # fig.add_annotation(text={input_size,hidden_size1,hidden_size2,hidden_size3,hidden_size4,hidden_size4,num_classes}, x=4, y=4, showarrow=True, arrowhead=1)
    fig.write_image("scatter_plot_none.png")
    
    image = Image.open("scatter_plot_none.png")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", 20)  # 字体和大小可以根据需要调整
    # draw.text((20, 60), f"Network paras: {input_size},{hidden_size1},{hidden_size2},{hidden_size3},{hidden_size4},{hidden_size5},{hidden_size6},{output_size}", fill="red", font=font)
    # draw.text((20, 60), f"Network paras: {input_size},{hidden_size1},{hidden_size2},{hidden_size3},{hidden_size4},{hidden_size5},{hidden_size6},{hidden_size7},{output_size}", fill="red", font=font)
    # draw.text((20, 60), f"Network paras: {input_size},{hidden_size1},{hidden_size2},{hidden_size3},{hidden_size4},{hidden_size5},{hidden_size6},{hidden_size7},{hidden_size8},{output_size}", fill="red", font=font)
    # draw.text((20, 60), f"Network paras: {input_size},{hidden_size1},{hidden_size2},{hidden_size3},{hidden_size4},{hidden_size5},{hidden_size6},{hidden_size7},{hidden_size8},{hidden_size9},{output_size}", fill="red", font=font)
    draw.text((20, 60), f"Network paras: {input_size},{hidden_size1},{hidden_size2},{hidden_size3},{hidden_size4},{hidden_size5},{output_size}", fill="red", font=font)
    draw.text((20, 30), f"MSE loss: {test_mse}", fill="red", font=font)
    # draw.text((20, 0), f"l1 loss: {test_loss}", fill="red", font=font)
    
    # filename = f"none_test_LeakyReLu_{input_size}_{hidden_size1}_{hidden_size2}_{hidden_size3}_{hidden_size4}_{hidden_size5}_{hidden_size6}_{hidden_size7}_{hidden_size8}_{hidden_size9}.png"
    # filename = f"none_test_ReLu_{input_size}_{hidden_size1}_{hidden_size2}_{hidden_size3}_{hidden_size4}_{hidden_size5}_{hidden_size6}_{hidden_size7}_{hidden_size8}.png"
    # filename = f"none_test_ReLu_{input_size}_{hidden_size1}_{hidden_size2}_{hidden_size3}_{hidden_size4}_{hidden_size5}_{hidden_size6}_{hidden_size7}.png"
    # filename = f"none_test_ReLu_{input_size}_{hidden_size1}_{hidden_size2}_{hidden_size3}_{hidden_size4}_{hidden_size5}_{hidden_size6}.png"
    filename = f"none_test_ReLu_{input_size}_{hidden_size1}_{hidden_size2}_{hidden_size3}_{hidden_size4}_{hidden_size5}.png"
    folder_path = "PE"
    os.makedirs(folder_path, exist_ok=True)

    image.save(os.path.join(folder_path, filename))

    filename_txt = f"loss.txt"
    folder_path = "PE"
    os.makedirs(folder_path, exist_ok=True)

    file_path = os.path.join(folder_path, filename_txt)
    with open(file_path, 'a+') as f:
        f.write(f"none_ReLu_{input_size}_{hidden_size1}_{hidden_size2}_{hidden_size3}_{hidden_size4}_{hidden_size5}\n")
        # f.write(f"none_ReLu_{input_size}_{hidden_size1}_{hidden_size2}_{hidden_size3}_{hidden_size4}_{hidden_size5}_{hidden_size6}\n")
        # f.write(f"none_ReLu_{input_size}_{hidden_size1}_{hidden_size2}_{hidden_size3}_{hidden_size4}_{hidden_size5}_{hidden_size6}_{hidden_size7}\n")
        # f.write(f"none_ReLu_{input_size}_{hidden_size1}_{hidden_size2}_{hidden_size3}_{hidden_size4}_{hidden_size5}_{hidden_size6}_{hidden_size7}_{hidden_size8}\n")
        # f.write(f"none_ReLu_{input_size}_{hidden_size1}_{hidden_size2}_{hidden_size3}_{hidden_size4}_{hidden_size5}_{hidden_size6}_{hidden_size7}_{hidden_size8}_{hidden_size9}\n")
        f.write(f"test_loss:{test_loss:.4f}")
        f.write('\n')






    
