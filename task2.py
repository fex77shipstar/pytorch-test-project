from pickletools import optimize
from pyexpat import model
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('task2.csv')
x = torch.tensor(df.iloc[:, 0].values.reshape(-1, 1), dtype=torch.float32)
y = torch.tensor(df.iloc[:, 1].values.reshape(-1, 1), dtype=torch.float32)


class MLP(nn.Module):
    def __init__(self,input_size=1,hidden_size=64,output_size=1):
        super(MLP,self).__init__()
        self.layers=nn.Sequential(
            nn.Linear(input_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,output_size),
        )
        
    def forward(self,x):
        return self.layers(x)
        
model=MLP(input_size=1,hidden_size=64,output_size=1)
criterion=nn.MSELoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)

       
epochs = 1000
save_epochs = [10, 100, 1000]  
model_states = {}  
for epoch in range(1, epochs+1):
    model.train()
    optimizer.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()

    
    if epoch in save_epochs:
        model_states[epoch] = model.state_dict()
        print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

plt.figure(figsize=(10,6))


sorted_idx = x.numpy().argsort(axis=0).flatten()
sorted_x = x.numpy()[sorted_idx]

colors = ['orange', 'green', 'red']
line_styles = ['--', '-.', '-']
for idx, epoch in enumerate(save_epochs):
    model.load_state_dict(model_states[epoch])
    model.eval()
    with torch.no_grad():
        y_pred = model(x).numpy()
    sorted_y_pred = y_pred[sorted_idx]
    plt.plot(sorted_x, sorted_y_pred, 
             color=colors[idx], 
             linestyle=line_styles[idx],
             label=f'Epoch {epoch}',
             linewidth=2)


plt.scatter(x.numpy(), y.numpy(), 
            label='原始数据', 
            s=10,  
            alpha=0.8,  
            color='blue',
            zorder=3) 

plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='best')
plt.title('MLP不同训练轮次拟合效效果')
plt.show()
