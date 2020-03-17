from model.model import *
from data.dataset import *
from torch.utils.data import Dataset, DataLoader 
from matplotlib import pyplot as plt 

data_root = "./data/dataset/"
trainSet = EnergyData(root_dir=data_root, history_len=50, forecast_len=1, interval=2)
trainLoader = DataLoader(dataset=trainSet, batch_size=16,shuffle=True)
testLoader = DataLoader(dataset=trainSet, batch_size=16,shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TestNet(name='TSP_1',seq_len=50).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
loss_fn = nn.MSELoss()

model.train()
for epoch in range(10):
    for i, data in enumerate(trainLoader):
        x, y = data
        x = x.unsqueeze(2).to(device)
        y = y.to(device)
        output = model(x)
        loss = loss_fn(output,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('epoch #{} loss: {}'.format(epoch,loss))

model.eval()
results = []
labels = []
with torch.no_grad():
    for i, data in enumerate(testLoader):
            x, y = data
            ## x.size() = (batch_size, seq_len)
            ## y.size() = (batch_size. 1)
            if(i==0):
                results+=list(x[0].numpy().squeeze())
                labels+=list(x[0].numpy().squeeze())
            x = x.unsqueeze(2).to(device)
            output = list(model(x).cpu().numpy().squeeze())
            label = list(y.numpy().squeeze())
            results+=output
            labels+=label

plt.figure(figsize=(20,10))#设置画布的尺寸
plt.title('Examples of line chart',fontsize=20)#标题，并设定字号大小
plt.xlabel('time',fontsize=14)#设置x轴，并设定字号大小
plt.ylabel('elec',fontsize=14)#设置y轴，并设定字号大小

plt.plot(list(range(300)),labels[300:600], label='label')
plt.plot(list(range(300)),results[300:600], label='pridiction')
plt.legend(loc='upper left')
plt.show()
