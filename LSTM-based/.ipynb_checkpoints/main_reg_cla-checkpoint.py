from model.model import FCNet,LSTMNet
from data.dataset import *
from torch.utils.data import Dataset, DataLoader 
from matplotlib import pyplot as plt 
import numpy as np

def compute_confusion(pred, label):
    result = torch.zeros(2,2)
    for i in range(len(pred)):
        if(pred[i]==1 and label[i]==1):
            result[0][0]+=1
        elif(pred[i]==0 and label[i]==1):
            result[0][1]+=1
        elif(pred[i]==1 and label[i]==0):
            result[1][0]+=1
        elif(pred[i]==0 and label[i]==0):
            result[1][1]+=1
    return result

# data_path = "./data/dataset/PCC_50cycle_directACK.csv"
data_path = "./data/dataset/PCC_50cycle_EWMAACK.csv"

HISTORY_LEN = 5
FORECAST_LEN = 1
INTERVAL = FORECAST_LEN

trainSet = RTTData(data_path=data_path, history_len=HISTORY_LEN, forecast_len=FORECAST_LEN, interval=INTERVAL, train=True)
testSet = RTTData(data_path=data_path, history_len=HISTORY_LEN, forecast_len=FORECAST_LEN, interval=INTERVAL, train=False)
trainLoader = DataLoader(dataset=trainSet, batch_size=32,shuffle=True)
testLoader = DataLoader(dataset=testSet, batch_size=32,shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMNet(name='LSTMNet_1',history_len=HISTORY_LEN,forecast_len=FORECAST_LEN).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
loss_reg_func = torch.nn.MSELoss()
loss_cla_func = torch.nn.CrossEntropyLoss()

model.train()
count=0
total = 0
for epoch in range(20):
    for i, data in enumerate(trainLoader):
        ## x.size() = (batch_size, seq_len)
        ## y.size() = (batch_size. 1)
        x, (y_reg, y_cla) = data
        x = x.unsqueeze(2).to(device)
        count += y_cla.sum().item()
        total += len(y_cla)
        y_reg = y_reg.to(device)
        y_cla = y_cla.to(device)
        out_reg, out_cla = model(x)
        loss_reg = loss_reg_func(out_reg,y_reg)
        loss_cla = loss_cla_func(out_cla,y_cla)
        loss = 0.8*loss_reg + 0.2*loss_cla
#         loss = loss_cla
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print('epoch #{} loss: {}'.format(epoch,loss))
    
    
model.eval()
results_reg = []
results_cla = []
labels_reg = []
labels_cla = []
correct = 0
total = 0
# count_label = 0
# count_pred = 0
confusion = torch.zeros(2,2)
with torch.no_grad():
    for i, data in enumerate(testLoader):
            x, (y_reg, y_cla) = data
            ## x.size() = (batch_size, seq_len)
            ## y.size() = (batch_size. 1)
            if(i==0):
                results_reg+=list(x[0].numpy())
                results_cla+=list(y_cla.numpy())
                labels_reg+=list(x[0].numpy())
                labels_cla+=list(y_cla.numpy())
            x = x.unsqueeze(2).to(device)
            out_reg, out_cla = model(x)
#             print(out_cla)
            _, pred = torch.max(out_cla, dim=1)
            
            label_reg = list(np.concatenate(y_reg.numpy()))
            label_cla = list(y_cla.numpy())
            
#             count_label+=y_cla.cpu().sum().item()
#             count_pred+=pred.cpu().sum().item()
            confusion+=compute_confusion(pred, y_cla)
            correct += (pred.cpu()==y_cla).sum().item()
            total += len(y_cla)
            
#             print(out_reg.reshape(1,-1))
            results_reg+=list(np.concatenate(out_reg.cpu().numpy()))
            results_cla+=list(pred.cpu().numpy())
            labels_reg+=label_reg
            labels_cla+=label_cla

print('Test Acc: {}'.format(correct/total))
# print('Count:', count_label, count_pred)
print('total:', total)
# print(confusion)
# print(len(results))
# print(len(labels))

confusion = confusion.numpy()
print('RTT实际增大，预测增大的次数：', confusion[0][0])
print('RTT实际增大，预测减小的次数：', confusion[0][1])
print('RTT实际减小，预测增大的次数：', confusion[1][0])
print('RTT实际减小，预测减小的次数：', confusion[1][1])
print()
print('精确率 = {}%'.format(round(confusion[0][0]/(confusion[0][0]+confusion[1][0]) * 100, 2)))
print('召回率 = {}%'.format(round(confusion[0][0]/(confusion[0][0]+confusion[0][1]) * 100, 2)))


import numpy as np
results_reg = np.array(results_reg)
results_cla = np.array(results_cla)
labels_reg = np.array(labels_reg)
labels_cla = np.array(labels_cla)

plt.figure(figsize=(20,10))#设置画布的尺寸
plt.title('Examples of line chart',fontsize=20)#标题，并设定字号大小
plt.xlabel('time',fontsize=14)#设置x轴，并设定字号大小
plt.ylabel('elec',fontsize=14)#设置y轴，并设定字号大小


plot_range = [400,500]
plot_len = plot_range[1] - plot_range[0]

print(len(results_reg))
print(len(labels_reg))

plt.plot(list(range(plot_len)), results_reg[plot_range[0]:plot_range[1]], label='label')
plt.plot(list(range(plot_len)), labels_reg[plot_range[0]:plot_range[1]], label='pridiction')
plt.legend(loc='upper left')
plt.show()