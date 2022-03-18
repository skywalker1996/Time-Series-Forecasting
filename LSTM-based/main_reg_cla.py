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
data_path = "./data/train.csv"

HISTORY_LEN = 6
FORECAST_LEN = 1
INTERVAL = FORECAST_LEN

trainSet = RTTData(data_path=data_path, history_len=HISTORY_LEN, forecast_len=FORECAST_LEN, interval=INTERVAL, train=True)
testSet = RTTData(data_path=data_path, history_len=HISTORY_LEN, forecast_len=FORECAST_LEN, interval=INTERVAL, train=False)
trainLoader = DataLoader(dataset=trainSet, batch_size=32,shuffle=True)
testLoader = DataLoader(dataset=testSet, batch_size=32,shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FCNet(name='LSTMNet_1',history_len=HISTORY_LEN,forecast_len=FORECAST_LEN).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
loss_reg_func = torch.nn.MSELoss()

model.train()
count=0
total = 0
for epoch in range(20):
    for i, data in enumerate(trainLoader):
        ## x.size() = (batch_size, seq_len)
        ## y.size() = (batch_size. 1)
        x, (y_reg, y_cla) = data 
        x = x.unsqueeze(2).to(device)
        y_reg = y_reg.to(device)

        out_reg = model(x)
        loss_reg = loss_reg_func(out_reg,y_reg)
        loss = loss_reg
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print('epoch #{} loss: {}'.format(epoch,loss))
	
	
print("train success!")
model.eval()
results_reg = []
labels_reg = []
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
                labels_reg+=list(x[0].numpy())          
            x = x.unsqueeze(2).to(device)
            out_reg = model(x)     
            label_reg = list(np.concatenate(y_reg.numpy()))     
            results_reg+=list(np.concatenate(out_reg.cpu().numpy()))
            labels_reg+=label_reg

			
# torch.save(model.state_dict(), "./model/save/FCNet.pth")

results_cla = []
labels_cla = []
for i in range(len(results_reg)-2):
    if((results_reg[i+1]+ results_reg[i+2])/2 - results_reg[i] > 0):
        results_cla.append(True)
    else:
        results_cla.append(False)
    
    if((labels_reg[i+1]+ labels_reg[i+2])/2 - labels_reg[i] > 0):
        labels_cla.append(True)
    else:
        labels_cla.append(False)
        
print('total = ', len(results_cla))
print(np.array(results_cla).sum())
print(np.array(labels_cla).sum())        
        
    

confusion = confusion.numpy()
print('RTT实际增大，预测增大的次数：', confusion[0][0])
print('RTT实际增大，预测减小的次数：', confusion[0][1])
print('RTT实际减小，预测增大的次数：', confusion[1][0])
print('RTT实际减小，预测减小的次数：', confusion[1][1])
print()
print('精确率 = {}%'.format(round(confusion[0][0]/(confusion[0][0]+confusion[1][0]) * 100, 2)))
print('召回率 = {}%'.format(round(confusion[0][0]/(confusion[0][0]+confusion[0][1]) * 100, 2)))