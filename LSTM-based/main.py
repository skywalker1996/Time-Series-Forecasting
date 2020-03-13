from lib.model import *
from data.dataset import *
from torch.utils.data import Dataset, DataLoader 


data_root = "./data/dataset/"
trainSet = EnergyData(root_dir=data_root, history_len=50, forecast_len=1, interval=2)
trainLoader = DataLoader(dataset=trainSet, batch_size=16,shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TestNet(seq_len=50).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
loss_fn = nn.MSELoss()

model.train()
for epoch in range(50):
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
    