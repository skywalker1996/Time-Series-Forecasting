from torch.utils.data import DataLoader,Dataset
import torch
import pandas 

class EnergyData(Dataset):
    def __init__(self, root_dir, history_len=10, forecast_len=1, interval=5):
    
        self.history_len = history_len
        self.forecast_len = forecast_len
        self.interval = interval
        train_df = pandas.read_csv(root_dir+'train.csv')
        self.raw_data = train_df[train_df['building_id'].apply(lambda x:x==1)&
                     train_df['meter_reading'].apply(lambda x:x>0)&
                     train_df['meter'].apply(lambda x:x==0)].values[-3000:][:,3].astype('float32')
        
        self.len = int((self.raw_data.shape[0]-self.history_len-self.forecast_len)/self.interval)+1
    
    def __getitem__(self, index):
        if(index<self.len):
            x_start = self.interval*index
            y_start = x_start+self.history_len
            x_data = torch.from_numpy(self.raw_data[x_start:x_start+self.history_len])
            y_data = torch.from_numpy(self.raw_data[y_start:y_start+self.forecast_len])
            return x_data, y_data
        else:
            return None
    
    def __len__(self):
        return self.len
    
