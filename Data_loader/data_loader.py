import pandas as pd
import numpy as np
import sys,os

class dataloader():
    def __init__(self, path,annotation_path):
        self.path=path
        self.annotation_path=annotation_path
    def __getitem__(self,idx):
        print(self.path[idx])
        data=pd.read_csv(self.path[idx])
        mlii=data.iloc[:,1]
        mlii=np.asarray(mlii.tolist())
        name=self.path[idx]
        annotation_name=self.annotation_path[idx]
        #print(mlii)
        #na=path
        return mlii,name,annotation_name
    def __len__(self):
        return len(self.path)
    

