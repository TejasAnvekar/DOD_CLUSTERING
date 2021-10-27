"""
set global seed
save latent,
save recon_metric csv,
save loss csv,
save eval metric csv,
save recon image 
save best model
"""


import os
import pickle
import numpy as np
import random
import torch
import pandas as pd
import torchvision
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


def set_seed_globally(seed_value=0, if_cuda=True, gpu=0):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    device = "cuda" if torch.cuda.is_available() and if_cuda else "cpu"

    if device == "cuda":
        print(device)
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(gpu)
    else:
        print("device not found")
        exit()

    return device


class save_results():
    def __init__(self, r_path):
        self.r_path = r_path

        self.latent_path = self.r_path+"/Latents/"
        self.csv_path = self.r_path+"/CSV/"
        self.model_path = self.r_path+"/Model/"
        self.images_path = self.r_path+"/Images/"

        """
        r_path = batch:{}_lr:{}_optim:{}_alpha:{}_beta:{}_gamma:{}_recentre:{}
        
        """

        if not os.path.exists(self.latent_path):
            os.makedirs(self.latent_path)

        if not os.path.exists(self.csv_path):
            os.makedirs(self.csv_path)

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        if not os.path.exists(self.images_path):
            os.makedirs(self.images_path)

    def save_latent(self, epoch, latent, y, name='z'):
        data = {"latent": latent, "target": y}
        with open(self.latent_path+name+f"_{epoch}_.pickle", 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def save_loss(self, loss):
        data = loss
        dataframe = pd.DataFrame(data=data)
        dataframe.to_csv(self.csv_path+"loss.csv")

    def save_eval_metric(self, metric, name):
        data = metric
        dataframe = pd.DataFrame(data=data)
        dataframe.to_csv(self.csv_path+f"{name}_eval_metric.csv")

    def save_recon_metric(self, recon_metric):#//@todo
        data = recon_metric
        dataframe = pd.DataFrame(data=data)
        dataframe.to_csv(self.csv_path+"recon_metric.csv")

    def save_images(self, epoch, GT, P):
        
        comparison = torch.cat([GT, P], axis=2)
        if not os.path.exists(self.images_path+f"EPOCH:{epoch}"):
            os.makedirs(self.images_path+f"EPOCH:{epoch}")
        # img_grid = torchvision.utils.make_grid(comparison.unsqueeze(0))
        torchvision.utils.save_image(comparison, self.images_path+f"EPOCH:{epoch}/COMPARE.png", normalize=True,nrow=10)
        for i in range(P.shape[0]):
            torchvision.utils.save_image(P[i,:,:,:], self.images_path+f"EPOCH:{epoch}/P_{i}.png", normalize=True,nrow=10)





    def save_model(self, dic,name):
        torch.save(dic, self.model_path+f"{name}_best.pth.tar")
