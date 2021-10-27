import torch.nn.functional as F 
import torch 
from torch.autograd.functional import jacobian

class DOD():
    def __init__(self,**kwargs):
        self.x1 = kwargs["x1"]
        self.x2 = kwargs["x2"]
        self.x1 = self.x1.reshape(self.x1.shape[0],-1)
        self.x2 = self.x2.reshape(self.x2.shape[0],-1)
        self.a1 = kwargs["a1"]
        self.a2 = kwargs["a2"]
        self.z1 = kwargs["z1"]
        self.z2 = kwargs["z2"]
        self.dis = kwargs["dis"]

    def distance(self,v1,v2):
        if self.dis == "mse":
            return torch.mean((v1-v2)**2,dim=1)
        if self.dis == "mae":
            return torch.mean(torch.abs(v1-v2),dim=1)
        if self.dis =="cosine":
            return 1-F.cosine_similarity(v1,v2)
        
    def LDM(self,v1,v2):
        return torch.mean(v1-v2,dim=0)

    def tuple_of_tensors_to_tensor(self,tuple_of_tensors):
        return torch.cat(list(tuple_of_tensors),dim=0)

    def regularizor(self,p=1):
        a1_norm = torch.norm(self.a1,p=p)
        a2_norm = torch.norm(self.a2,p=p)

        z1_norm = torch.norm(self.z1,p=p)
        z2_norm = torch.norm(self.z2,p=p)

        x1_norm = torch.norm(self.x1,p=p)
        x2_norm = torch.norm(self.x2,p=p)

        return torch.mean(a1_norm+a2_norm+z1_norm+z2_norm+x1_norm+x2_norm)


    def __call__(self):
        X = self.distance(self.x1,self.x2)
        A = self.distance(self.a1,self.a2)
        Z = self.distance(self.z1,self.z2)
        # Z = self.tuple_of_tensors_to_tensor(jacobian(self.LDM,(self.z1,self.z2)))
        # print(X.shape,A.shape,Z.shape)

        # exit()

        return torch.mean((torch.abs(X-A)+torch.abs(A-Z))/2,dim=0) + self.regularizor()
        
