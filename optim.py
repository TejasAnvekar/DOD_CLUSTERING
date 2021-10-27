import torch.optim as optim


class optimizer():
    def __init__(self, model, lr, optimizer='adam'):
        params = model.parameters()
        if optimizer == 'adam':
            self.optim = optim.Adam(params=params, lr=lr,betas=(0.9,0.999))
        if optimizer == 'sgd':
            self.optim = optim.SGD(params=params, lr=lr)


    def call(self):
        return self.optim
