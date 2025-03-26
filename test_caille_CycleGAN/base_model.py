import torch
import os

class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = [opt.gpu_ids] if opt.gpu_ids != -1 else []
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda' if self.gpu_ids else 'cpu')
        self.model_names = []
        self.optimizers = []

    def setup(self, opt):
        pass  # On peut ignorer les schedulers pour commencer

    def save_networks(self, epoch):
        for name in self.model_names:
            net = getattr(self, 'net' + name)
            save_path = os.path.join('checkpoints', self.opt.name, f'net{name}_epoch{epoch}.pth')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(net.state_dict(), save_path)

    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        """Set requies_grad=False for all the networks to avoid unnecessary computations"""
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def update_learning_rate(self):
        pass
