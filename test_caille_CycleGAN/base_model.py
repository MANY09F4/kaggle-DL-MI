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
        self.visual_names = []
        self.image_paths = []
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.schedulers = []

    def setup(self, opt):
        pass  # On peut ignorer les schedulers pour commencer

    def get_current_visuals(self):
        visual_ret = {}
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def save_networks(self, epoch):
        for name in self.model_names:
            net = getattr(self, 'net' + name)
            save_path = os.path.join(self.save_dir, f'net{name}_epoch{epoch}.pth')
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

    def get_current_losses(self):
        losses_ret = {}
        for name in self.loss_names:
            if isinstance(name, str):
                losses_ret[name] = float(getattr(self, 'loss_' + name))
        return losses_ret

    def print_networks(self, verbose=False):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            net = getattr(self, 'net' + name)
            num_params = sum(p.numel() for p in net.parameters())
            print(f'[Network {name}] Total params: {num_params/1e6:.3f} M')
        print('---------------------------------------------')
