class Visualizer():
    def __init__(self, opt):
        self.opt = opt

    def print_current_losses(self, epoch, iters, losses):
        message = f"(Epoch: {epoch}, Iters: {iters}) "
        for k, v in losses.items():
            message += f"{k}: {v:.3f} "
        print(message)
