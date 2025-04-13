import argparse

class BaseOptions():
    """
    This class defines common options used in both training and testing phases.
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # General experiment options
        self.parser.add_argument('--dataroot', default='', help='(Unused in this simplified version)')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='Name of the experiment (used for checkpoints directory)')
        self.parser.add_argument('--gpu_ids', type=int, default=0, help='GPU ID to use. Set to -1 for CPU')
        self.parser.add_argument('--model', type=str, default='multistain_cyclegan', help='Name of the model to use')
        self.parser.add_argument('--direction', type=str, default='AtoB', help='Direction of translation: AtoB or BtoA')
        self.parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')

        # Model architecture
        self.parser.add_argument('--input_nc', type=int, default=3, help='Number of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='Number of output image channels')
        self.parser.add_argument('--ngf', type=int, default=64, help='Number of generator filters in the last conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='Number of discriminator filters in the first conv layer')
        self.parser.add_argument('--netG', type=str, default='resnet_9blocks', help='Generator architecture')
        self.parser.add_argument('--netD', type=str, default='basic', help='Discriminator architecture')
        self.parser.add_argument('--norm', type=str, default='instance', help='Normalization method: instance | batch')
        self.parser.add_argument('--no_dropout', action='store_true', help='Disable dropout for the generator')
        self.parser.add_argument('--init_type', type=str, default='normal', help='Weight initialization method')
        self.parser.add_argument('--init_gain', type=float, default=0.02, help='Scaling factor for weight initialization')

        # Dataset mode
        self.parser.add_argument('--dataset_mode', type=str, default='unaligned', help='Dataset loading mode (unaligned)')

        # Data augmentation (color jitter)
        self.parser.add_argument('--color_augment', action='store_true', help='Enable color jitter augmentation')
        self.parser.add_argument('--brightness', type=float, default=0.1, help='Brightness jitter factor')
        self.parser.add_argument('--contrast', type=float, default=0.1, help='Contrast jitter factor')
        self.parser.add_argument('--saturation', type=float, default=0.1, help='Saturation jitter factor')
        self.parser.add_argument('--hue', type=float, default=0.05, help='Hue jitter factor')

        # GAN and training specifics
        self.parser.add_argument('--gan_mode', type=str, default='lsgan', help='GAN loss mode: lsgan | vanilla | wgangp')
        self.parser.add_argument('--pool_size', type=int, default=50, help='Size of image buffer that stores previously generated images')
        self.parser.add_argument('--D_thresh', action='store_true', help='Enable discriminator loss thresholding')
        self.parser.add_argument('--D_thresh_value', type=float, default=0.1, help='Discriminator threshold value')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='Number of conv layers in PatchGAN discriminator')

        # Learning rate scheduling
        self.parser.add_argument('--lr_policy', type=str, default='linear', help='Learning rate scheduler: linear | step | plateau | cosine')
        self.parser.add_argument('--lr_decay_iters', type=int, default=50, help='Decay interval for step policy')

        # Visualization options
        self.parser.add_argument('--display_id', type=int, default=-1, help='Window ID for visualization tool (Visdom)')
        self.parser.add_argument('--display_winsize', type=int, default=256, help='Display window size')
        self.parser.add_argument('--display_port', type=int, default=8097, help='Port for Visdom server')
        self.parser.add_argument('--display_server', type=str, default="http://localhost", help='Address of Visdom server')
        self.parser.add_argument('--display_env', type=str, default='main', help='Visdom environment')
        self.parser.add_argument('--display_ncols', type=int, default=4, help='Number of images per row for Visdom')
        self.parser.add_argument('--no_html', action='store_true', help='Do not save intermediate results to HTML')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='Directory to save model checkpoints')

        self.initialized = True

    def parse(self):
        """Parse command line arguments and return options."""
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()
        opt.isTrain = True  # Always set to True for training mode
        return opt
