import torch
import itertools
from CycleGAN.util.image_pool import ImagePool
from CycleGAN.base_model import BaseModel
from CycleGAN import networks
from torchvision.transforms import Grayscale, ColorJitter

# AMP support for mixed precision training
if hasattr(torch.cuda.amp, 'autocast'):
    autocast = torch.cuda.amp.autocast
    GradScaler = torch.cuda.amp.GradScaler
else:
    autocast = torch.amp.autocast
    GradScaler = torch.amp.GradScaler

class MultiStainCycleGANModel(BaseModel):
    """
    Implements the CycleGAN model with grayscale and color-augmented input support,
    adapted for multi-center stain normalization in histopathology images.

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Adds model-specific options."""
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0)
            parser.add_argument('--lambda_B', type=float, default=10.0)
            parser.add_argument('--lambda_identity', type=float, default=0.5)
        return parser

    def __init__(self, opt):
        """Initialize the model, generators, discriminators, and losses."""
        super().__init__(opt)
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        self.visual_names = ['real_A', 'fake_B', 'rec_A', 'real_B', 'fake_A', 'rec_B', 'gray_A', 'gray_B']

        if self.isTrain and opt.lambda_identity > 0.0:
            self.visual_names += ['idt_B', 'idt_A']

        self.to_grayscale = Grayscale(3)
        self.color_augment = ColorJitter(brightness=opt.brightness, contrast=opt.contrast,
                                         saturation=opt.saturation, hue=opt.hue)

        # Network initialization
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:
            self.model_names = ['G_A', 'G_B']

        self.netG_A = networks.define_G(3, 3, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(3, 3, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)

            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()

            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr_G, betas=(opt.beta1, 0.999))
            if opt.netD_opt == "adam":
                self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                    lr=opt.lr_D, betas=(opt.beta1, 0.999))
            elif opt.netD_opt == "sgd":
                self.optimizer_D = torch.optim.SGD(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                   lr=opt.lr_D)

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            self.grad_scaler_G = GradScaler(enabled=self.device.type == 'cuda')
            self.grad_scaler_D = GradScaler(enabled=self.device.type == 'cuda')

        self.D_thresh = opt.D_thresh
        self.D_thresh_value = opt.D_thresh_value

    def set_input(self, input):
        """Prepare the input images and perform grayscale/color jitter preprocessing."""
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)

        if self.isTrain:
            self.gray_A = self.to_grayscale(self.color_augment(self.real_A))
            self.gray_B = self.to_grayscale(self.color_augment(self.real_B))
        else:
            self.gray_A = self.to_grayscale(self.real_A)
            self.gray_B = self.to_grayscale(self.real_B)

        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Forward pass for both directions: A→B→A and B→A→B."""
        self.fake_B = self.netG_A(self.gray_A)
        self.gray_fake_B = self.to_grayscale(self.color_augment(self.fake_B)) if self.isTrain else self.to_grayscale(self.fake_B)
        self.rec_A = self.netG_B(self.gray_fake_B)

        self.fake_A = self.netG_B(self.gray_B)
        self.gray_fake_A = self.to_grayscale(self.color_augment(self.fake_A)) if self.isTrain else self.to_grayscale(self.fake_A)
        self.rec_B = self.netG_A(self.gray_fake_A)

    def compute_visuals(self):
        """Override to compute additional visual outputs."""
        pass

    def test(self):
        """Inference mode with no gradient computation."""
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def backward_D_basic(self, netD, real, fake):
        """Compute discriminator loss."""
        pred_real = netD(real)
        pred_fake = netD(fake.detach())
        loss_D = (self.criterionGAN(pred_real, True) + self.criterionGAN(pred_fake, False)) * 0.5
        return loss_D

    def backward_D_A(self):
        """Update D_A."""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
        return self.loss_D_A

    def backward_D_B(self):
        """Update D_B."""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
        return self.loss_D_B

    def backward_G(self):
        """Compute generator losses: adversarial, cycle, identity."""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        if lambda_idt > 0:
            self.idt_A = self.netG_A(self.to_grayscale(self.real_B) if self.opt.color_augment else self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt

            self.idt_B = self.netG_B(self.to_grayscale(self.real_A) if self.opt.color_augment else self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        self.loss_G = (self.loss_G_A + self.loss_G_B +
                       self.loss_cycle_A + self.loss_cycle_B +
                       self.loss_idt_A + self.loss_idt_B)
        return self.loss_G

    def optimize_parameters(self):
        """Main optimization loop: update generators and discriminators."""
        with autocast():
            self.forward()

        # Optimize generators
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        with autocast():
            loss_G = self.backward_G()
        self.grad_scaler_G.scale(loss_G).backward()
        self.grad_scaler_G.step(self.optimizer_G)
        self.grad_scaler_G.update()

        # Optimize discriminators
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        with autocast():
            loss_D_A = self.backward_D_A()
            loss_D_B = self.backward_D_B()
            loss_D = loss_D_A + loss_D_B

        if not (self.D_thresh and (loss_D_A <= self.D_thresh_value or loss_D_B <= self.D_thresh_value)):
            self.grad_scaler_D.scale(loss_D).backward()
            self.grad_scaler_D.step(self.optimizer_D)
            self.grad_scaler_D.update()
