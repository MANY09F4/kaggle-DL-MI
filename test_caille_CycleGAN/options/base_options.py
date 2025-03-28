import argparse

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        #self.parser.add_argument('--dataroot', required=True, help='chemin vers le dossier contenant trainA/trainB')
        self.parser.add_argument('--dataroot', default='', help='(inutilisé dans notre version simplifiée)')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='nom de l\'expérience (checkpoints)')
        self.parser.add_argument('--gpu_ids', type=int, default=0, help='ID du GPU à utiliser. -1 pour CPU')
        self.parser.add_argument('--model', type=str, default='multistain_cyclegan', help='nom du modèle à utiliser')
        self.parser.add_argument('--direction', type=str, default='AtoB', help='AtoB ou BtoA')
        self.parser.add_argument('--batch_size', type=int, default=4, help='taille de batch')
        self.parser.add_argument('--input_nc', type=int, default=3, help='nombre de canaux des images d\'entrée')
        self.parser.add_argument('--output_nc', type=int, default=3, help='nombre de canaux des images de sortie')
        self.parser.add_argument('--ngf', type=int, default=64, help='nombre de filtres dans les couches G')
        self.parser.add_argument('--ndf', type=int, default=64, help='nombre de filtres dans les couches D')
        self.parser.add_argument('--netG', type=str, default='resnet_9blocks', help='architecture du générateur')
        self.parser.add_argument('--netD', type=str, default='basic', help='architecture du discriminateur')
        self.parser.add_argument('--norm', type=str, default='instance', help='type de normalisation : instance | batch')
        self.parser.add_argument('--no_dropout', action='store_true', help='désactive le dropout dans le générateur')
        self.parser.add_argument('--init_type', type=str, default='normal', help='initialisation des poids')
        self.parser.add_argument('--init_gain', type=float, default=0.02, help='gain pour initialisation')
        self.parser.add_argument('--dataset_mode', type=str, default='unaligned', help='type de dataset (doit être unaligned)')
        self.parser.add_argument('--color_augment', action='store_true', help='active la color jitter')
        self.parser.add_argument('--brightness', type=float, default=0.1)
        self.parser.add_argument('--contrast', type=float, default=0.1)
        self.parser.add_argument('--saturation', type=float, default=0.1)
        self.parser.add_argument('--hue', type=float, default=0.05)
        self.parser.add_argument('--gan_mode', type=str, default='lsgan', help='type de GAN loss : lsgan | vanilla | wgangp')
        self.parser.add_argument('--pool_size', type=int, default=50, help='taille du buffer d\'images générées')
        self.parser.add_argument('--D_thresh', action='store_true', help='active un seuil pour la mise à jour de D')
        self.parser.add_argument('--D_thresh_value', type=float, default=0.5)
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='nombre de couches pour le discriminateur (PatchGAN)')
        self.parser.add_argument('--lr_policy', type=str, default='linear', help='type de scheduler de lr')
        self.parser.add_argument('--lr_decay_iters', type=int, default=50, help='intervalle de réduction de lr si step')
        self.parser.add_argument('--display_id', type=int, default=-1, help='window id of the web display')
        self.parser.add_argument('--display_winsize', type=int, default=256, help='visdom display window size')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port')
        self.parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server')
        self.parser.add_argument('--display_env', type=str, default='main', help='visdom display environment')
        self.parser.add_argument('--display_ncols', type=int, default=4, help='number of images per row in visdom')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/web/')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()
        opt.isTrain = True  # on force pour l'entraînement
        return opt
