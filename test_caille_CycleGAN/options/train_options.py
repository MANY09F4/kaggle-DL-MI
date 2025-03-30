#from test_caille_CycleGAN.options.base_options import BaseOptions
from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        super().initialize()
        self.parser.add_argument('--epoch_count', type=int, default=1, help='première époque')
        self.parser.add_argument('--n_epochs', type=int, default=100, help='nombre d\'époques avec lr constant')
        self.parser.add_argument('--n_epochs_decay', type=int, default=0, help='nombre d\'époques avec décroissance de lr')
        self.parser.add_argument('--lr_G', type=float, default=0.0002, help='learning rate pour le générateur')
        self.parser.add_argument('--lr_D', type=float, default=0.0002, help='learning rate pour le discriminateur')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum pour Adam')
        self.parser.add_argument('--netD_opt', type=str, default='adam', help='optimizer pour D : adam | sgd')
        self.parser.add_argument('--print_freq', type=int, default=100, help='fréquence d\'affichage des pertes')
        self.parser.add_argument('--save_epoch_freq', type=int, default=10, help='fréquence de sauvegarde des modèles (en époques)')
        self.parser.add_argument('--lambda_A', type=float, default=10.0, help='poids cycle loss A')
        self.parser.add_argument('--lambda_B', type=float, default=10.0, help='poids cycle loss B')
        self.parser.add_argument('--lambda_identity', type=float, default=0.5, help='poids identity loss')
        self.parser.add_argument('--max_items_A', type=int, default=None, help='Nombre max d\'images dans train (domaine A)')
        self.parser.add_argument('--max_items_B', type=int, default=None, help='Nombre max d\'images dans val (domaine B)')
        self.parser.add_argument('--train_path', type=str, default='data/train.h5', help='chemin vers train.h5')
        self.parser.add_argument('--val_path', type=str, default='data/val.h5', help='chemin vers val.h5')
        self.parser.add_argument('--test', action='store_true', help='Activer le mode test (train+val -> test)')
        self.parser.add_argument('--test_path', type=str, default='', help='Chemin vers test.h5 si --test est activé')
