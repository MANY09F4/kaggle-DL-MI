from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    """
    This class includes training-specific options.
    It also includes shared options defined in BaseOptions.
    """

    def initialize(self):
        super().initialize()

        # Training epochs and learning rate scheduling
        self.parser.add_argument('--epoch_count', type=int, default=1, help='Starting epoch count')
        self.parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs with initial learning rate')
        self.parser.add_argument('--n_epochs_decay', type=int, default=0, help='Number of epochs to linearly decay learning rate to zero')

        # Learning rates
        self.parser.add_argument('--lr_G', type=float, default=0.0002, help='Initial learning rate for the generator')
        self.parser.add_argument('--lr_D', type=float, default=0.0002, help='Initial learning rate for the discriminator')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='Momentum term for Adam optimizer')
        self.parser.add_argument('--netD_opt', type=str, default='adam', help='Optimizer for discriminator: adam | sgd')

        # Logging and saving
        self.parser.add_argument('--print_freq', type=int, default=100, help='Frequency of showing training losses')
        self.parser.add_argument('--save_epoch_freq', type=int, default=10, help='Frequency of saving checkpoints (in epochs)')

        # Loss weighting
        self.parser.add_argument('--lambda_A', type=float, default=10.0, help='Weight for forward cycle loss (A -> B -> A)')
        self.parser.add_argument('--lambda_B', type=float, default=10.0, help='Weight for backward cycle loss (B -> A -> B)')
        self.parser.add_argument('--lambda_identity', type=float, default=0.5, help='Weight for identity loss')

        # Dataset options
        self.parser.add_argument('--max_items_A', type=int, default=None, help='Maximum number of samples to load from source domain (A)')
        self.parser.add_argument('--max_items_B', type=int, default=None, help='Maximum number of samples to load from target domain (B)')
        self.parser.add_argument('--train_path', type=str, default='data/train.h5', help='Path to training HDF5 file')
        self.parser.add_argument('--val_path', type=str, default='data/val.h5', help='Path to validation HDF5 file')
        self.parser.add_argument('--test_path', type=str, default='data/test.h5', help='Path to test HDF5 file')

        # Domain filtering
        self.parser.add_argument('--domain', type=int, default=None, help='Filter to use only images from this source domain (center index)')

        # Filtering aberrant images
        self.parser.add_argument('--aberrant_ids', type=str, default="", help='Comma-separated list of aberrant image IDs to exclude (unused)')
        self.parser.add_argument('--aberrant_ids_train', type=str, default="", help='Comma-separated list of aberrant train image IDs')
        self.parser.add_argument('--aberrant_ids_val', type=str, default="", help='Comma-separated list of aberrant val image IDs')
