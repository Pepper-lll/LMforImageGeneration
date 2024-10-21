from .base import HparamsBase


class HparamsBinaryAE(HparamsBase):
    def __init__(self, dataset):
        super().__init__(dataset)
        # defaults that are same for all datasets
        self.base_lr = 4.5e-6
        self.beta = 0.25
        self.path_to_data = None
        self.code_weight = 0.0
        self.norm_first = False
        self.use_tanh = False
        self.deterministic = False
        self.gen_mul = 1.0
        
        if self.dataset == 'churches' or self.dataset == "bedrooms":
            self.attn_resolutions = [16]
            self.batch_size = 3
            self.ch_mult = [1, 1, 2, 2, 4]
            self.codebook_size = 1024
            self.disc_layers = 3
            self.disc_weight_max = 1
            self.disc_start_step = 30001
            self.emb_dim = 256
            self.img_size = 256
            self.latent_shape = [1, 16, 16]
            self.n_channels = 3
            self.ndf = 64
            self.nf = 128
            self.perceptual_weight = 1.0
            self.res_blocks = 2

        elif self.dataset == 'ffhq':
            self.attn_resolutions = [16]
            self.batch_size = 3
            self.ch_mult = [1, 1, 2, 2, 4]
            self.codebook_size = 1024
            self.disc_layers = 3
            self.disc_weight_max = 1
            self.disc_start_step = 30001
            self.emb_dim = 256
            self.img_size = 256
            self.latent_shape = [1, 16, 16]
            self.n_channels = 3
            self.ndf = 64
            self.nf = 128
            self.perceptual_weight = 1.0
            self.res_blocks = 2
        
        elif self.dataset == 'celeba':
            self.attn_resolutions = [16]
            self.batch_size = 3
            self.ch_mult = [1, 1, 2, 2, 4]
            self.codebook_size = 1024
            self.disc_layers = 3
            self.disc_weight_max = 1
            self.disc_start_step = 30001
            self.emb_dim = 256
            self.img_size = 1024
            self.latent_shape = [1, 32, 32]
            self.n_channels = 3
            self.ndf = 64
            self.nf = 128
            self.perceptual_weight = 1.0
            self.res_blocks = 2
        
        elif self.dataset == 'imagenet':
            self.attn_resolutions = [16]
            self.batch_size = 3
            self.ch_mult = [1, 1, 2, 2, 4]
            self.codebook_size = 1024
            self.disc_layers = 3
            self.disc_weight_max = 1
            self.disc_start_step = 30001
            self.emb_dim = 256
            self.img_size = 256
            self.latent_shape = [1, 16, 16]
            self.n_channels = 3
            self.ndf = 64
            self.nf = 128
            self.perceptual_weight = 1.0
            self.res_blocks = 2

        elif self.dataset.startswith('laion'):
            self.attn_resolutions = [16]
            self.batch_size = 3
            self.ch_mult = [1, 1, 2, 2, 4]
            self.codebook_size = 1024
            self.disc_layers = 3
            self.disc_weight_max = 1
            self.disc_start_step = 30001
            self.emb_dim = 256
            self.img_size = 256
            self.latent_shape = [1, 16, 16]
            self.n_channels = 3
            self.ndf = 64
            self.nf = 128
            self.perceptual_weight = 1.0
            self.res_blocks = 2
            
        elif self.dataset == 'imagenet_mini':
            self.attn_resolutions = [16]
            self.batch_size = 3
            self.ch_mult = [1, 1, 2, 2, 4]
            self.codebook_size = 1024
            self.disc_layers = 3
            self.disc_weight_max = 1
            self.disc_start_step = 30001
            self.emb_dim = 256
            self.img_size = 256
            self.latent_shape = [1, 16, 16]
            self.n_channels = 3
            self.ndf = 64
            self.nf = 128
            self.perceptual_weight = 1.0
            self.res_blocks = 2
        
        elif self.dataset == 'custom':
            self.attn_resolutions = [16]
            self.batch_size = 3
            self.ch_mult = [1, 1, 2, 2, 4]
            self.codebook_size = 1024
            self.disc_layers = 3
            self.disc_weight_max = 1
            self.disc_start_step = 30001
            self.emb_dim = 256
            self.img_size = 256
            self.latent_shape = [1, 16, 16]
            self.n_channels = 3
            self.ndf = 64
            self.nf = 128
            self.perceptual_weight = 1.0
            self.res_blocks = 2

        else:
            raise KeyError(f'Defaults not defined for BinaryAE model on dataset: {self.dataset}')


def add_vqgan_args(parser):
    parser.add_argument('--attn_resolutions', nargs='+', type=int)
    parser.add_argument('--base_lr', type=float)
    parser.add_argument('--beta', type=float)
    parser.add_argument('--ch_mult', nargs='+', type=int)
    parser.add_argument('--codebook_size', type=int)
    parser.add_argument('--disc_layers', type=int)
    parser.add_argument('--disc_start_step', type=int)
    parser.add_argument('--disc_weight_max', type=float)
    parser.add_argument('--emb_dim', type=int)
    parser.add_argument('--horizontal_flip', const=True, action='store_const', default=False)
    parser.add_argument('--img_size', type=int)
    parser.add_argument('--latent_shape', nargs='+', type=int)
    parser.add_argument('--n_channels', type=int)
    parser.add_argument('--ndf', type=int)
    parser.add_argument('--nf', type=int)
    parser.add_argument('--perceptual_weight', type=int)
    parser.add_argument('--res_blocks', type=int)
    parser.add_argument('--path_to_data', type=str)
    parser.add_argument('--code_weight', type=float)
    parser.add_argument('--gen_mul', type=float)
    parser.add_argument('--norm_first', action="store_true")
    parser.add_argument('--use_tanh', action="store_true")
    parser.add_argument('--deterministic', action="store_true")
    parser.add_argument('--cp_data', action="store_true")
    parser.add_argument('--root_path', type=str)