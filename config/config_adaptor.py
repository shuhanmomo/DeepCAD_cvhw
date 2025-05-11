class ConfigAdaptor(object):
    def __init__(self, phase='train'):
        self.phase = phase
        
        # Paths
        self.latent_path = "proj_log/newDeepCAD/results/all_zs_ckpt1000.h5"
        self.clip_path = "data/CLIP_feats.json"
        
        # Training parameters
        self.batch_size = 32
        self.num_workers = 4
        self.lr = 1e-4
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.weight_decay = 0.0
        self.n_epochs = 100
        
        # Model parameters
        self.clip_dim = 512
        self.latent_dim = 256
        self.hidden_dim = 512
        self.n_layers = 3
        self.dropout = 0.1
        
        # Conditional GAN parameters
        self.noise_dim = 64
        self.lambda_gp = 10.0  # For WGAN-GP
        self.n_critic = 5      # Number of critic iterations per generator iteration
        
        # Logging and saving
        self.save_freq = 10
        self.log_freq = 100
        self.val_freq = 500
        
        # Experiment
        self.exp_name = "adaptor_training"
        self.exp_dir = f"proj_log/{self.exp_name}"
        self.device = "cuda" 