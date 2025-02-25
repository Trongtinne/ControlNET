class DataConfig:
    def __init__(self):
        self.data_root = "data/colorization"
        self.image_size = 1024  # SDXL default resolution
        self.train_val_split = 0.9
        self.num_workers = 4