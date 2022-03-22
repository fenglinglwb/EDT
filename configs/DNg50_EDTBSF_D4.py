from easydict import EasyDict as edict


class Config:
    # dataset
    DATASET = edict()
    DATASET.TYPE = 'FinetuneTrainMsDataset'  # ['ImagenetTrainMsDataset', 'FinetuneTrainMsDataset']
    DATASET.DATASETS = ['DIV2K', 'Flickr2K', 'WED', 'BSDS500']
    # DATASET.SPLITS = ['TRAIN']  # only for imagenet, ['TRAIN', 'VAL']
    # DATASET.START_END = [[0, 200000]]  # only for imagenet, [[None:None], [None,None]]
    DATASET.INPUT_SIZE = 144
    DATASET.NOISE_DOWN = 2  # times of downsampling
    DATASET.INPUT_SIZE_DENOISE = DATASET.INPUT_SIZE # * DATASET.NOISE_DOWN ** 2
    DATASET.INPUT_SIZE_DERAIN = DATASET.INPUT_SIZE # * DATASET.NOISE_DOWN ** 2
    DATASET.SCALES = []
    DATASET.NOISE_LEVELS = [50]
    DATASET.RAIN_LEVELS = []
    DATASET.HFLIP = True
    DATASET.VFLIP = True
    DATASET.ROTATION = True
    DATASET.REPEAT = 1
    DATASET.SEED = 100

    # dataloader
    DATALOADER = edict()
    DATALOADER.IMG_PER_GPU = 1
    DATALOADER.NUM_WORKERS = 1

    # model
    MODEL = edict()
    MODEL.IN_CHANNEL = 3
    MODEL.DEPTH = DATASET.NOISE_DOWN
    MODEL.IMAGE_SIZE = DATASET.INPUT_SIZE
    MODEL.SCALES = DATASET.SCALES
    MODEL.NOISE_LEVELS = DATASET.NOISE_LEVELS
    MODEL.RAIN_LEVELS = DATASET.RAIN_LEVELS
    MODEL.WINDOW_SIZE = (6, 24)
    MODEL.IMAGE_RANGE = 1.0
    MODEL.NUM_FEAT = 32
    MODEL.DEPTHS = [6, 6, 6, 6, 6, 6]
    MODEL.EMBED_DIM = 180
    MODEL.NUM_HEADS = [6, 6, 6, 6, 6, 6]
    MODEL.MLP_RATIO = 2
    MODEL.UPSAMPLER = 'pixelshuffle'
    MODEL.RESI_CONNECTION = '1conv'
    MODEL.DOWNSAMPLE = 1  # to make sure the input size is divisible by window_size
    MODEL.DEVICE = 'cuda'

    # solver
    SOLVER = edict()
    SOLVER.OPTIMIZER = 'Adam'
    SOLVER.BASE_LR = 2e-4
    SOLVER.BETA1 = 0.9
    SOLVER.BETA2 = 0.99
    SOLVER.WEIGHT_DECAY = 0
    SOLVER.MOMENTUM = 0
    SOLVER.T_PERIOD = [800000, 1200000, 1400000, 1500000]
    SOLVER.MAX_ITER = 1600001

    # initialization
    CONTINUE_ITER = None
    INIT_MODEL = None
    FILTER_LIST = []  # to filter some params in init_model

    # log and save
    LOG_PERIOD = 100
    SAVE_PERIOD = 10000

    # validation
    VAL = edict()
    VAL.PERIOD = 10000
    VAL.TYPE = 'ValDataset'
    VAL.DATASET = 'CBSD68'
    VAL.SCALES = DATASET.SCALES
    VAL.NOISE_LEVELS = DATASET.NOISE_LEVELS
    VAL.RAIN_LEVELS = DATASET.RAIN_LEVELS
    VAL.IMG_PER_GPU = 1
    VAL.NUM_WORKERS = 1
    VAL.SAVE_IMG = False


config = Config()



