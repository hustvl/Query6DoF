CFG_NAME = ''
OUTPUT_DIR = 'runs'
RUN_NAME=''
PRINT_FREQ = 40
DIST_BACKEND = 'nccl'
AUTO_RESUME = False
VERBOSE = True
DDP = True
RESUME_FILE=''
ONLY_MODEL=False
CHANGE_SCHEDULE=False
find_unused_parameters=False
is_iter=False
# Cudnn related params
CUDNN=dict(
    BENCHMARK = True,
    DETERMINISTIC = False,
    ENABLED = True
)

DATASET=dict(
    type='PoseDataset',
    source='Real',
    mode='train',
    data_dir='data',
    n_pts=1024
)

DATALOADER=dict(
    type='DataLoader',
    batch_size=10,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)

DATALOADER['persistent_workers']=DATALOADER['num_workers']>0

OPTIMIZER=dict(
    type='AdamW',
    lr=2e-4,
    weight_decay=1e-7
)


TRAIN=dict(
    BEGIN_EPOCH=0,
    END_EPOCH=75,
    SAVE_EPOCH_STEP=5,
    VIS=False
)
