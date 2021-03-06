gpu_id = [0]
IMAGES = '/media/lyan/lyan/train_color'
MASKS = '/media/lyan/lyan/train_label'
CHECKPOINT_DIR = 'checkpoints'
USE_MATPLOTLIB_VIS = False
RESIZE_TO = 256
VAL_EPOCH_DICE = 'val_epoch_dice'
VAL_EPOCH_BCE = 'val_epoch_bce'
PER_ITER_LOSS = 'per_iter_loss'
PER_EPOCH_LOSS = 'per_epoch_loss'
BASE_BATCH_SIZE = 8
BATCH_SIZE = len(gpu_id) * BASE_BATCH_SIZE
EPOCH_NUM = 200
PER_ITER_DICE = 'per_iter_dice'
PER_ITER_IOU = 'per_iter_iou'
RESTORE_FROM = None
RESTORE_INTERRUPTED = 'interrupted.pth'
LEARNING_RATE = 1e-3
NUM_CLASSES = 1  # fixme
