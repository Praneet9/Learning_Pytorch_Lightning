from classifier import Classifier
from dataset import ImageClassificationDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import config as cfg

model = Classifier(cfg.N_CLASSES, cfg.LR)

train_dataloader = DataLoader(
    ImageClassificationDataset(cfg.TRAIN_DIR), 
    batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS
)
test_dataloader = DataLoader(ImageClassificationDataset(cfg.TEST_DIR), 
                             batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS)

trainer = pl.Trainer(
    accelerator=cfg.ACCELERATOR,
    devices=cfg.DEVICES,
    min_epochs=2,
    max_epochs=cfg.N_EPOCHS,
    precision=cfg.PRECISION,
    fast_dev_run=cfg.DEV_MODE,
)
trainer.fit(model, train_dataloader, test_dataloader)
