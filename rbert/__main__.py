from pytorch_lightning.core import datamodule
from rbert_dataloader import RbertDataModule
from rbert_classifier import RbertClassifier
import pytorch_lightning as pl

# init model
model = RbertClassifier()

# load data
datamodule = RbertDataModule()

# train 
trainer = pl.Trainer()
trainer.fit(model, datamodule=datamodule)

# validate
trainer.validate(datamodule=datamodule)

# test
trainer.test(datamodule=datamodule)

# predict 
predictions = trainer.predict(datamodule=datamodule)