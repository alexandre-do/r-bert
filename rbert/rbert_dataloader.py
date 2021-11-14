from pytorch_lightning import LightningDataModule
from typing import Optional

class RbertDataModule(LightningDataModule):
    def __init__(self, train_transforms=None, val_transforms=None, test_transforms=None, dims=None):
        super().__init__(train_transforms=train_transforms, val_transforms=val_transforms, test_transforms=test_transforms, dims=dims)

    def prepare_data(self) -> None:
        return super().prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        return super().setup(stage=stage)

    def train_dataloader(self):
        return super().train_dataloader()

    def val_dataloader(self):
        return super().val_dataloader()
    
    def test_dataloader(self) :
        return super().test_dataloader()

    def predict_dataloader(self):
        return super().predict_dataloader()

    