from abc import ABC, abstractmethod

from torch import nn

from .ce import RobustCrossEntropyLoss, TopKLoss
from .dice import MemoryEfficientSoftDiceLoss
from .xor_dice import SoftXORDiceLoss
from .loss import DC_and_CE_loss

class LossFactory(ABC):

    def __init__(self, **kwargs):
        self.kwargs = self._set_kwargs(**kwargs)
    
    @abstractmethod
    def _set_kwargs(self, **kwargs) -> dict:
        """Should set the kwargs to use for the loss"""

    @abstractmethod
    def get_loss(self):
        """Returns the actual loss"""


class CELossFactory(LossFactory):
    def _set_kwargs(self, **kwargs):
        default_kwargs = {}
        default_kwargs.update(kwargs)
        return default_kwargs

class RobustCELossFactory(CELossFactory):
    def get_loss(self):
        return RobustCrossEntropyLoss(**self.kwargs)

class BCELossFactory(CELossFactory):
    def get_loss(self):
        return nn.BCEWithLogitsLoss(**self.kwargs)
    
class TopKLossFactory(LossFactory):
    def _set_kwargs(self, **kwargs):
        default_kwargs = {
            'k': 10,
            'label_smoothing': 0.1
        }
        default_kwargs.update(kwargs)
        return default_kwargs    
    
    def get_loss(self):
        return TopKLoss(**self.kwargs)
    
class DiceLossFactory(LossFactory):
    def _set_kwargs(self, **kwargs):    
        default_kwargs = {
            'batch_dice': False,
            'do_bg': False,
            'smooth': 1e-5
        }
        default_kwargs.update(kwargs)
        return default_kwargs
    
class MemoryEfficientSoftDiceLossFactory(DiceLossFactory):
    def get_loss(self):
        return MemoryEfficientSoftDiceLoss(**self.kwargs)
    
class SoftXORDiceLossFactory(DiceLossFactory):
    def get_loss(self):
        return SoftXORDiceLoss(**self.kwargs)
    
class CompoundLossFactory(ABC):
    def __init__(self, ce_kwargs: dict, dc_kwargs: dict):
        self.ce_kwargs = self._set_ce_kwargs(**ce_kwargs)
        self.dc_kwargs = self._set_dc_kwargs(**dc_kwargs)
    
    @abstractmethod
    def _set_ce_kwargs(self, **ce_kwargs) -> dict: 
        ...
    
    @abstractmethod
    def _set_dc_kwargs(self, **dc_kwargs) -> dict:
        ...
    
    @abstractmethod
    def get_loss(self):
        ...

class CEDiceLossFactory(CompoundLossFactory):

    def _set_ce_kwargs(self, **ce_kwargs):
        default_ce_kwargs = BCELossFactory().kwargs
        default_ce_kwargs.update(ce_kwargs)
        return default_ce_kwargs
    
    def _set_dc_kwargs(self, **dc_kwargs):
        default_dc_kwargs = MemoryEfficientSoftDiceLossFactory().kwargs
        default_dc_kwargs.update(dc_kwargs)
        return default_dc_kwargs
    
    def get_loss(self, celoss: CELossFactory, dcloss: DiceLossFactory):
        return


if __name__ == "__main__":
    print(vars(CEDiceLossFactory({},{})))