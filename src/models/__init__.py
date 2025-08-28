from typing import Type
from torch import nn

def get_net_from_name(name: str) -> Type[nn.Module]:
    if name.lower() == 'mednext':
        from .MedNeXT.mednextv2 import MedNeXT
        return MedNeXT

    from .PlainUNet import unet as n
    if name.lower() in ['unet', 'plainconvunet']:
        return n.PlainConvUNet
    elif name.lower() == 'resunet':
        return n.ResidualUNet

    msg = (
        'Invalid network name. Implemented options are: [MedNeXT, UNet/PlainConvUNet, ResUNet]. Had "%s"' % name
    )
    raise ValueError(msg)