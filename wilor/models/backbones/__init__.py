from .vit import vit

def create_backbone(cfg):
    if cfg.MODEL.BACKBONE.TYPE == 'vit':
        return vit(cfg)
    elif cfg.MODEL.BACKBONE.TYPE == 'fast_vit':
        import torch 
        import sys 
        from timm.models import create_model
        #from models.modules.mobileone import reparameterize_model
        fast_vit = create_model("fastvit_ma36", drop_path_rate=0.2)
        checkpoint = torch.load('./pretrained_models/fastvit_ma36.pt')
        fast_vit.load_state_dict(checkpoint['state_dict'])
        return fast_vit
        
    else:
        raise NotImplementedError('Backbone type is not implemented')
