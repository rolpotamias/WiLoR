# Copyright (c) OpenMMLab. All rights reserved.
import math
import numpy as np
import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from ...utils.geometry import rot6d_to_rotmat, aa_to_rotmat
from timm.models.layers import drop_path, to_2tuple, trunc_normal_

def vit(cfg):
    return ViT(
                img_size=(256, 192),
                patch_size=16,
                embed_dim=1280,
                depth=32,
                num_heads=16,
                ratio=1,
                use_checkpoint=False,
                mlp_ratio=4,
                qkv_bias=True,
                drop_path_rate=0.55,
                cfg = cfg
            )

def get_abs_pos(abs_pos, h, w, ori_h, ori_w, has_cls_token=True):
    """
    Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
        dimension for the original embeddings.
    Args:
        abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
        has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
        hw (Tuple): size of input image tokens.

    Returns:
        Absolute positional embeddings after processing with shape (1, H, W, C)
    """
    cls_token = None
    B, L, C = abs_pos.shape
    if has_cls_token:
        cls_token = abs_pos[:, 0:1]
        abs_pos = abs_pos[:, 1:]

    if ori_h != h or ori_w != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, ori_h, ori_w, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1).reshape(B, -1, C)

    else:
        new_abs_pos = abs_pos
    
    if cls_token is not None:
        new_abs_pos = torch.cat([cls_token, new_abs_pos], dim=1)
    return new_abs_pos

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self):
        return 'p={}'.format(self.drop_prob)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None,):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim

        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm, attn_head_dim=None
                 ):
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim
            )

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, ratio=1):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (ratio ** 2)
        self.patch_shape = (int(img_size[0] // patch_size[0] * ratio), int(img_size[1] // patch_size[1] * ratio))
        self.origin_patch_shape = (int(img_size[0] // patch_size[0]), int(img_size[1] // patch_size[1]))
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=(patch_size[0] // ratio), padding=4 + 2 * (ratio//2-1))

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]

        x = x.flatten(2).transpose(1, 2)
        return x, (Hp, Wp)


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ViT(nn.Module):

    def __init__(self,
                 img_size=224, patch_size=16, in_chans=3, num_classes=80, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=None, use_checkpoint=False, 
                 frozen_stages=-1, ratio=1, last_norm=True,
                 patch_padding='pad', freeze_attn=False, freeze_ffn=False,cfg=None, 
                 ):
        # Protect mutable default arguments
        super(ViT, self).__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.frozen_stages = frozen_stages
        self.use_checkpoint = use_checkpoint
        self.patch_padding = patch_padding
        self.freeze_attn = freeze_attn
        self.freeze_ffn = freeze_ffn
        self.depth = depth

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, ratio=ratio)
        num_patches = self.patch_embed.num_patches
        
        ##########################################
        self.cfg = cfg
        self.joint_rep_type = cfg.MODEL.MANO_HEAD.get('JOINT_REP', '6d')
        self.joint_rep_dim = {'6d': 6, 'aa': 3}[self.joint_rep_type]
        npose = self.joint_rep_dim * (cfg.MANO.NUM_HAND_JOINTS + 1)
        self.npose = npose
        mean_params = np.load(cfg.MANO.MEAN_PARAMS)
        init_cam = torch.from_numpy(mean_params['cam'].astype(np.float32)).unsqueeze(0)
        self.register_buffer('init_cam', init_cam)
        init_hand_pose = torch.from_numpy(mean_params['pose'].astype(np.float32)).unsqueeze(0)
        init_betas = torch.from_numpy(mean_params['shape'].astype('float32')).unsqueeze(0)
        self.register_buffer('init_hand_pose', init_hand_pose)
        self.register_buffer('init_betas', init_betas)
        
        self.pose_emb  = nn.Linear(self.joint_rep_dim  , embed_dim)
        self.shape_emb = nn.Linear(10  , embed_dim)
        self.cam_emb   = nn.Linear(3 , embed_dim)
        
        self.decpose  = nn.Linear(self.num_features, 6)
        self.decshape = nn.Linear(self.num_features, 10)
        self.deccam   = nn.Linear(self.num_features, 3)
        if cfg.MODEL.MANO_HEAD.get('INIT_DECODER_XAVIER', False):
            # True by default in MLP. False by default in Transformer
            nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
            nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
            nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

            
        ##########################################
        
        # since the pretraining model has class token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                )
            for i in range(depth)])

        self.last_norm = norm_layer(embed_dim) if last_norm else nn.Identity()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)

        self._freeze_stages()

    def _freeze_stages(self):
        """Freeze parameters."""
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = self.blocks[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

        if self.freeze_attn:
            for i in range(0, self.depth):
                m = self.blocks[i]
                m.attn.eval()
                m.norm1.eval()
                for param in m.attn.parameters():
                    param.requires_grad = False
                for param in m.norm1.parameters():
                    param.requires_grad = False

        if self.freeze_ffn:
            self.pos_embed.requires_grad = False
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            for i in range(0, self.depth):
                m = self.blocks[i]
                m.mlp.eval()
                m.norm2.eval()
                for param in m.mlp.parameters():
                    param.requires_grad = False
                for param in m.norm2.parameters():
                    param.requires_grad = False

    def init_weights(self):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        B, C, H, W = x.shape
        x, (Hp, Wp) = self.patch_embed(x)
  
        if self.pos_embed is not None:
            # fit for multiple GPU training
            # since the first element for pos embed (sin-cos manner) is zero, it will cause no difference
            x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]
        # X [B, 192, 1280]
        # x cat [ mean_pose, mean_shape, mean_cam] tokens 
        pose_tokens  = self.pose_emb(self.init_hand_pose.reshape(1, self.cfg.MANO.NUM_HAND_JOINTS + 1, self.joint_rep_dim)).repeat(B, 1, 1)
        shape_tokens = self.shape_emb(self.init_betas).unsqueeze(1).repeat(B, 1, 1)
        cam_tokens   = self.cam_emb(self.init_cam).unsqueeze(1).repeat(B, 1, 1)
        
        x = torch.cat([pose_tokens, shape_tokens, cam_tokens, x], 1)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        x = self.last_norm(x)
        
        
        pose_feat  = x[:, :(self.cfg.MANO.NUM_HAND_JOINTS + 1)]
        shape_feat = x[:, (self.cfg.MANO.NUM_HAND_JOINTS + 1):1+(self.cfg.MANO.NUM_HAND_JOINTS + 1)]
        cam_feat   = x[:, 1+(self.cfg.MANO.NUM_HAND_JOINTS + 1):2+(self.cfg.MANO.NUM_HAND_JOINTS + 1)]
        
        #print(pose_feat.shape, shape_feat.shape, cam_feat.shape)
        pred_hand_pose = self.decpose(pose_feat).reshape(B, -1) + self.init_hand_pose  #B , 96
        pred_betas = self.decshape(shape_feat).reshape(B, -1) + self.init_betas        #B , 10
        pred_cam = self.deccam(cam_feat).reshape(B, -1) + self.init_cam                #B , 3
        
        pred_mano_feats = {}
        pred_mano_feats['hand_pose'] = pred_hand_pose
        pred_mano_feats['betas']     = pred_betas
        pred_mano_feats['cam']       = pred_cam

        
        joint_conversion_fn = {
                '6d': rot6d_to_rotmat,
                'aa': lambda x: aa_to_rotmat(x.view(-1, 3).contiguous())
            }[self.joint_rep_type]
 
        pred_hand_pose = joint_conversion_fn(pred_hand_pose).view(B, self.cfg.MANO.NUM_HAND_JOINTS+1, 3, 3)
        pred_mano_params = {'global_orient': pred_hand_pose[:, [0]],
                            'hand_pose': pred_hand_pose[:, 1:],
                            'betas': pred_betas}
        
        img_feat = x[:, 2+(self.cfg.MANO.NUM_HAND_JOINTS + 1):].reshape(B, Hp, Wp, -1).permute(0, 3, 1, 2)
        return pred_mano_params, pred_cam, pred_mano_feats, img_feat

    def forward(self, x):
        x = self.forward_features(x)
        return x

    def train(self, mode=True):
        """Convert the model into training mode."""
        super().train(mode)
        self._freeze_stages()
