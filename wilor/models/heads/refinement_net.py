import torch
import torch.nn as nn 
import torch.nn.functional as F 
import math 
from ...utils.geometry import rot6d_to_rotmat, aa_to_rotmat
from typing import Optional

def make_linear_layers(feat_dims, relu_final=True, use_bn=False):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(nn.Linear(feat_dims[i], feat_dims[i+1]))

        # Do not use ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and relu_final):
            if use_bn:
                layers.append(nn.BatchNorm1d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

def make_conv_layers(feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.Conv2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=kernel,
                stride=stride,
                padding=padding
                ))
        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

def make_deconv_layers(feat_dims, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.ConvTranspose2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False))

        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

def sample_joint_features(img_feat, joint_xy):
    height, width = img_feat.shape[2:]
    x = joint_xy[:, :, 0] / (width - 1) * 2 - 1
    y = joint_xy[:, :, 1] / (height - 1) * 2 - 1
    grid = torch.stack((x, y), 2)[:, :, None, :]
    img_feat = F.grid_sample(img_feat, grid, align_corners=True)[:, :, :, 0]  # batch_size, channel_dim, joint_num
    img_feat = img_feat.permute(0, 2, 1).contiguous()  # batch_size, joint_num, channel_dim
    return img_feat

def perspective_projection(points: torch.Tensor,
                           translation: torch.Tensor,
                           focal_length: torch.Tensor,
                           camera_center: Optional[torch.Tensor] = None,
                           rotation: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Computes the perspective projection of a set of 3D points.
    Args:
        points (torch.Tensor): Tensor of shape (B, N, 3) containing the input 3D points.
        translation (torch.Tensor): Tensor of shape (B, 3) containing the 3D camera translation.
        focal_length (torch.Tensor): Tensor of shape (B, 2) containing the focal length in pixels.
        camera_center (torch.Tensor): Tensor of shape (B, 2) containing the camera center in pixels.
        rotation (torch.Tensor): Tensor of shape (B, 3, 3) containing the camera rotation.
    Returns:
        torch.Tensor: Tensor of shape (B, N, 2) containing the projection of the input points.
    """
    batch_size = points.shape[0]
    if rotation is None:
        rotation = torch.eye(3, device=points.device, dtype=points.dtype).unsqueeze(0).expand(batch_size, -1, -1)
    if camera_center is None:
        camera_center = torch.zeros(batch_size, 2, device=points.device, dtype=points.dtype)
    # Populate intrinsic camera matrix K.
    K = torch.zeros([batch_size, 3, 3], device=points.device, dtype=points.dtype)
    K[:,0,0] = focal_length[:,0]
    K[:,1,1] = focal_length[:,1]
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center
    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:,:,-1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]

class DeConvNet(nn.Module):
    def __init__(self, feat_dim=768, upscale=4):
        super(DeConvNet, self).__init__()
        self.first_conv = make_conv_layers([feat_dim, feat_dim//2], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.deconv = nn.ModuleList([])
        for i in range(int(math.log2(upscale))+1):
            if i==0:
                self.deconv.append(make_deconv_layers([feat_dim//2, feat_dim//4]))
            elif i==1:
                self.deconv.append(make_deconv_layers([feat_dim//2, feat_dim//4, feat_dim//8]))
            elif i==2:
                self.deconv.append(make_deconv_layers([feat_dim//2, feat_dim//4, feat_dim//8, feat_dim//8]))

    def forward(self, img_feat):
        
        face_img_feats = []
        img_feat = self.first_conv(img_feat)
        face_img_feats.append(img_feat)
        for i, deconv in enumerate(self.deconv):
            scale = 2**i
            img_feat_i = deconv(img_feat)
            face_img_feat = img_feat_i
            face_img_feats.append(face_img_feat)
        return face_img_feats[::-1]   # high resolution -> low resolution

class DeConvNet_v2(nn.Module):
    def __init__(self, feat_dim=768):
        super(DeConvNet_v2, self).__init__()
        self.first_conv = make_conv_layers([feat_dim, feat_dim//2], kernel=1, stride=1, padding=0, bnrelu_final=False)
        self.deconv = nn.Sequential(*[nn.ConvTranspose2d(in_channels=feat_dim//2, out_channels=feat_dim//4, kernel_size=4, stride=4, padding=0, output_padding=0, bias=False), 
                       nn.BatchNorm2d(feat_dim//4), 
                       nn.ReLU(inplace=True)])
    
    def forward(self, img_feat):
        
        face_img_feats = []
        img_feat = self.first_conv(img_feat)
        img_feat = self.deconv(img_feat) 
        
        return [img_feat]
        
class RefineNet(nn.Module):
    def __init__(self, cfg, feat_dim=1280, upscale=3):
        super(RefineNet, self).__init__()
        #self.deconv     = DeConvNet_v2(feat_dim=feat_dim) 
        #self.out_dim    = feat_dim//4
        
        self.deconv     = DeConvNet(feat_dim=feat_dim, upscale=upscale)
        self.out_dim    = feat_dim//8  + feat_dim//4 + feat_dim//2 
        self.dec_pose   = nn.Linear(self.out_dim, 96) 
        self.dec_cam    = nn.Linear(self.out_dim, 3)
        self.dec_shape  = nn.Linear(self.out_dim, 10)
        
        self.cfg        = cfg
        self.joint_rep_type = cfg.MODEL.MANO_HEAD.get('JOINT_REP', '6d')
        self.joint_rep_dim = {'6d': 6, 'aa': 3}[self.joint_rep_type]
        
    def forward(self, img_feat, verts_3d, pred_cam, pred_mano_feats, focal_length):
        B = img_feat.shape[0]
        
        img_feats = self.deconv(img_feat)
        
        img_feat_sizes = [img_feat.shape[2] for img_feat in img_feats] 
        
        temp_cams  = [torch.stack([pred_cam[:, 1], pred_cam[:, 2],  
                                  2*focal_length[:, 0]/(img_feat_size * pred_cam[:, 0] +1e-9)],dim=-1) for img_feat_size in img_feat_sizes] 

        verts_2d   = [perspective_projection(verts_3d,
                                translation=temp_cams[i],
                                focal_length=focal_length / img_feat_sizes[i]) for i in range(len(img_feat_sizes))]
        
        vert_feats = [sample_joint_features(img_feats[i], verts_2d[i]).max(1).values  for i in range(len(img_feat_sizes))] 

        vert_feats = torch.cat(vert_feats, dim=-1)

        delta_pose  = self.dec_pose(vert_feats)
        delta_betas = self.dec_shape(vert_feats)
        delta_cam   = self.dec_cam(vert_feats)

        
        pred_hand_pose = pred_mano_feats['hand_pose'] + delta_pose
        pred_betas     = pred_mano_feats['betas']     + delta_betas 
        pred_cam       = pred_mano_feats['cam']       + delta_cam

        joint_conversion_fn = {
                '6d': rot6d_to_rotmat,
                'aa': lambda x: aa_to_rotmat(x.view(-1, 3).contiguous())
            }[self.joint_rep_type]
 
        pred_hand_pose = joint_conversion_fn(pred_hand_pose).view(B, self.cfg.MANO.NUM_HAND_JOINTS+1, 3, 3)
        
        pred_mano_params = {'global_orient': pred_hand_pose[:, [0]],
                            'hand_pose': pred_hand_pose[:, 1:],
                            'betas': pred_betas}
        
        return  pred_mano_params, pred_cam
    
    