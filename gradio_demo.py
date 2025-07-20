import os 
import sys 
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["MESA_GL_VERSION_OVERRIDE"] = "4.1"
# os.system('pip install /home/user/app/pyrender')
# sys.path.append('/home/user/app/pyrender')

import gradio as gr
#import spaces
import cv2 
import numpy as np 
import torch 
from ultralytics import YOLO
from pathlib import Path
import argparse
import json
from typing import Dict, Optional

from wilor.models import WiLoR, load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from wilor.utils.renderer import Renderer, cam_crop_to_full
device = torch.device('cpu') if torch.cuda.is_available() else torch.device('cuda')

LIGHT_PURPLE=(0.25098039,  0.274117647,  0.65882353)

model, model_cfg = load_wilor(checkpoint_path = './pretrained_models/wilor_final.ckpt' , cfg_path= './pretrained_models/model_config.yaml')
# Setup the renderer
renderer = Renderer(model_cfg, faces=model.mano.faces)
model = model.to(device)
model.eval()

detector = YOLO(f'./pretrained_models/detector.pt').to(device)

def render_reconstruction(image, conf, IoU_threshold=0.3): 
    input_img, num_dets, reconstructions = run_wilow_model(image, conf, IoU_threshold=0.5)
    if num_dets> 0: 
    # Render front view
    
        misc_args = dict(
            mesh_base_color=LIGHT_PURPLE,
            scene_bg_color=(1, 1, 1),
            focal_length=reconstructions['focal'],
        )

        cam_view = renderer.render_rgba_multiple(reconstructions['verts'], 
                                                 cam_t=reconstructions['cam_t'], 
                                                 render_res=reconstructions['img_size'], 
                                                 is_right=reconstructions['right'], **misc_args)

        # Overlay image
        
        input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
        input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

        return input_img_overlay, f'{num_dets} hands detected'  
    else: 
        return input_img, f'{num_dets} hands detected' 

#@spaces.GPU()
def run_wilow_model(image, conf, IoU_threshold=0.5):
    img_cv2 = image[...,::-1]
    img_vis = image.copy()
    
    detections = detector(img_cv2, conf=conf, verbose=False, iou=IoU_threshold)[0]
    
    bboxes    = []
    is_right  = []
    for det in detections: 
        Bbox = det.boxes.data.cpu().detach().squeeze().numpy()
        Conf = det.boxes.conf.data.cpu().detach()[0].numpy().reshape(-1).astype(np.float16)
        Side = det.boxes.cls.data.cpu().detach()
        #Bbox[:2] -= np.int32(0.1 * Bbox[:2])
        #Bbox[2:] += np.int32(0.1 * Bbox[ 2:])
        is_right.append(det.boxes.cls.cpu().detach().squeeze().item())
        bboxes.append(Bbox[:4].tolist())
        
        color = (255*0.208, 255*0.647 ,255*0.603 ) if Side==0. else (255*1, 255*0.78039, 255*0.2353)
        label = f'L - {Conf[0]:.3f}' if Side==0 else f'R - {Conf[0]:.3f}'

        cv2.rectangle(img_vis, (int(Bbox[0]), int(Bbox[1])), (int(Bbox[2]), int(Bbox[3])), color , 3) 
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img_vis, (int(Bbox[0]), int(Bbox[1]) - 20), (int(Bbox[0]) + w, int(Bbox[1])), color, -1)
        cv2.putText(img_vis, label, (int(Bbox[0]), int(Bbox[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
          
    if len(bboxes) != 0: 
        boxes = np.stack(bboxes)
        right = np.stack(is_right)
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=2.0 )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    
        all_verts = []
        all_cam_t = []
        all_right = []
        all_joints= []
    
        for batch in dataloader: 
            batch = recursive_to(batch, device)
    
            with torch.no_grad():
                out = model(batch) 
                
            multiplier    = (2*batch['right']-1)
            pred_cam      = out['pred_cam']
            pred_cam[:,1] = multiplier*pred_cam[:,1]
            box_center    = batch["box_center"].float()
            box_size      = batch["box_size"].float()
            img_size      = batch["img_size"].float()
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full     = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                
                verts  = out['pred_vertices'][n].detach().cpu().numpy()
                joints = out['pred_keypoints_3d'][n].detach().cpu().numpy()
                
                is_right = batch['right'][n].cpu().numpy()
                verts[:,0] = (2*is_right-1)*verts[:,0]
                joints[:,0] = (2*is_right-1)*joints[:,0]
                
                cam_t = pred_cam_t_full[n]
                
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right)
                all_joints.append(joints)

        reconstructions = {'verts': all_verts, 'cam_t': all_cam_t, 'right': all_right, 'img_size': img_size[n], 'focal': scaled_focal_length}
        return img_vis.astype(np.float32)/255.0, len(detections), reconstructions
    else: 
        return img_vis.astype(np.float32)/255.0, len(detections), None       



header = ('''
<div class="embed_hidden" style="text-align: center;">
    <h1> <b>WiLoR</b>: End-to-end 3D hand localization and reconstruction in-the-wild</h1>
    <h3>
        <a href="https://rolpotamias.github.io" target="_blank" rel="noopener noreferrer">Rolandos Alexandros Potamias</a><sup>1</sup>,
        <a href="" target="_blank" rel="noopener noreferrer">Jinglei Zhang</a><sup>2</sup>,
        <br>
        <a href="https://jiankangdeng.github.io/" target="_blank" rel="noopener noreferrer">Jiankang Deng</a><sup>1</sup>,
        <a href="https://wp.doc.ic.ac.uk/szafeiri/" target="_blank" rel="noopener noreferrer">Stefanos Zafeiriou</a><sup>1</sup>
    </h3>
    <h3>
        <sup>1</sup>Imperial College London;
        <sup>2</sup>Shanghai Jiao Tong University
    </h3>
</div>
<div style="display:flex; gap: 0.3rem; justify-content: center; align-items: center;" align="center">
<a href=''><img src='https://img.shields.io/badge/Arxiv-......-A42C25?style=flat&logo=arXiv&logoColor=A42C25'></a> 
<a href='https://rolpotamias.github.io/pdfs/WiLoR.pdf'><img src='https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow'></a> 
<a href='https://rolpotamias.github.io/WiLoR/'><img src='https://img.shields.io/badge/Project-Page-%23df5b46?style=flat&logo=Google%20chrome&logoColor=%23df5b46'></a> 
<a href='https://github.com/rolpotamias/WiLoR'><img src='https://img.shields.io/badge/GitHub-Code-black?style=flat&logo=github&logoColor=white'></a> 
''')


with gr.Blocks(title="WiLoR: End-to-end 3D hand localization and reconstruction in-the-wild", css=".gradio-container") as demo:

    gr.Markdown(header)

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input image", type="numpy")
            threshold = gr.Slider(value=0.3, minimum=0.05, maximum=0.95, step=0.05, label='Detection Confidence Threshold')
            #nms = gr.Slider(value=0.5, minimum=0.05, maximum=0.95, step=0.05, label='IoU NMS Threshold')
            submit = gr.Button("Submit", variant="primary")
        
        
        with gr.Column():
            reconstruction = gr.Image(label="Reconstructions", type="numpy")
            hands_detected = gr.Textbox(label="Hands Detected")
    
        submit.click(fn=render_reconstruction, inputs=[input_image, threshold], outputs=[reconstruction, hands_detected])

    with gr.Row():
        example_images = gr.Examples([

            ['./demo_img/test1.jpg'], 
            ['./demo_img/test2.png'], 
            ['./demo_img/test3.jpg'], 
            ['./demo_img/test4.jpg'],
            ['./demo_img/test5.jpeg'],
            ['./demo_img/test6.jpg'], 
            ['./demo_img/test7.jpg'],
            ['./demo_img/test8.jpg'], 
            ], 
            inputs=input_image)

demo.launch()
