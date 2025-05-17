import os
import json
import numpy as np
import argparse
from pytubefix import YouTube

parser = argparse.ArgumentParser()

parser.add_argument("--root", type=str, help="Directory of WiLoR")
parser.add_argument("--mode", type=str, choices=['train', 'test'], default= 'train', help="Train/Test set")

args = parser.parse_args()

with open(os.path.join(args.root, f'./whim/{args.mode}_video_ids.json')) as f:
    video_dict = json.load(f)

Video_IDs = video_dict.keys()
failed_IDs = []
os.makedirs(os.path.join(args.root, 'Videos'), exist_ok=True)  

for Video_ID in Video_IDs:
    res = video_dict[Video_ID]['res'][0]
    try:
        YouTube('https://youtu.be/'+Video_ID).streams.filter(only_video=True, 
                                                             file_extension='mp4', 
                                                             res =f'{res}p'
                                                             ).order_by('resolution').desc().first().download(
                                                             output_path=os.path.join(args.root, 'Videos') , 
                                                             filename = Video_ID +'.mp4')
    except:
        print(f'Failed {Video_ID}')
        failed_IDs.append(Video_ID)
        continue
        
        
    cap = cv2.VideoCapture(os.path.join(args.root, 'Videos', Video_ID + '.mp4'))
    if (cap.isOpened()== False): 
        print(f"Error opening video stream {os.path.join(args.root, 'Videos', Video_ID + '.mp4')}")

    VIDEO_LEN = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    
    fps_org = video_dict[Video_ID]['fps']
    fps_rate = round(fps / fps_org)
    
    all_frames   = os.listdir(os.path.join(args.root, 'WHIM', args.mode, 'anno', Video_ID))
    
    for frame in all_frames: 
        frame_gt = int(frame[:-4])
        frame_idx = (frame_gt * fps_rate)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, img_cv2 = cap.read()
    
        cv2.imwrite(os.path.join(args.root, 'WHIM', args.mode, 'anno', Video_ID, frame +'.jpg' ), img_cv2.astype(np.float32))
                              
np.save(os.path.join(args.root, 'failed_videos.npy'), failed_IDs)  
