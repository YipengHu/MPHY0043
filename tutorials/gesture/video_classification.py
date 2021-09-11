
import torch
import torchvision
from PIL import Image

from loader import SimpleVideoDataset as VideoDataset


video_folder = "data/Knot_Tying/video"
meta_filepath = "data/Knot_Tying/meta_file_Knot_Tying.txt"

## dataset loader
video_label_dataset = VideoDataset(
     video_dir=video_folder,
     num_frames=50,
     frame_size=(160,120),
     capture_type=1,  #  {0:'both', 1:'_capture1', 2:'_capture2'}
     meta_file=meta_filepath,
     pre_load=False)

# example video
video, label = video_label_dataset[10]
im = [Image.fromarray(video[i,...,0:3]) for i in range(video.shape[0])]
im[0].save('test_video.gif',save_all=True,append_images=im[1:])

## CNN network

## training

## prediction
