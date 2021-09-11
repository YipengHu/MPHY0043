import os

import torch
import numpy as np
from PIL import Image

from loader import SimpleVideoDataset as VideoDataset
from network import CNN3D


os.environ["CUDA_VISIBLE_DEVICES"]="0"
use_cuda = torch.cuda.is_available()

video_folder = "data/Knot_Tying/video"
meta_filepath = "data/Knot_Tying/meta_file_Knot_Tying.txt"

## dataset loader
video_label_dataset = VideoDataset(
     video_dir=video_folder,
     num_frames=50,
     frame_size=(160,120),
     capture_type=1,  #  {0:'both', 1:'_capture1', 2:'_capture2'}
     meta_file=meta_filepath,
     pre_load=True)

# example video
video, label = video_label_dataset[10]
im = [Image.fromarray(np.transpose(video[0:3,i,...]*255,[2,1,0]).astype(np.uint8)) for i in range(video.shape[1])]
im[0].save('test_video.gif',save_all=True,append_images=im[1:])

# split train-test
train_size = round(0.8 * len(video_label_dataset))
test_size = len(video_label_dataset) - train_size
train_set, test_set = torch.utils.data.random_split(video_label_dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=2, 
    shuffle=True,
    num_workers=2)

"""
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=1, 
    shuffle=False,
    num_workers=1)
"""

del video_label_dataset  # for memory


## 3D CNN network
cnn_3d = CNN3D(video.shape[0],3)
if use_cuda:
    cnn_3d.cuda()


## training
criterion = torch.nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.Adam(cnn_3d.parameters(), lr=1e-4)

freq_print = 10
for epoch in range(200):
    for step, (videos, labels) in enumerate(train_loader):
        if use_cuda:
            videos, labels = videos.cuda(), labels.cuda()

        optimizer.zero_grad()
        outputs = cnn_3d(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Compute and print loss
        if step % freq_print == (freq_print-1):    # print every 20 mini-batches
            print('[Epoch %d, iter %05d] loss: %.3f' % (epoch, step, loss.item()))

print('Training done.')


## save trained model
torch.save(cnn_3d, os.path.join('saved_model'))  # https://pytorch.org/tutorials/beginner/saving_loading_models.html
print('Model saved.')


## prediction
""" Q: How to load the model back and predict using the test_set/test_loader """
