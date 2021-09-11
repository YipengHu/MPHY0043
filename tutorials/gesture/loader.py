import os

import av
import numpy as np


class SimpleVideoDataset():
    def __init__(self, video_dir, num_frames=None, frame_size=None, capture_type=0, meta_file=None, pre_load=False):
        self.num_frames=num_frames
        self.frame_size=frame_size
        # check video files
        filenames = sorted([f for f in os.listdir(video_dir) if f.endswith(".avi")])  # this is necessary for capture_type=0
        self.ids = list({f.split('_')[-2] for f in filenames})
        # read meta file
        if meta_file is not None:
            skill_dict = {"N":0, "I":1, "E":2} # see details in readme.txt
            with open(meta_file,"r") as fid:
                self.labels = {line.split('\t')[0].split('_')[-1]:skill_dict[line.split('\t')[2]] for line in fid.readlines() if len(line)>2}
            
            if set(self.ids) != set(self.labels.keys()):
                raise('Inconsistent video and meta files.')
        # save all video filenames or preload all videos
        
        _str = {0:'_capture', 1:'_capture1', 2:'_capture2'}[capture_type]
        self.video_filenames = {id: [os.path.join(video_dir,fn) for fn in filenames if id+_str in fn] for id in self.ids}
        if pre_load:
            print('Decode and pre-load all videos...')
            self.preloaded_videos = {id: np.concatenate([self._read_video_to_images(fn,num_frames,frame_size) for fn in self.video_filenames[id]],axis=3) for id in self.ids}
            print('Done.')
        else:
            self.preloaded_videos = None


    def __len__(self):
        return len(self.ids)


    def __getitem__(self, idx):
        id = self.ids[idx]
        if self.preloaded_videos is None:
            video = np.concatenate([self._read_video_to_images(fn,self.num_frames,self.frame_size) for fn in self.video_filenames[id]],axis=3)
        else:
            video = self.preloaded_videos[id]
        return np.transpose(video,[3,0,2,1]).astype(np.float32)/255, self.labels[id]  # for torch [C,D,H,W]

    
    def _read_video_to_images(self,filename,num_frames=None,size=None,video_id=0):
        """
        when num_frames<0, use the first num_frames frames without resampling
        """
        with av.open(filename,mode='r') as container:
            total_num_frames = container.streams.video[video_id].frames
            frames = list(container.decode(video=video_id))

        if num_frames is None:
            num_frames = -total_num_frames 
        if abs(num_frames)>total_num_frames:
            num_frames = abs(num_frames)
            print('\n\nWARNING: num_frames > total_num_frames. Replicated frames returned.\n\n')

        if num_frames > 0:
            frame_indices = [round(total_num_frames/num_frames*i) for i in range(num_frames)]
        else:
            frame_indices = [i for i in range(-num_frames)]  # first num_frames frames
        
        images = [frames[i].to_image() for i in frame_indices]
        if size is not None:
            images = map(lambda im: im.resize(size), images)

        return np.stack([np.array(im) for im in images],0)
