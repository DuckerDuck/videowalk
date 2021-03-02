import torchvision.datasets.video_utils

from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.vision import VisionDataset
from torch import Tensor
import torch
from .kinetics import Kinetics400
from pathlib import Path
from data.saliency.methods import gbvs_from_video, itti_from_video
from typing import Tuple

import numpy as np

class SalientKinetics400(Kinetics400):
    """
    Args:
        root (string): Root directory of the Kinetics-400 Dataset.
        frames_per_clip (int): number of frames in a clip
        step_between_clips (int): number of frames between each clip
        transform (callable, optional): A function/transform that  takes in a TxHxWxC video
            and returns a transformed version.

    Returns:
        video (Tensor[T, H, W, C]): the `T` video frames
        audio(Tensor[K, L]): the audio frames, where `K` is the number of channels
            and `L` is the number of points
        label (int): class of the video clip
    """

    def __init__(self, root, salient_root, frames_per_clip, step_between_clips=1, frame_rate=None,
                 extensions=('mp4',), transform=None, salient_transform=None, 
                 cached=None, _precomputed_metadata=None):
        super(SalientKinetics400, self).__init__(root, frames_per_clip, 
                                                step_between_clips=step_between_clips,
                                                frame_rate=frame_rate, extensions=extensions, 
                                                transform=transform, cached=cached, 
                                                _precomputed_metadata=_precomputed_metadata)

        self.salient_transform = salient_transform
        self.salient_root = Path(salient_root)
        if not self.salient_root.is_dir():
            # No salient cache available, create new one
            self.salient_root.mkdir()
            self.saliency_maps = {}
        else:
            self.saliency_maps = self.init_from_cache()
 
    def init_from_cache(self):
        """
        Initializes saliency_maps from existing cache.
        """
#        for video_dir in self.salient_root:
#            pass

        return {}

    def generate_saliency(self, video: Tensor):
        """Generate saliency map for given video clip, will overwrite if 
        files already exist.

        Args:
            video (Tensor): The video from which to generate saliency maps.
            idx (int): Index into the VideoClip object from Kinetics400 dataset.
        """
        # TODO: logic for switching method
        # saliency = gbvs_from_video(video)
        saliency = itti_from_video(video)


        return saliency

    def get_saliency_clip(self, clip: Tensor, clip_location: Tuple[int, int]) -> Tensor:
        """
        Get (precomputed) saliency clip
        """
        video_idx, clip_idx = clip_location

        video_path = self.video_clips.metadata['video_paths'][video_idx]
        video_name = Path(video_path).stem
        
        cached_path = self.salient_root / video_name / f'{clip_idx}.pt'

        if cached_path.is_file():
            saliency = torch.load(cached_path)
        else:
            print('Generating saliency for ', video_name)
            saliency = self.generate_saliency(clip)
            if not (self.salient_root / video_name).is_dir():
                (self.salient_root / video_name).mkdir()
            torch.save(saliency, cached_path)

        return saliency


    def __getitem__(self, idx):
        success = False
        while not success:
            try:
                video, audio, info, video_idx = self.video_clips.get_clip(idx)

                # This information is needed for saliency caching
                clip_location = self.video_clips.get_clip_location(idx)
                saliency = self.get_saliency_clip(video, clip_location)
                success = True
            except:
                print('skipped idx', idx)
                idx = np.random.randint(self.__len__())

        label = self.samples[video_idx][1]
        if self.transform is not None:
            video = self.transform(video)

        if self.salient_transform is not None:
            saliency = self.salient_transform(saliency)

        return video, audio, saliency, label
