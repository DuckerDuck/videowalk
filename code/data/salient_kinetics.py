import torchvision.datasets.video_utils

from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.vision import VisionDataset
from torch import Tensor
from .kinetics import Kinetics400
from pathlib import Path
from data.saliency.methods import gbvs_from_video

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
                 extensions=('mp4',), transform=None, cached=None, _precomputed_metadata=None):
        super(SalientKinetics400, self).__init__(root, frames_per_clip, 
                                                step_between_clips=step_between_clips,
                                                frame_rate=frame_rate, extensions=extensions, 
                                                transform=transform, cached=cached, 
                                                _precomputed_metadata=_precomputed_metadata)
 
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
        # TODO: implement
        print('init_from_cache not implemented')
        return {}

    def generate_saliency(self, video: Tensor, idx: int):
        """Generate saliency map for given video clip, will overwrite if 
        files already exist. Transform should be applied before passing to this function.

        Args:
            video (Tensor): The video from which to generate saliency maps.
            idx (int): Index into the VideoClip object from Kinetics400 dataset.
        """
        video_path = Path(self.video_clips.metadata['video_paths'][idx])
        video_name = video_path.stem

        destination_path = self.salient_root / video_name

        if not destination_path.is_dir():
            destination_path.mkdir()

        saliency = gbvs_from_video(video, destination_path)
        self.saliency_maps[idx] = saliency

        return saliency


    def __getitem__(self, idx):
        success = False
        while not success:
            try:
                video, audio, info, video_idx = self.video_clips.get_clip(idx)
                success = True
            except:
                print('skipped idx', idx)
                idx = np.random.randint(self.__len__())

        if idx in self.saliency_maps:
            saliency = self.saliency_maps[idx]
        else:
            saliency = self.generate_saliency(video, video_idx)

        label = self.samples[video_idx][1]
        if self.transform is not None:
            # video: tuple of (train_tranform, plain_transform)
            video = self.transform(video)
            saliency = self.transform(saliency)
        return video, audio, saliency, label
