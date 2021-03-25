from torchvision.transforms.functional import to_tensor
from PIL import Image
from torch import Tensor
import torch
from .kinetics import Kinetics400
from pathlib import Path
from typing import Tuple, List

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
                 extensions=('mp4',), transform=None, salient_transform=None, rescale=1, 
                 cached=None, _precomputed_metadata=None):
        super(SalientKinetics400, self).__init__(root, frames_per_clip, 
                                                step_between_clips=step_between_clips,
                                                frame_rate=frame_rate, extensions=extensions, 
                                                transform=transform, cached=cached, 
                                                _precomputed_metadata=_precomputed_metadata)

        self.salient_transform = salient_transform
        self.rescale = rescale
        self.salient_root = Path(salient_root)
        if not self.salient_root.is_dir():
            # No salient cache available, create new one
            self.salient_root.mkdir()

    def clip_idx_to_frame(self, clip_location: Tuple[int, int]) -> List:
        video_idx, clip_idx = clip_location

        video_pts = self.video_clips.metadata['video_pts'][video_idx]
        clip_pts = self.video_clips.clips[video_idx][clip_idx]

        # Find specific frame
        # clip_length = clip.shape[0]
        # start_frame = (clip_idx - 1) * clip_length
        # frame_idx = video_pts == clip_pts[0]
        # assert start_frame == frame_idx.nonzero(as_tuple=True)[0]
        
        # Map video_pts values to indices, theses indices are the frame ids
        to_frame = { pts.item(): i for i, pts in enumerate(video_pts) }
        frames = [to_frame[pts.item()] for pts in clip_pts]
        return frames

    def load_frame(self, path: Path) -> Tensor:
        with open(str(path), 'rb') as f:
            img = Image.open(f)
            img = img.convert('L')
        img = to_tensor(img)
        
        if (torch.max(img) < 2):
            img *= 255
        return img.squeeze()


    def get_saliency_clip(self, clip: Tensor, clip_location: Tuple[int, int]) -> Tensor:
        """
        Get (precomputed) saliency clip
        """
        video_idx, clip_idx = clip_location

        video_path = self.video_clips.metadata['video_paths'][video_idx]
        video_path = Path(video_path)
        video_name = video_path.stem

        # Maintain folder structure of original dataset
        subfolders = video_path.relative_to(Path(self.root).parent).parent
        
        frames = self.clip_idx_to_frame(clip_location)
        cached_folder = self.salient_root / subfolders / video_name
        
        saliencies = []
        for frame in frames:
            cached_file = cached_folder / f'{frame}.jpg'

            if cached_file.is_file():
                saliency_frame = self.load_frame(cached_file)
            else:
                raise Exception(f'Could not load frame {cached_file} from video {video_idx}.')

            saliencies.append(saliency_frame.byte())
        return torch.stack(saliencies)


    def __getitem__(self, idx):
        success = False
        while not success:
            try:
                video, audio, info, video_idx = self.video_clips.get_clip(idx)

                # This information is needed for saliency caching
                clip_location = self.video_clips.get_clip_location(idx)
                # saliency = self.get_saliency_clip(video, clip_location)
                success = True
            except:
                print('skipped idx', idx)
                idx = np.random.randint(self.__len__())
        
        saliency = self.get_saliency_clip(video, clip_location)
        label = self.samples[video_idx][1]

        # The random state is kept constant for the two transforms, this
        # makes sure RandomResizedCrop is applied the same way in both 
        # video and saliency maps.
        random_state = torch.get_rng_state()

        if self.transform is not None:
            video = self.transform(video)

        if self.salient_transform is not None:
            torch.set_rng_state(random_state)
            saliency = self.salient_transform(saliency)

        return video, audio, saliency, label
