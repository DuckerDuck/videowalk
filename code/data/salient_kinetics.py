from torchvision.transforms.functional import to_tensor
from PIL import Image
from torch import Tensor
import torch
from .kinetics import Kinetics400
from pathlib import Path
from typing import Tuple, List
from saliency.flow.optflow import flow_read
from typing import Tuple, List, Optional

import numpy as np

class SalientKinetics400(Kinetics400):
    """
    Args:
        root (string): Root directory of the Kinetics-400 Dataset.
        prior_root (string, optional): Root directory of the Prior Dataset, 
            if None generate a simple saliency map
        frames_per_clip (int): number of frames in a clip
        step_between_clips (int): number of frames between each clip
        transform (callable, optional): A function/transform that  takes in a TxHxWxC video
            and returns a transformed version.

    Returns:
        video (Tensor[T, H, W, C]): the `T` video frames
        audio (Tensor[K, L]): the audio frames, where `K` is the number of channels
            and `L` is the number of points
        saliency (Tensor[T, H, W, C]): Saliency information of video
        label (int): class of the video clip
    """

    def __init__(self, root, prior_root: Optional[str], frames_per_clip, step_between_clips=1, frame_rate=None,
                 extensions=('mp4',), transform=None, salient_transform=None, 
                 cached=None, _precomputed_metadata=None, frame_offset=0, saliency_channels=1):
        super(SalientKinetics400, self).__init__(root, frames_per_clip, 
                                                step_between_clips=step_between_clips,
                                                frame_rate=frame_rate, extensions=extensions, 
                                                transform=transform, cached=cached, 
                                                _precomputed_metadata=_precomputed_metadata)

        self.salient_transform = salient_transform
        # Frame offset can be used if a saliency method uses 1-indexing
        self.frame_offset = frame_offset

        # Saliency maps are grayscale (1 channel) and optical flow contains Fx and Fy  (2 channels)
        self.saliency_channels = saliency_channels
        
        if prior_root is None:
            self.prior_root = None
        else:
            self.prior_root = Path(prior_root)
            if not self.prior_root.is_dir():
                raise FileNotFoundError(f'Could not find saliency data at {self.prior_root}')

    def clip_idx_to_frame(self, clip_location: Tuple[int, int]) -> List:
        video_idx, clip_idx = clip_location

        video_pts = self.video_clips.metadata['video_pts'][video_idx]
        clip_pts = self.video_clips.clips[video_idx][clip_idx]
        
        # Map video_pts values to indices, theses indices are the frame ids
        to_frame = { pts.item(): i for i, pts in enumerate(video_pts) }
        frames = [to_frame[pts.item()] for pts in clip_pts]
        return frames

    def load_optical_flow_frame(self, path: Path) -> Tensor:
        flow = flow_read(str(path))
        return to_tensor(flow)

    def load_frame(self, path: Path) -> Tensor:
        with open(str(path), 'rb') as f:
            img = Image.open(f)
            if self.saliency_channels == 1:
                img = img.convert('L')
            elif self.saliency_channels == 2:
                img = img.convert('RGB')

        img = to_tensor(img)

        if self.saliency_channels == 2:
            # Discard B channel, img shape: (h, w, 2)
            img = img[:2, :, :].permute(1, 2, 0)

        # TODO: check if this code can be removed
        if torch.max(img) < 2 and self.saliency_channels != 2:
            img *= 255
        return img.squeeze()

    def generate_saliency_clip(self, clip: Tensor) -> Tensor:
        """ Used to generate simple saliency maps """
        T, W, H, C = clip.shape
        frame = torch.cartesian_prod(torch.arange(W), torch.arange(H))
        frame = frame.reshape(W, H, 2).float()
        center = torch.tensor([W/2, H/2]).unsqueeze(0)
        distance = torch.cdist(frame, center).squeeze()

        distance -= distance.min()
        distance /= distance.max()
        distance = (1 - distance) * 255
        distance = distance.byte()

        return torch.stack(T * [distance])


    def get_saliency_clip(self, clip_location: Tuple[int, int]) -> Tensor:
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
        cached_folder = self.prior_root / subfolders / video_name
        
        saliencies = []
        for frame in frames:
            cached_file = cached_folder / f'{frame + self.frame_offset}.jpg'
            
            if self.saliency_channels == 2:
                cached_file = cached_file.with_suffix('.flo')

            if cached_file.is_file():
                if self.saliency_channels == 2:
                    saliency_frame = self.load_optical_flow_frame(cached_file)
                else:
                    saliency_frame = self.load_frame(cached_file)
            else:
                raise Exception(f'Could not load frame {cached_file} from video {video_idx}.')

            saliencies.append(saliency_frame)
        return torch.stack(saliencies)


    def __getitem__(self, idx):
        success = False
        while not success:
            try:
                video, audio, info, video_idx = self.video_clips.get_clip(idx)

                # This information is needed for saliency caching
                clip_location = self.video_clips.get_clip_location(idx)
                if self.prior_root is None:
                    saliency = self.generate_saliency_clip(video)
                else:
                    saliency = self.get_saliency_clip(clip_location)
                
                success = True
            except Exception as e:
                print('skipped idx', idx, e)
                idx = np.random.randint(self.__len__())
        
        # saliency = self.get_saliency_clip(video, clip_location)
        label = self.samples[video_idx][1]

        # Scale saliency to size of video if they are not the same
        if saliency.shape[-2:] != video.shape[1:3]:
            correct_size = (video.shape[1], video.shape[2])
            saliency = torch.nn.functional.interpolate(saliency, size=correct_size)

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
