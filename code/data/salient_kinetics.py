from utils.augs import get_resized_transform
import torchvision.datasets.video_utils

from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.vision import VisionDataset
from torchvision.utils import save_image
from torchvision.transforms.functional import to_tensor
from PIL import Image
from torch import Tensor
import torch
from .kinetics import Kinetics400
from pathlib import Path
from data.saliency.methods import gbvs_from_frame, itti_from_frame, harris_from_frame, mbs_from_frame
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
         

    def generate_saliency(self, frame: Tensor):
        """Generate saliency map for given frame

        Args:
            frame (Tensor): The frame from which to generate saliency maps.
        """
        # TODO: logic for switching method
        # saliency = gbvs_from_video(video)
        # saliency = harris_from_frame(video)

        method = harris_from_frame

        if self.rescale < 1:
            transform = get_resized_transform(method, self.rescale)
            saliency = transform(frame)
        else:
            saliency = method(frame)
        
        return saliency

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

    def save_frame(self, frame: Tensor, path: Path):
        if torch.max(frame) < 2:
            frame *= 255

        frame = frame.numpy().astype(np.uint8)

        with open(str(path), 'w') as f:
            img = Image.fromarray(frame)
            img.save(f, format='jpeg')

    def get_saliency_clip(self, clip: Tensor, clip_location: Tuple[int, int]) -> Tensor:
        """
        Get (precomputed) saliency clip
        """
        video_idx, clip_idx = clip_location

        video_path = self.video_clips.metadata['video_paths'][video_idx]
        video_path = Path(video_path)
        video_name = video_path.stem

        # Maintain folder structure or original dataset
        subfolders = video_path.relative_to(self.root).parent
        
        frames = self.clip_idx_to_frame(clip_location)

        saliencies = []
        for frame_in_clip, frame in enumerate(frames):
            cached_folder = self.salient_root / subfolders / video_name
            cached_file = cached_folder / f'{frame}.jpg'

            if cached_file.is_file():
                saliency_frame = self.load_frame(cached_file)
            else:
                # print(f'Generating saliency for video {video_name} frame {frame}')
                saliency_frame = self.generate_saliency(clip[frame_in_clip])

                if not cached_folder.is_dir():
                    cached_folder.mkdir(parents=True)
                 
                self.save_frame(saliency_frame, cached_file)


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
