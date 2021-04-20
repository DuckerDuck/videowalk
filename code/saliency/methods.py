from .gbvs.gbvs import compute_saliency as compute_gbvs
from .gbvs.ittikochneibur import compute_saliency as compute_itti
from torchvision.transforms.functional import to_tensor
from skimage.color import rgb2gray
from torchvision.utils import save_image
from pathlib import Path
from PIL import Image
from typing import Optional
import torch
import numpy as np
import cv2
import docker
import hashlib

# method_from_frame should return tensor of shape (height, width) in range 0 - 255

def mbs_from_folder(input_path: Path, output_path: Path):

    # Setup docker connection
    client = docker.from_env()
    volumes = {
        str(input_path.resolve()): {'bind': '/MBS/input', 'mode': 'ro'},
        str(output_path.resolve()): {'bind': '/MBS/output', 'mode': 'rw'}
    }
    result = client.containers.run('mbs:latest', 'octave process_folder.m', volumes=volumes)
    return result


def gbvs_from_frame(frame: torch.Tensor) -> torch.Tensor:
    frame = frame.numpy()
    salience = torch.from_numpy(compute_gbvs(frame))
    return salience

def itti_from_frame(frame: torch.Tensor) -> torch.Tensor:
    frame = frame.numpy()
    salience = torch.from_numpy(compute_itti(frame))
    return salience

def harris_from_frame(frame: torch.Tensor) -> torch.Tensor:
    frame = rgb2gray(frame.numpy())
    frame = np.float32(frame)
    corners = corners = cv2.cornerHarris(frame, 3, 7, 0.02)
    return torch.from_numpy(corners)

def optical_flow_from_frames(frame_a: torch.Tensor, frame_b: torch.Tensor) -> torch.Tensor:
    frame_a = rgb2gray(frame_a.numpy())
    frame_b = rgb2gray(frame_b.numpy())
    
    flow = cv2.calcOpticalFlowFarneback(frame_a, frame_a, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    
    # Remove camera motion
    v = v - np.mean(v)
    u = u - np.mean(u)

    mag = np.sqrt(u ** 2 + v ** 2)
    norm = (mag - np.min(mag)) / np.max(mag)
    return norm

def _method_from_video(video: torch.Tensor, method, target: Optional[Path] = None) -> torch.Tensor:
    frames, height, width, channels = video.shape

    saliencies = []

    for f in range(frames):
        frame = video[f, :, :, :]
        salience = method(frame)

        if target is not None:
            save_image(salience, target / f'{f}.png', normalize=True)

        saliencies.append(salience)

    return torch.stack(saliencies)

def gbvs_from_video(video: torch.Tensor, target: Optional[Path] = None) -> torch.Tensor:
    return _method_from_video(video, gbvs_from_frame, target)

def itti_from_video(video: torch.Tensor, target: Optional[Path] = None) -> torch.Tensor:
    return _method_from_video(video, itti_from_frame, target)

def harris_from_video(video: torch.Tensor, target: Optional[Path] = None) -> torch.Tensor:
    return _method_from_video(video, harris_from_frame, target)
