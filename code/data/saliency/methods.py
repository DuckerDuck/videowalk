from .gbvs.gbvs import compute_saliency as compute_gbvs
from .gbvs.ittikochneibur import compute_saliency as compute_itti
from torchvision.utils import save_image
from pathlib import Path
from typing import Optional
import torch

def gbvs_from_frame(frame: torch.Tensor) -> torch.Tensor:
    frame = frame.numpy()
    salience = torch.from_numpy(compute_gbvs(frame))
    return salience

def itti_from_frame(frame: torch.Tensor) -> torch.Tensor:
    frame = frame.numpy()
    salience = torch.from_numpy(compute_itti(frame))
    return salience

def _method_from_video(video: torch.Tensor, method, target: Optional[Path] = None) -> torch.Tensor:
    frames, height, width, channels = video.shape

    saliencies = []

    for f in range(frames):
        frame = video[f, :, :, :]
        salience = method(frame)

        if target is not None:
            save_image(salience, target / f'{f}.png', normalize=True)

        saliencies.append(salience.byte())

    return torch.stack(saliencies)

def gbvs_from_video(video: torch.Tensor, target: Optional[Path] = None) -> torch.Tensor:
    return _method_from_video(video, gbvs_from_frame, target)

def itti_from_video(video: torch.Tensor, target: Optional[Path] = None) -> torch.Tensor:
    return _method_from_video(video, itti_from_frame, target)


