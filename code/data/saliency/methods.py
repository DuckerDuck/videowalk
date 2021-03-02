from .gbvs.gbvs import compute_saliency
from torchvision.utils import save_image
from pathlib import Path
from typing import Optional
import torch

def gbvs_from_video(video: torch.Tensor, target: Optional[Path] = None) -> torch.Tensor:
    """
    Generate saliency maps using Graph Based Visual Saliency algorithm:
    @article{harel2007graph,
      title={Graph-based visual saliency},
      author={Harel, Jonathan and Koch, Christof and Perona, Pietro},
      year={2007},
      publisher={MIT Press}
    }
    Args:
        video (Tensor): Video split into patches of size (frames, N*channels, height, width)
        target (Path): Location on disk to save the frames
    """
    video = video.numpy()
    frames, height, width, channels = video.shape

    saliencies = []

    for f in range(frames):
        frame = video[f, :, :, :]
        salience = torch.from_numpy(compute_saliency(frame))
        if target is not None:
            save_image(salience, target / f'{f}.png', normalize=True)
        saliencies.append(salience.byte())

    return torch.stack(saliencies)
