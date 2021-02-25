from .gbvs.gbvs import compute_saliency
from torchvision.utils import save_image
from pathlib import Path
from torch import Tensor, from_numpy, stack

def gbvs_from_video(video: Tensor, target: Path) -> Tensor:
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
        salience = from_numpy(compute_saliency(frame))
        save_image(salience, target / f'{f}.png', normalize=True)
        saliencies.append(salience)

    return stack(saliencies)
