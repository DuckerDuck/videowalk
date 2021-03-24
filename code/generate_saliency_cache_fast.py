import argparse
import torch
from torch.utils.data import Dataset
from typing import List
from pathlib import Path
import hashlib
import ffmpeg
import docker
from tqdm import tqdm


class VideoDataset(Dataset):
    """Dataset for looping through videos in folder"""

    def __init__(self, folder: Path, extension='mp4', transform=None):
        self.root = folder
        self.extension = extension
        self.videos = self.get_video_list()
        self.transform = transform

    def get_video_list(self) -> List[Path]:
        return list(self.root.glob(f'**/*.{self.extension}'))

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        return self.videos[index]


def to_saliency(video: Path, args):
    # Check where to store temporary files
    tmp_dir = Path('/scratch/')
    if not tmp_dir.exists():
        tmp_dir = Path('./.tmp_videos')
        if not tmp_dir.exists():
            tmp_dir.mkdir()
    
    # Create temporary folder to store image sequence
    folder_name = hashlib.sha224(str(video).encode()).hexdigest()
    input_path = tmp_dir / (folder_name + '_in')
    input_path.mkdir()

    # Convert video file to image sequence
    (
        ffmpeg
        .input(str(video))
        .output(str(input_path / '%01d.jpg'))
        .run(quiet=True)
    )
    print('Converted video', video.name)

    # Setup saliency destination directory
    root = Path(args.data_path)
    subfolders = video.relative_to(root).parent
    output_path = Path(args.saliency_path) / subfolders
    output_path.mkdir(parents=True)
    
    # Setup docker connection
    client = docker.from_env()
    volumes = {
        str(input_path.resolve()): {'bind': '/MBS/input', 'mode': 'ro'},
        str(output_path.resolve()): {'bind': '/MBS/output', 'mode': 'rw'}
    }
    client.containers.run('mbs:latest', 'octave process_folder.m', volumes=volumes)

    # Remove temporary files
    for image in input_path.glob('*.jpg'):
        image.unlink()
    input_path.rmdir()

def no_collate(input):
    return input

def generate(args):
    dataset = VideoDataset(Path(args.data_path))
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=None, num_workers=args.workers//2,
        pin_memory=True, collate_fn=no_collate)

    print(f'Total number of videos: {len(dataset)}')

    data_generator = enumerate(data_loader)
    pbar = tqdm(total=len(data_loader))
    i = 0
    while True:
        pbar.update(1)
        try:
            i, videos = next(data_generator)
            for video in videos:
                to_saliency(video, args)
        except StopIteration:
            break
        except Exception as e:
            print('skipped video clip', i, str(e))
            return

    pbar.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Saliency Map Cache Generator')

    parser.add_argument('--data-path', default='/../kinetics/', help='Path to dataset')
    parser.add_argument('--saliency-path', default='./saliency_cache/',
        help='Path to saliency cache')
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('-rs', '--rescale', default=1, type=float, 
    help='How much to scale video frames before computing saliency')

    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
 
    args = parser.parse_args()

    generate(args)
