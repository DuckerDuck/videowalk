import argparse
import torch
import hashlib
import ffmpeg
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from typing import List, Callable, Tuple
from pathlib import Path
from tqdm import tqdm

from saliency.methods import *
from saliency.flow.optflow import flow_write, flow_write_as_png


# Method name: (callable, is docker method)
method_index = {
    'harris': (harris_from_video, False),
    'gbvs': (gbvs_from_video, False),
    'flow': (optical_flow_from_video, False),
    'mbs': (mbs_from_folder, True)
}

class VideoDataset(Dataset):
    """Dataset for looping through videos in folder and generating saliency maps"""

    def __init__(self, dataset: Path, saliency_path: Path, method: str, extension='mp4'):
        self.root = dataset
        self.extension = extension
        self.videos = self.get_video_list()
        self.saliency_path = saliency_path
        self.method = method

    def get_video_list(self) -> List[Path]:
        return list(self.root.glob(f'**/*.{self.extension}'))

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video = self.videos[index]
        method, use_docker = method_index[self.method]
        if use_docker:
            self.to_saliency_docker(method, video)
        else:
            self.to_saliency(method, video)
        return video


    def saliency_destination(self, video_path: Path) -> Tuple[Path, bool]:
        # Setup saliency destination directory
        subfolders = video_path.relative_to(self.root).parent
        output_path = self.saliency_path / subfolders / video_path.stem
        if output_path.exists():
            return output_path, True
        output_path.mkdir(parents=True)

        return output_path, False

    def save_optical_flow(self, flow: torch.Tensor, path: Path):
        H, W, C = flow.shape
        assert C == 2

        flow_write_as_png(flow.numpy(), str(path))
        path = path.with_suffix('.flo')
        flow_write(flow.numpy(), str(path))

    def save_frame(self, frame: torch.Tensor, path: Path):
        if torch.max(frame) < 2:
            frame *= 255

        frame = frame.numpy().astype(np.uint8)

        with open(str(path), 'w') as f:
            img = Image.fromarray(frame)
            img.save(f, format='jpeg')


    def to_saliency(self, method: Callable, video_path: Path):
        # Get video info
        probe = ffmpeg.probe(str(video_path))
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        
        # Read video data
        out, _ = (
            ffmpeg
            .input(str(video_path))
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(quiet=True)
        )
        video = np.frombuffer(out, np.uint8)\
            .reshape([-1, height, width, 3])
        
        saliency = method(torch.from_numpy(video), None)

        output_path, exists = self.saliency_destination(video_path)

        if exists:
            print('Video already generated, skipping', video_path.name)
            return

        for i, frame in enumerate(saliency):
            frame_path = output_path / f'{i}.jpg'

            if self.method == 'flow':
                self.save_optical_flow(frame, frame_path)
            else:
                self.save_frame(frame, frame_path)


    def to_saliency_docker(self, method: Callable, video: Path):
        # Check where to store temporary files
        tmp_dir = Path('/scratch/')
        if not tmp_dir.exists():
            tmp_dir = Path('./.tmp_videos')
            if not tmp_dir.exists():
                tmp_dir.mkdir()

        output_path, exists = self.saliency_destination(video)
        if exists:
            print('Video already generated, skipping', video.name)
            return
        
        # Create temporary folder to store image sequence
        folder_name = hashlib.sha224(str(video).encode()).hexdigest()
        input_path = tmp_dir / folder_name
        input_path.mkdir()

        # Convert video file to image sequence
        (
            ffmpeg
            .input(str(video))
            .output(str(input_path / '%01d.jpg'))
            .run(quiet=True)
        )
        print('Converted video to image sequence', video.name)
        
        stdout = method(input_path, output_path)
        
        print('Converted image sequence to saliency', video.name)

        # Remove temporary files
        for image in input_path.glob('*.jpg'):
            image.unlink()
        input_path.rmdir()

def no_collate(input):
    return input

def generate(args):
    dataset = VideoDataset(Path(args.data_path), Path(args.saliency_path), args.method)
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
            i, _ = next(data_generator)
        except StopIteration:
            break
        except Exception as e:
            print('skipped video clip', str(i), str(e))

    pbar.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Saliency generator')

    parser.add_argument('--data-path', default='../kinetics/', help='Path to dataset')
    parser.add_argument('--saliency-path', default='./saliency_cache/',
        help='Path to saliency cache')
    parser.add_argument('--method', default='harris', help='Method to use with saliency generation')
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    # parser.add_argument('-rs', '--rescale', default=1, type=float, 
    # help='How much to scale video frames before computing saliency')

    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
 
    args = parser.parse_args()

    if args.method not in method_index.keys():
        print(f'Unknown method: {args.method}, use one of: {method_index.keys()}')
        exit(-1)

    generate(args)
