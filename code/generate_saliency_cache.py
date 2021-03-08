import argparse
import torch
from data.salient_kinetics import SalientKinetics400
from train import _get_cache_path

def generate(args):
    dataset = SalientKinetics400(
                    args.data_path,
                    args.saliency_path,
                    frames_per_clip=1,
                    step_between_clips=1,
                    transform=None,
                    salient_transform=None,
                    extensions=('mp4'),
                    frame_rate=None,
                    _precomputed_metadata=None
                )
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=None, num_workers=args.workers//2,
        pin_memory=True, collate_fn=None)

    print(f'Total number of videos: {len(data_loader)}')

    prev_progress = -1
    for i, _ in enumerate(data_loader):
        progress = int((i / len(data_loader)) * 100)
        if prev_progress != progress:
            print(f'{progress}% videos processed')
            prev_progress = progress
        




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Saliency Map Cache Generator')

    parser.add_argument('--data-path', default='/../kinetics/', help='Path to dataset')
    parser.add_argument('--saliency-path', default='./saliency_cache/',
        help='Path to saliency cache')
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
 
    args = parser.parse_args()

    generate(args)
