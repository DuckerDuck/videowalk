import pytest
from torch.utils import data
from torch.utils.data import dataset
import torch
import utils
from pathlib import Path
from argparse import Namespace
from data.salient_kinetics import SalientKinetics400
from train import salient_collate_fn
from torchvision.datasets.samplers.clip_sampler import RandomClipSampler
from .conftest import set_seeds


args = Namespace(affinity_variant='multiply', batch_size=1, cache_dataset=False, cache_dataset_path=None, clip_len=4, clips_per_video=5, data_parallel=True, data_path='./tests/dataset_test/train_256/', device='cuda', dropout=0.1, epochs=25, fast_test=False, featdrop=0.0, flip=False, frame_aug='grid', frame_skip=8, frame_transforms='crop', head_depth=0, img_size=256, lr=0.0001, lr_gamma=0.3, lr_milestones=[20, 30, 40], lr_warmup_epochs=0, manualSeed=1235, model_type='scratch', momentum=0.9, name='6-1-_drop0.1-len4-ftranscrop-fauggrid-optimadam-temp0.05-fdrop0.0-lr0.0001-mlp0', optim='adam', output_dir='checkpoints/_drop0.1-len4-ftranscrop-fauggrid-optimadam-temp0.05-fdrop0.0-lr0.0001-mlp0/', partial_reload='', patch_size=[64, 64, 3], port=8095, print_freq=10, prior_dataset='itti', prior_frame_index=0, remove_layers=[], restrict=-1, resume='', server='localhost', sk_align=False, sk_targets=False, start_epoch=0, steps_per_epoch=10000000000.0, temp=0.05, theta_affinity=0.1, visualize=True, weight_decay=0.0001, with_guiding=True, workers=0, zero_diagonal=False)
image = Path('./tests/img.jpg')
reference = Path('./tests/img.pth')

salient_transform = utils.augs.get_train_saliency_transform(args)
transform_train = utils.augs.get_train_transforms(args)
dataset = SalientKinetics400(args.data_path, './tests/dataset_saliency_test/', args.frame_skip, transform=transform_train, salient_transform=salient_transform)
train_sampler = RandomClipSampler(dataset.video_clips, args.clips_per_video)
data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1,
        sampler=train_sampler, num_workers=1,
        pin_memory=True, collate_fn=salient_collate_fn)

def test_same_as_file():
    frame = dataset.load_frame(image)
    frame_ref = torch.load(reference)
    assert torch.isclose(frame, frame_ref).all()

def test_dataloader():
    set_seeds(args)
    (video, _), (saliency, _) = next(iter(data_loader))
    
    # torch.save(video, 'tests/video_sample.pth')
    # torch.save(saliency, 'tests/saliency_sample.pth')

    video_reference = torch.load('tests/video_sample.pth')
    saliency_reference = torch.load('tests/saliency_sample.pth')

    assert torch.isclose(video, video_reference).all()
    assert torch.isclose(saliency, saliency_reference).all()