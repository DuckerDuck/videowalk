import pytest
import utils
import torch
import random
from argparse import Namespace

args = Namespace(affinity_variant='multiply', batch_size=1, cache_dataset=False, cache_dataset_path=None, clip_len=4, clips_per_video=5, data_parallel=True, data_path='../kinetics', device='cuda', dropout=0.1, epochs=25, fast_test=False, featdrop=0.0, flip=False, frame_aug='grid', frame_skip=8, frame_transforms='crop', head_depth=0, img_size=256, lr=0.0001, lr_gamma=0.3, lr_milestones=[20, 30, 40], lr_warmup_epochs=0, manualSeed=1235, model_type='scratch', momentum=0.9, name='6-1-_drop0.1-len4-ftranscrop-fauggrid-optimadam-temp0.05-fdrop0.0-lr0.0001-mlp0', optim='adam', output_dir='checkpoints/_drop0.1-len4-ftranscrop-fauggrid-optimadam-temp0.05-fdrop0.0-lr0.0001-mlp0/', partial_reload='', patch_size=[64, 64, 3], port=8095, print_freq=10, prior_dataset='itti', prior_frame_index=0, remove_layers=[], restrict=-1, resume='', server='localhost', sk_align=False, sk_targets=False, start_epoch=0, steps_per_epoch=10000000000.0, temp=0.05, theta_affinity=0.1, visualize=True, weight_decay=0.0001, with_guiding=True, workers=0, zero_diagonal=False)

image = torch.arange(4*10*10).reshape(4, 10, 10).byte()

def set_seeds(args):
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

def test_saliency_transform():
    set_seeds(args)
    
    transform = utils.augs.get_train_saliency_transform(args)
    result, _ = transform(image)
    torch.save(result, 'tests/aug_output.pth')

def test_similar_to_file():
    set_seeds(args)
    
    a = torch.load('tests/a.pth')
    transform = utils.augs.get_train_saliency_transform(args)
    b, _ = transform(image)
    a = torch.tensor(a)
    b = torch.tensor(b)

    assert torch.isclose(a, b).all()