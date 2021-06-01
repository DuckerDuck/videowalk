import pytest
import random
import torch

def set_seeds(args):
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)