
from pathlib import Path
import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
import numpy as np
import torch.multiprocessing as mp
from PIL import Image, ImageOps, ImageFilter
from torch import nn, optim
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.distributed as dist
import random
from dataset import ImageNet

parser = argparse.ArgumentParser(description='Barlow Twins Training')
parser.add_argument('--dataset', type=str, metavar='dataset', default="imagenet",
                    help='imagenet and imagenet100')
parser.add_argument('--workers', default=32, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=2048, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate-weights', default=0.2, type=float, metavar='LR',
                    help='base learning rate for weights')
parser.add_argument('--learning-rate-biases', default=0.0048, type=float, metavar='LR',
                    help='base learning rate for biases and batch norm parameters')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                    help='weight on off-diagonal terms')
parser.add_argument('--projector', default='8192-8192-8192', type=str,
                    metavar='MLP', help='projector MLP')
parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--method', default='fea', type=str,
                    metavar='INS', help='instance whitening or feature whitening')
parser.add_argument('--backbone', default='resnet18', type=str, help='resnet18 and resnet50')
parser.add_argument('--group', default=1, type=int, help='group of divided')
parser.add_argument('--swap', default='true', type=str, help='swap on feature dimension')
parser.add_argument('--temperature', default=0.5, type=float, help='simclr temperature')

def main():
    args = parser.parse_args()
    args.ngpus_per_node = torch.cuda.device_count()
    if 'SLURM_JOB_ID' in os.environ:
        # single-node and multi-node distributed training on SLURM cluster
        # requeue job on SLURM preemption
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)
        # find a common host name on all nodes
        # assume scontrol returns hosts in the same order on all nodes
        cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
        stdout = subprocess.check_output(cmd.split())
        host_name = stdout.decode().splitlines()[0]
        args.rank = 0
        args.world_size = int(os.getenv('SLURM_NNODES')) * args.ngpus_per_node
        args.dist_url = f'tcp://{host_name}:58472'
        # os.environ["NCCL_DEBUG"] = "INFO"
    else:
        # single-node distributed training
        args.rank = 0
        args.dist_url = 'tcp://localhost:58472'
        args.world_size = args.ngpus_per_node
    main_worker(int(os.environ["SLURM_PROCID"]), args, int(os.environ['SLURM_LOCALID']))

def main_worker(gpu, args, local_rank):
    args.rank += gpu
    hidden = args.projector.split('-')[0]
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stats_str = "stats-%s-%s-%d-%d-%s.txt" % (args.dataset, args.method, int(hidden), args.batch_size, args.backbone)
    stats_file = open(args.checkpoint_dir / stats_str, 'a', buffering=1)
    if args.rank == 0:
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)
        print(args)
    gpu = local_rank
    torch.distributed.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)
    torch.cuda.set_device(gpu)

    checkpoint_str = "checkpoint-%s-%s-%d-%d-%s.pth" % (args.dataset, args.method, int(hidden), args.batch_size, args.backbone)
    resnet_str = "%s-%s-%s-%d-%d.pth" % (args.backbone, args.dataset, args.method, int(hidden), args.batch_size)

    model = Assignment(args)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda(gpu)

    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    optimizer = LARS(parameters, lr=0, weight_decay=args.weight_decay,
                     weight_decay_filter=exclude_bias_and_norm,
                     lars_adaptation_filter=exclude_bias_and_norm)

    # automatically resume from checkpoint if it exists
    if (args.checkpoint_dir / checkpoint_str).is_file():
        ckpt = torch.load(args.checkpoint_dir / checkpoint_str,
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    else:
        start_epoch = 0

    out_dim = 100 if args.dataset == 'imagenet100' else 1000
    dataset = ImageNet('/mnt/lustre/share/images/train/', Transform(), num_class=out_dim)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=per_device_batch_size, num_workers=args.workers,
        pin_memory=True, sampler=sampler)

    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        for step, (y1, y2, label) in enumerate(loader, start=epoch * len(loader)):
            y1 = y1.cuda(gpu, non_blocking=True)
            y2 = y2.cuda(gpu, non_blocking=True)
            adjust_learning_rate(args, optimizer, loader, step)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = model.forward(y1, y2)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if step % args.print_freq == 0:
                if args.rank == 0:
                    stats = dict(epoch=epoch, step=step,
                                 lr_weights=optimizer.param_groups[0]['lr'],
                                 lr_biases=optimizer.param_groups[1]['lr'],
                                 loss=loss.item(),
                                 time=int(time.time() - start_time))
                    print(json.dumps(stats))
                    print(json.dumps(stats), file=stats_file)
        # save checkpoint
        if args.rank == 0:
            state = dict(epoch=epoch + 1, model=model.state_dict(),
                         optimizer=optimizer.state_dict())

            torch.save(state, args.checkpoint_dir / checkpoint_str)
            torch.save(model.module.backbone.state_dict(), args.checkpoint_dir / resnet_str)


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * args.learning_rate_weights
    optimizer.param_groups[1]['lr'] = lr * args.learning_rate_biases


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class Assignment(nn.Module):
    def __init__(self, args):
        super(Assignment, self).__init__()
        self.args = args
        if args.backbone == 'resnet18':
            self.backbone = torchvision.models.resnet18(zero_init_residual=True)
            outdim = 512
        else:
            self.backbone = torchvision.models.resnet50(zero_init_residual=True)
            outdim = 2048
        self.backbone.fc = nn.Identity()
        # projector
        sizes = [outdim] + list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

    def forward(self, y1, y2):
        z1 = self.projector(self.backbone(y1))
        z2 = self.projector(self.backbone(y2))
        if 'assignment_fea' in self.args.method:
            loss = assignment_fea_loss(z1, z2, self.args)
        elif 'simclr' in self.args.method:
            loss = simclr_loss(z1, z2, self.args)
        else:
            loss = barlow_twins_loss(z1, z2, self.args)
        return loss

def simclr_loss(z1, z2, args):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    batch_size = z1.size(0)
    out = torch.cat([z1, z2], dim=0)
    # [2*B, 2*B]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / args.temperature)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
    # [2*B, 2*B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

    # compute loss
    pos_sim = torch.exp(torch.sum(z1 * z2, dim=-1) / args.temperature)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return loss

def assignment_fea_loss(z1, z2, args):
    z1 = z1.transpose(0, 1)
    z2 = z2.transpose(0, 1)
    if args.swap == 'true':
        idx = torch.randperm(z1.size(0))
        z1 = z1[idx, :]
        z2 = z2[idx, :]
    whiten_net = WhitenTran(args)
    z1 = standardization(z1)
    z2 = standardization(z2)
    z1_group = whiten_net.zca_forward(z1).detach()
    z2_group = whiten_net.zca_forward(z2).detach()

    c = (z1 * z2_group).sum(dim=-1)
    c.div_(args.batch_size)
    torch.distributed.all_reduce(c)
    loss1 = c.add_(-1).pow_(2).sum()

    c = (z2 * z1_group).sum(dim=-1)
    c.div_(args.batch_size)
    torch.distributed.all_reduce(c)
    loss2 = c.add_(-1).pow_(2).sum()

    loss = loss1 + loss2
    return loss

def barlow_twins_loss(z1, z2, args):
    z1 = z1.transpose(0, 1)
    z2 = z2.transpose(0, 1)
    z1 = standardization(z1)
    z2 = standardization(z2)
    c = torch.mm(z1, z2.transpose(0, 1))
    c.div_(args.batch_size)
    torch.distributed.all_reduce(c)
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()
    loss = on_diag + args.lambd * off_diag
    return loss

def standardization(data, eps=1e-5):
    # N * d
    mu = torch.mean(data, dim=-1, keepdim=True)
    sigma = torch.std(data, dim=-1, keepdim=True)
    return (data - mu) / (sigma+eps)

class WhitenTran(nn.Module):
    def __init__(self, args):
        super(WhitenTran, self).__init__()
        self.args = args

    def zca_forward(self, x):
        feature_dim, batch_size = x.size()
        assert feature_dim % self.args.group == 0
        x = x.view(self.args.group, feature_dim // self.args.group, batch_size)
        eps = 1e-4
        f_cov = (torch.bmm(x, x.transpose(1, 2)) / (batch_size - 1)).float()  # N * N
        eye = torch.eye(feature_dim // self.args.group).float().to(f_cov.device)
        x_stack = torch.FloatTensor().to(f_cov.device)
        for i in range(f_cov.size(0)):
            f_cov = torch.where(torch.isnan(f_cov), torch.zeros_like(f_cov), f_cov)
            U, S, V = torch.svd((1 - eps) * f_cov[i] + eps * eye)
            diag = torch.diag(1.0 / torch.sqrt(S + 1e-5))
            rotate_mtx = torch.mm(torch.mm(U, diag), U.transpose(0, 1)).detach()  # N * N
            x_transform = torch.mm(rotate_mtx, x[i])
            x_stack = torch.cat([x_stack, x_transform], dim=0)
        return x_stack


class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)


    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])


def exclude_bias_and_norm(p):
    return p.ndim == 1


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class Transform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    # def __call__(self, x):
    #     y1 = self.transform(x)
    #     y2 = self.transform_prime(x)
    #     return y1, y2


if __name__ == '__main__':
    main()
