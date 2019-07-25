import argparse

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from apex import amp

except ImportError:
    amp = None

from dataset import LMDBDataset
from pixelsnail import PixelSNAIL
from scheduler import CycleScheduler


def train(args, epoch, loader, model, optimizer, scheduler, device):
    loader = tqdm(loader)

    criterion = nn.CrossEntropyLoss()
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    for i, codes in enumerate(loader):
        model.zero_grad()

        if args.hier == 0:
            target = codes[0].cuda()
            loss, acc = model(target)
        else:
            codes = [c.cuda() for c in codes[args.hier-1:args.hier]]
            top = codes[0]
            bottom = codes[1]
            target = bottom
            loss, acc = model(bottom, condition=top)
        loss = loss.mean()
        accuracy = acc.mean().item()
        #loss = criterion(out, target)
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        #_, pred = out.max(1)
        #correct = (pred == target).float()
        #accuracy = correct.sum() / target.numel()

        lr = optimizer.param_groups[0]['lr']

        loader.set_description(
            (
                f'epoch: {epoch + 1}; loss: {loss.item():.5f}; '
                f'acc: {accuracy:.5f}; lr: {lr:.5f}'
            )
        )


class PixelTransform:
    def __init__(self):
        pass

    def __call__(self, input):
        ar = np.array(input)

        return torch.from_numpy(ar).long()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=420)
    parser.add_argument('--hier', type=int, default=0)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--channel', type=int, default=256)
    parser.add_argument('--n_res_block', type=int, default=4)
    parser.add_argument('--n_res_channel', type=int, default=256)
    parser.add_argument('--n_out_res_block', type=int, default=0)
    parser.add_argument('--n_cond_res_block', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--amp', type=str, default='O0')
    parser.add_argument('--sched', type=str)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('path', type=str)

    args = parser.parse_args()

    print(args)

    device = 'cuda:0'

    dataset = LMDBDataset(args.path)
    loader = DataLoader(
        dataset, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True
    )

    ckpt = {}

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt)
        args = ckpt['args']

    if args.hier == 0:
        model = PixelSNAIL(
            [16, 16],
            512,
            args.channel,
            5,
            4,
            args.n_res_block,
            args.n_res_channel,
            dropout=args.dropout,
            n_out_res_block=args.n_out_res_block,
        )

    else:
        size = 16 * (2 ** args.hier)
        print(size)
        model = PixelSNAIL(
            [size, size],
            512,
            args.channel,
            5,
            4,
            args.n_res_block,
            args.n_res_channel,
            attention=False,
            dropout=args.dropout,
            n_cond_res_block=args.n_cond_res_block,
            cond_res_channel=args.n_res_channel,
        )

    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    ckpt = None


    model = model.cuda()#to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    #if amp is not None:
    #    model, optimizer = amp.initialize(model, optimizer)

    model = nn.DataParallel(model)
    #model = model.to(device)

    scheduler = None
    if args.sched == 'cycle':
        scheduler = CycleScheduler(
            optimizer, args.lr, n_iter=len(loader) * args.epoch, momentum=None
        )

    for i in range(args.epoch):
        train(args, i, loader, model, optimizer, scheduler, device)
        torch.save(
            {'model': model.module.state_dict(), 'args': args},
            f'checkpoint/pixelsnail_{args.hier}_{str(i + 1).zfill(3)}.pt',
        )
