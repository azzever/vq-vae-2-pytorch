import argparse
import pickle

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import lmdb
from tqdm import tqdm

from dataset import ImageFileDataset, CodeRow
from vqvae import VQVAE


def extract(lmdb_env, loader, model, device):
    index = 0

    with lmdb_env.begin(write=True) as txn:
        pbar = tqdm(loader)

        sss = 0 

        for img, _, filename in pbar:
            img = img.to(device)

            _, _, indexes = model(img)
            indexes = [f.data.cpu().numpy() for f in indexes]

            for i, file in enumerate(filename):
                index = [f[i] for f in indexes]
                txn.put(str(sss).encode('utf-8'), pickle.dumps(index))
                pbar.set_description(f'inserted: {sss}')
                sss += 1

        txn.put('length'.encode('utf-8'), str(sss).encode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('path', type=str)

    args = parser.parse_args()

    device = 'cuda'

    transform = transforms.Compose(
        [
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dataset = ImageFileDataset(args.path, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8)

    model = VQVAE()
    model.load_state_dict(torch.load(args.ckpt))
    model = model.to(device)
    model.forward = model.encode
    model.eval()
    #model = torch.nn.DataParallel(model)

    map_size = 100 * 1024 * 1024 * 1024

    env = lmdb.open(args.name, map_size=map_size)

    extract(env, loader, model, device)
