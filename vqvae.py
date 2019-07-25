import torch
from torch import nn
from torch.nn import functional as F


# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


# Borrowed from https://github.com/deepmind/sonnet and ported it to PyTorch


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', embed.clone())

    def forward(self, input):
        input = input.permute(0, 2, 3, 1)
        oririnal_size = input.shape
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            print("updating")
            self.cluster_size.data.mul_(self.decay).add_(
                1 - self.decay, embed_onehot.sum(0)
            )
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.embed_avg.data.mul_(self.decay).add_(1 - self.decay, embed_sum)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize.permute(0, 3, 1, 2), diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 3, padding=1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]
        
        if stride == 8:
            blocks = [
                nn.Conv2d(in_channel, channel // 4, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 4, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )
        
        if stride == 8:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(channel // 2,  channel // 4, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(channel // 4,  out_channel, 4, stride=2, padding=1),
                    
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class VQVAE(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
        decay=0.99,
    ):
        super().__init__()

        self.enc = nn.ModuleList()
        self.enc.append(Encoder(in_channel, channel, n_res_block, n_res_channel, stride=8)) # 128
        self.enc.append(Encoder(channel, channel, n_res_block, n_res_channel, stride=2)) # 64
        self.enc.append(Encoder(channel, channel, n_res_block, n_res_channel, stride=2)) # 32
        self.enc.append(Encoder(channel, channel, n_res_block, n_res_channel, stride=2)) # 16
        
        self.quantize_proj = nn.ModuleList()
        self.quantize_proj.append(nn.Conv2d(channel, embed_dim, 1)) # 16
        self.quantize_proj.append(nn.Conv2d(channel * 2, embed_dim, 1)) # 32
        self.quantize_proj.append(nn.Conv2d(channel * 2, embed_dim, 1)) # 64
        self.quantize_proj.append(nn.Conv2d(channel * 2, embed_dim, 1)) # 128
        
        self.quantize = nn.ModuleList()
        self.quantize.append(Quantize(embed_dim, n_embed))
        self.quantize.append(Quantize(embed_dim, n_embed))
        self.quantize.append(Quantize(embed_dim, n_embed))
        self.quantize.append(Quantize(embed_dim, n_embed))
        
        self.upsample_proj = nn.ModuleList()
        self.upsample_proj.append(nn.ConvTranspose2d(embed_dim, channel, 4, stride=2, padding=1))
        self.upsample_proj.append(nn.ConvTranspose2d(embed_dim, channel, 4, stride=2, padding=1))
        self.upsample_proj.append(nn.ConvTranspose2d(embed_dim, channel, 4, stride=2, padding=1))
        
        self.dec = nn.ModuleList()
        self.dec.append(Decoder(embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2)) # 16 -> 32
        self.dec.append(Decoder(embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2)) # 32 -> 64
        self.dec.append(Decoder(embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2)) # 64 -> 128
        self.dec.append(Decoder(embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2)) # 128 -> 256 

        self.to_rgb = nn.ModuleList()
        for i in range(0, 4):
            self.to_rgb.append(
                nn.Sequential(
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(embed_dim, embed_dim // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(embed_dim // 2, 3, 4, stride=2, padding=1)))
        self.criterion = nn.MSELoss()
        self.latent_loss_weight = 0.25

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        
        dec = self.decode(quant)
        if self.training:
            loss_img = 0
            for img in dec:
                loss_img += self.criterion(img, F.interpolate(input, size=(img.shape[2], img.shape[3]), mode='area'))
            loss_img /= len(dec)

            return loss_img.unsqueeze(0), diff
        else:
            return dec[-1]

    def encode(self, input):
        
        enc1 = self.enc[0](input)
        enc2 = self.enc[1](enc1)
        enc3 = self.enc[2](enc2)
        enc4 = self.enc[3](enc3)
        
        proj_to_quantize_4 = self.quantize_proj[0](enc4)
        quant_4, diff_t_4, id_t_4 = self.quantize[0](proj_to_quantize_4)
        
        up_from_quant_4 = F.relu(self.upsample_proj[0](quant_4), inplace=True)
        
        proj_to_quantize_3 = self.quantize_proj[1](torch.cat([enc3, up_from_quant_4], 1))
        quant_3, diff_t_3, id_t_3 = self.quantize[1](proj_to_quantize_3)
        
        up_from_quant_3 = F.relu(self.upsample_proj[1](quant_3), inplace=True)
        
        proj_to_quantize_2 = self.quantize_proj[2](torch.cat([enc2, up_from_quant_3], 1))
        quant_2, diff_t_2, id_t_2 = self.quantize[2](proj_to_quantize_2)
        
        up_from_quant_2 = F.relu(self.upsample_proj[2](quant_2), inplace=True)
        
        proj_to_quantize_1 = self.quantize_proj[3](torch.cat([enc1, up_from_quant_2], 1))
        quant_1, diff_t_1, id_t_1 = self.quantize[3](proj_to_quantize_1)
        
        
        diff = diff_t_4 + diff_t_3 + diff_t_2 + diff_t_1
        diff = diff.unsqueeze(0)

        return [quant_4, quant_3, quant_2, quant_1], diff, [id_t_4, id_t_3, id_t_2, id_t_1]

    def decode(self, quant):
        dec = self.dec[0](quant[0])
        rgbs = []
        rgb = self.to_rgb[0](dec)
        rgbs.append(rgb)
        dec = self.dec[1](dec + quant[1])
        rgb = F.upsample(rgb, scale_factor=2) + self.to_rgb[1](dec)
        rgbs.append(rgb)
        dec = self.dec[2](dec + quant[2])
        rgb = F.upsample(rgb, scale_factor=2) + self.to_rgb[2](dec)
        rgbs.append(rgb)
        dec = self.dec[3](dec + quant[3])
        rgb = F.upsample(rgb, scale_factor=2) + self.to_rgb[3](dec)
        rgbs.append(rgb)
        return rgbs 

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)

        return dec
