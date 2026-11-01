"""Train ISLR_Transformer_VCN using XY landmark shape/average/position features.

Contains only the architecture and training path used for landmarks_100_v1.pt.
Importing this module does not start training or open the camera.
"""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import DEVICE, ROOT, DATA_ROOT, _COORD_DIM, _NUM_NODE, _N_HAND, _REMOVE_POSE_IDX, _REMOVE_VIDEO_IDS, set_seed
from app.landmarks import FusionComponent

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

_HAND_BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]

_POSE_BONES = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (15, 17), (15, 19), (15, 21), (17, 19), (16, 18), (16, 20), (16, 22), (18, 20)]

CUSTOM_EDGES = []

remove_set = set(_REMOVE_POSE_IDX)

remaining_nodes = [n for n in range(32) if n not in remove_set]

old_to_new = {old: new for (new, old) in enumerate(remaining_nodes)}

n_pose_kept = len(remaining_nodes)

def _remap(edges):
    return [(old_to_new[a], old_to_new[b]) for (a, b) in edges if a in old_to_new and b in old_to_new]

POSE_EDGES = _remap(_POSE_BONES)

_LEFT_WRIST = 0

_RIGHT_WRIST = _LEFT_WRIST + _N_HAND

_LEFT_HAND_EDGES = [(_LEFT_WRIST + i, _LEFT_WRIST + j) for (i, j) in _HAND_BONES]

_RIGHT_HAND_EDGES = [(_RIGHT_WRIST + i, _RIGHT_WRIST + j) for (i, j) in _HAND_BONES]

FULL_BODY_EDGES = POSE_EDGES + _LEFT_HAND_EDGES + _RIGHT_HAND_EDGES

HAND_EDGES = _LEFT_HAND_EDGES + _RIGHT_HAND_EDGES

assert len({tuple(sorted(e)) for e in HAND_EDGES}) == len(HAND_EDGES), 'duplicate edges'

def build_adjacency_from_edges(edges, num_nodes):
    A = torch.eye(num_nodes)
    for (i, j) in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    return A

class TemporalConv(nn.Module):

    def __init__(self, channels, kernel_size=9, stride=1, dropout=0.1):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(out_channels=channels, in_channels=channels, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1), bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = self.bn(x)
        x = self.drop(x)
        x = x.permute(0, 2, 3, 1)
        return x

class DecoupledGCN(nn.Module):

    def __init__(self, in_channels, out_channels, num_nodes, base_adjacency, decouple_p=4):
        super().__init__()
        self.V = num_nodes
        self.p = decouple_p
        self.phi = nn.Linear(in_channels, out_channels)
        base_adjacency = base_adjacency.float()
        self.register_buffer('I', torch.eye(self.V))
        self.A_in = nn.Parameter(base_adjacency.unsqueeze(0).repeat(self.p, 1, 1) * 0.001)

    def _raw_A(self):
        return self.I.unsqueeze(0) + self.A_in

    def _normalized_A(self, A_raw):
        deg = A_raw.sum(-1).clamp(min=1e-06)
        d_inv_sqrt = deg.pow(-0.5)
        D_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        return D_inv_sqrt @ A_raw @ D_inv_sqrt

    def forward(self, x):
        feat = self.phi(x)
        A_raw = self._raw_A()
        A_norm = self._normalized_A(A_raw)
        out = 0
        for k in range(self.p):
            out = out + torch.einsum('vw,btwc->btvc', A_norm[k], feat)
        out = out / self.p
        return (out, A_raw)

class V_GCN(DecoupledGCN):
    """Shape-only input-dependent graph convolution."""

    def forward(self, shape_feature):
        shape = self.phi(shape_feature)
        scores = torch.matmul(shape, shape.transpose(-1, -2)) * shape.size(-1) ** (-0.5)
        A_shape = (scores.unsqueeze(2) + self.A_in).softmax(dim=-1)
        A_raw = self.I + A_shape
        A_norm = self._normalized_A(A_raw)
        out = torch.einsum('btpvw,btwc->btpvc', A_norm, shape).mean(dim=2)
        return (out, A_raw)

class V_GCN_Block(nn.Module):
    """Shape-only counterpart of SelfPacingDroppingBlock."""

    def __init__(self, in_ch, out_ch, num_nodes, base_adjacency, decouple_p=4, drop=True, use_tcn=True, temporal_kernel_size=9, dropout=0.1):
        super().__init__()
        self.gcn = V_GCN(in_ch, out_ch, num_nodes, base_adjacency, decouple_p)
        self.use_tcn = use_tcn
        self.tcn = TemporalConv(out_ch, kernel_size=temporal_kernel_size if use_tcn else 1, dropout=dropout)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, shape_feature, mask=None):
        valid = torch.ones(shape_feature.shape[:2], device=shape_feature.device, dtype=torch.bool) if mask is None else mask.to(device=shape_feature.device, dtype=torch.bool)
        valid = valid[:, :, None, None]
        shape_feature = shape_feature.masked_fill(~valid, 0)
        (feat, _) = self.gcn(shape_feature)
        feat = feat.masked_fill(~valid, 0)
        if self.use_tcn:
            feat = self.act(self.tcn(feat))
        return self.dropout(feat).masked_fill(~valid, 0)
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        T = x.size(1)
        x = x + self.pe[:, :T, :].to(x.device)
        return self.dropout(x)
def masked_mean_pool(x, video_mask):
    mask = video_mask.unsqueeze(-1).float()
    summed = (x * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts
class ModelOutput(dict):

    def __init__(self, logits=None, loss=None, **kwargs):
        super().__init__(logits=logits, loss=loss, **kwargs)
        self.__dict__ = self

    def __iter__(self):
        yield self['logits']
        yield self['loss']

    def __getitem__(self, key):
        return super().__getitem__(key)

class ISLR_Transformer_VCN(nn.Module):

    def __init__(self, gcn_channels=(128, 128, 256, 256), d_model=256, num_encoder_layers=3, nhead=8, dim_feedforward=256 * 8, dropout=0.1, max_seq_len=300, num_classes=1000, num_nodes=_NUM_NODE):
        super().__init__()
        self.num_nodes = num_nodes
        self.A_in = build_adjacency_from_edges(FULL_BODY_EDGES, num_nodes=self.num_nodes)
        self.register_buffer('adjacency_matrix', self.A_in)
        channels = [_COORD_DIM, *gcn_channels]
        self.feature_dim = num_nodes * _COORD_DIM

        def make_projection(input_dim=self.feature_dim):
            return nn.Sequential(nn.Linear(input_dim, d_model), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(d_model))
        self.shape_projection = make_projection()
        self.average_projection = make_projection()
        self.position_projection = make_projection()
        self.gcn_block = nn.ModuleList([V_GCN_Block(channels[i], channels[i + 1], self.num_nodes, self.adjacency_matrix, dropout=dropout) for i in range(len(channels) - 1)])
        self.node_attention = nn.Linear(channels[-1], 1)
        self.gcn_output_projection = nn.Linear(channels[-1], d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_seq_len, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.encoder_norm = nn.LayerNorm(d_model)
        fusion_dim = d_model
        self.classifier = nn.Sequential(nn.LayerNorm(fusion_dim), nn.Dropout(dropout), nn.Linear(fusion_dim, num_classes))

    def encode_gcn(self, features, video_mask):
        for block in self.gcn_block:
            features = block(features, video_mask)
        features = features.mean(dim=-2)
        return features

    def encode_transformer(self, features, video_mask, x_rgb=None):
        video_mask = video_mask.to(device=features.device, dtype=torch.bool)
        features = features.masked_fill(~video_mask.unsqueeze(-1), 0)
        (B, T, D) = features.shape
        (shape_feature, average_feature, position_feature) = features.split(self.feature_dim, dim=-1)
        x_shape = self.shape_projection(shape_feature)
        x_average = self.average_projection(average_feature)
        x_position = self.position_projection(position_feature)
        x = x_shape + x_average + x_position
        x = self.pos_encoder(x)
        x = self.encoder(x, src_key_padding_mask=~video_mask)
        return self.encoder_norm(x)

    def forward(self, features, labels=None, video_mask=None, x_rgb=None):
        video_mask = video_mask.to(device=features.device, dtype=torch.bool)
        features = features.masked_fill(~video_mask.unsqueeze(-1), 0)
        (B, T, D) = features.shape
        (shape_feature, average_feature, position_feature) = features.split(self.feature_dim, dim=-1)
        t = self.encode_transformer(features, video_mask, x_rgb=x_rgb)
        x_gcn = self.encode_gcn(shape_feature.reshape(B, T, -1, _COORD_DIM), video_mask)
        t = masked_mean_pool(t, video_mask)
        x_gcn = masked_mean_pool(x_gcn, video_mask)
        t = t + x_gcn
        logits = self.classifier(t)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return ModelOutput(logits=logits, loss=loss, encoded=t)
def collate_fn(batch):
    (features, labels, video_ids) = ([], [], [])
    for (feature, label, video_id) in batch:
        features.append(torch.as_tensor(feature, dtype=torch.float32))
        labels.append(label)
        video_ids.append(video_id)
    real_lengths = [f.shape[0] for f in features]
    features = pad_sequence(features, batch_first=True)
    video_mask = (torch.arange(features.shape[1]).unsqueeze(0) < torch.tensor(real_lengths).unsqueeze(1)).long()
    labels = torch.tensor(labels, dtype=torch.long)
    return (features, labels, video_mask, video_ids)

def train_one_epoch(model, loader, optimizer, device='cuda'):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc='Training')
    for (features, labels, video_mask, _) in pbar:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        video_mask = video_mask.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        (_logits, loss) = model(features, labels=labels, video_mask=video_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=f'{loss.item():.4f}')
    return total_loss / len(loader)

@torch.no_grad()
def validate(model, loader, top_k=5, device='cuda'):
    model.eval()
    total_loss = 0
    total_correct_top1 = 0
    total_correct_topk = 0
    total_samples = 0
    with torch.no_grad():
        for (features, labels, video_mask, _) in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            video_mask = video_mask.to(device, non_blocking=True)
            (logits, loss) = model(features, labels=labels, video_mask=video_mask)
            total_loss += loss
            pred_top1 = logits.argmax(dim=-1)
            total_correct_top1 += (pred_top1 == labels).sum().item()
            k = min(top_k, logits.size(-1))
            pred_topk = logits.topk(k=k, dim=-1).indices
            in_topk = (pred_topk == labels.unsqueeze(-1)).any(dim=-1)
            total_correct_topk += in_topk.sum().item()
            total_samples += labels.size(0)
    avg_loss = total_loss / len(loader)
    top1_acc = total_correct_top1 / total_samples if total_samples > 0 else 0.0
    topk_acc = total_correct_topk / total_samples if total_samples > 0 else 0.0
    return (avg_loss, top1_acc, topk_acc)

class WLASLLandmarksDataset(Dataset):
    """Load pose.npy (T,33,3) and hand.npy (T,21,3), keeping XY coordinates."""

    def __init__(self, feature_dir, annotation_dir, mode='train'):
        self.fusion_component = FusionComponent()
        self.cache = {}
        annotation_dir = Path(annotation_dir)
        self.gloss2idx = json.loads((annotation_dir / 'gloss2idx.json').read_text(encoding='utf-8'))
        entries = json.loads((annotation_dir / f'{mode}.json').read_text(encoding='utf-8'))
        entries = {str(entry['video_id']): entry['gloss'] for entry in entries}
        self.samples = []
        excluded = set(map(str, _REMOVE_VIDEO_IDS))
        for video_id in sorted(entries):
            directory = Path(feature_dir) / video_id
            paths = [directory / f'{name}.npy' for name in ('pose', 'left_hand', 'right_hand')]
            if video_id in excluded or not all(path.is_file() for path in paths):
                continue
            gloss = entries[video_id]
            if gloss is None and (directory / 'gloss.txt').is_file():
                gloss = (directory / 'gloss.txt').read_text(encoding='utf-8').strip()
            if gloss in self.gloss2idx:
                self.samples.append((video_id, paths, self.gloss2idx[gloss]))
        if not self.samples:
            raise RuntimeError(f'No valid landmark samples for {mode}: {feature_dir}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        video_id, paths, label = self.samples[index]
        if video_id not in self.cache:
            pose, left, right = [np.load(path).reshape(-1, joints, 3)[..., :_COORD_DIM]
                                 for path, joints in zip(paths, (33, 21, 21))]
            fusion = self.fusion_component
            self.cache[video_id] = np.concatenate([
                fusion.fuse_follow_shape(pose, left, right),
                fusion.fuse(pose, left, right),
                fusion.fuse_follow_position(pose, left, right),
            ], axis=-1).astype(np.float32)
        return self.cache[video_id], label, video_id


def main(args):
    set_seed(args.seed)
    if args.epochs < 1 or args.batch_size < 1 or args.top_k < 1:
        raise ValueError('epochs, batch_size and top_k must be positive')
    train_data = WLASLLandmarksDataset(args.data_path, args.label_path, mode='train')
    val_data = WLASLLandmarksDataset(args.data_path, args.label_path, mode=args.val_mode)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0, pin_memory=DEVICE.type == 'cuda')
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0, pin_memory=DEVICE.type == 'cuda')
    model_kwargs = {'num_classes': len(train_data.gloss2idx), 'max_seq_len': 5000}
    model = ISLR_Transformer_VCN(**model_kwargs).to(DEVICE)
    print(f'ISLR_Transformer_VCN on {DEVICE}; parameters: {sum(p.numel() for p in model.parameters()):,}')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    best = -1.0
    no_improve = 0
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device=DEVICE)
        val_loss, top1, topk = validate(model, val_loader, top_k=args.top_k, device=DEVICE)
        val_loss = float(val_loss)
        scheduler.step(val_loss)
        print(f'Epoch {epoch + 1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, top1={top1:.2%}, top{args.top_k}={topk:.2%}')
        if top1 > best:
            best, no_improve = top1, 0
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'epoch': epoch, 'val_loss': val_loss,
                        'val_top1_acc': top1, 'val_topk_acc': topk, 'model_kwargs': model_kwargs}, args.output)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                break


def default_args():
    parser = argparse.ArgumentParser(description='Train ISLR_Transformer_VCN from landmark features')
    parser.add_argument('--data_path', default=str(Path(DATA_ROOT) / 'processed/wlasl_features_0_6'))
    parser.add_argument('--label_path', default=str(Path(DATA_ROOT) / 'annotations/WLASL100'))
    parser.add_argument('--output', default=str(Path(ROOT) / 'outputs/models/landmarks_100_v1.pt'))
    parser.add_argument('--val_mode', choices=['val', 'test'], default='test')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--top_k', type=int, default=2)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    return parser


if __name__ == '__main__':
    main(default_args().parse_args())
