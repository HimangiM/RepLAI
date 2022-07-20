import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from utils.distributed_ops import concat_all_gather
from typing import Dict, List

__all__ = [
    'SSRL_audiopeak_aotv2'
]

class SSRL_audiopeak_aotv2(nn.Module):
    """
    A Simple Framework for Contrastive Learning of Visual Representations
    Details can be found from:
    https://arxiv.org/abs/2002.05709
    """

    def __init__(
        self,
        backbone,
        in_dim: int,
        out_dim: int = 128,
        temperature: float = 0.07,
        num_clips: int = 8,
        aot_coeff: float = 0.5,
        clr_coeff: float = 0.5
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.mlp = nn.ModuleDict({
            'video_clr': nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU(inplace=True),
                nn.Linear(in_dim, out_dim)),
            'audio_clr': nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU(inplace=True),
                nn.Linear(in_dim, out_dim)),
            'video_aot': nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU(inplace=True),
                nn.Linear(in_dim, out_dim)),
            'audio_aot': nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU(inplace=True),
                nn.Linear(in_dim, out_dim)),
            'aot_diff': nn.Sequential(
                nn.Linear(out_dim, in_dim),
                nn.ReLU(inplace=True),
                nn.Linear(in_dim, out_dim)),
            # audio_clr left and right, in forward function
        })
        self.temperature = temperature
        self.audio_pool = nn.AdaptiveMaxPool2d((num_clips-1, 1))
        self.audio_pool_orig = nn.AdaptiveMaxPool2d((1, 1))
        self.aot_coeff = aot_coeff
        self.clr_coeff = clr_coeff

        import torch.nn.init as init
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)

    def get_backbone_params(self):
        return list(self.backbone['video'].parameters()) + list(self.backbone['audio'].parameters())

    def get_head_params(self):
        head_params = []
        for k,v in self.mlp.items():
            head_params += list(v.parameters())
        return head_params

    def contrastive_loss(self, xv_l: torch.Tensor, xv_r: torch.Tensor, xa: torch.Tensor) -> torch.Tensor:
        # xv_l, xv_r = B x out_dim
        xv = torch.stack((xv_l, xv_r), dim=1)  # B x 2 x out_dim

        # xv = (B x 2 x out_dim), xa = (B x out_dim)
        # xv_trg = (B x 2 x out_dim), xa_trg = (B x out_dim)
        xv_trg = concat_all_gather(xv)
        xa_trg = concat_all_gather(xa)

        # sim_va = (B x 2 x B), sim_av = (B x 2 x B)
        sim_va = torch.einsum('npd,md->nmp', xv, xa_trg).permute(0, 2, 1) / self.temperature
        sim_av = torch.einsum('nd,mpd->nmp', xa, xv_trg).permute(0, 2, 1) / self.temperature

        # bs = B, nc = 2, siz = 128, device= cuda
        (bs, nc, siz), device = xv.shape, xv.device
        # rank = 0
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        # gt = [B]
        gt = torch.arange(rank * bs, (rank + 1) * bs).to(device)
        # gt = (B x 2)
        gt = gt[:, None].expand(size=(bs, nc))

        # sim_va.flatten(0, 1) = (128 x 16), gt.flatten(0, 1) = 128
        # sim_av.flatten(0, 1) = (128 x 16), gt.flatten(0, 1) = 128
        loss_va = F.cross_entropy(sim_va.flatten(0, 1), gt.flatten(0, 1))
        loss_av = F.cross_entropy(sim_av.flatten(0, 1), gt.flatten(0, 1))
        loss = (loss_va + loss_av) / 2.
        return loss

    # use audio to decide what comes first, the order has to be conditioned on the audio.
    def arrow_of_time_loss(self, xv_l: torch.Tensor, xv_r: torch.Tensor, xa: torch.Tensor) -> torch.Tensor:
        # xv_l, xv_r = B x out_dim, B x out_dim
        diff_video_ones = F.normalize(self.mlp['aot_diff'](xv_r - xv_l), p=2, dim=-1)
        diff_video_zeros = F.normalize(self.mlp['aot_diff'](xv_l - xv_r), p=2, dim=-1)
        xa = F.normalize(xa, p=2, dim=-1)

        data_ones = torch.einsum('bc,bc->b', diff_video_ones, xa) / 0.2
        data_zeros = torch.einsum('bc,bc->b', diff_video_zeros, xa) / 0.2
        outputs = torch.cat((data_ones, data_zeros), dim=0)   # 2*B

        label_size = xv_l.shape[0]
        labels_ones = torch.ones(label_size).type(torch.LongTensor).to(xv_l.device)
        labels_zeros = torch.zeros(label_size).type(torch.LongTensor).to(xv_l.device)
        labels = torch.cat((labels_ones, labels_zeros), 0)  # 2*B

        loss = F.binary_cross_entropy_with_logits(outputs, labels.float())
        return loss

    def forward(self, video_l: List[torch.Tensor],
                video_r: List[torch.Tensor],
                audio: torch.Tensor) -> (torch.Tensor, Dict):
        """
        Args:
            x1 (torch.tensor): a batch of image with augmentation. The input tensor
                shape should able to be feed into the backbone.
            x2 (torch.tensor): the size batch of image with different augmentation. The
                input tensor shape should able to be feed into the backbone.
        """
        # video_l, video_r = 1 x B x 3 x 8 x 112 x 112
        # audio = 1 x B x 1 x 125 x 80
        # video - left
        bs, nc = video_l[0].shape[0], len(video_l)       # bs = B, nc = 1
        assert nc == 1

        # video_l[0] = B x 3 x 8 x 112 x 112
        # video_r[0] = B x 3 x 8 x 112 x 112
        video_lr = torch.cat((video_l[0], video_r[0]), 0)   # video_lr = B*2 x 3 x 8 x 112 x 112
        xv_lr = self.backbone['video'](video_lr)            # B*2 X 512 x 1 x 1 x 1
        # video clr
        xv_lr_clr = self.mlp['video_clr'](xv_lr.flatten(1, -1))   # B*2 x out_dim
        xv_lr_clr = F.normalize(xv_lr_clr, p=2, dim=-1)            # B*2 x out_dim
        xv_l_clr, xv_r_clr = torch.chunk(xv_lr_clr, chunks=2, dim=0)
        # video aot
        xv_lr_aot = self.mlp['video_aot'](xv_lr.flatten(1, -1))   # B*2 x out_dim
        xv_l_aot, xv_r_aot = torch.chunk(xv_lr_aot, chunks=2, dim=0)

        # audio[0] shape = (B x 1 x 562 x 80)
        xa = self.backbone['audio'](audio[0])           # xa: B x 512 x 8 x 5
        xa = self.audio_pool_orig(xa).flatten(1, -1)    # xa: B x 512
        xa_clr = self.mlp['audio_clr'](xa)              # B x out_dim
        xa_clr = F.normalize(xa_clr, p=2, dim=-1)
        xa_aot = self.mlp['audio_aot'](xa)              # B x 512

        aot_loss = self.arrow_of_time_loss(xv_l_aot, xv_r_aot, xa_aot)
        clr_loss = self.contrastive_loss(xv_l_clr, xv_r_clr, xa_clr)
        # contrastive loss (video, audio) and average the two
        total_loss = self.aot_coeff * aot_loss + self.clr_coeff * clr_loss
        stats = {
            'clr_loss': clr_loss,
            'aot_loss': aot_loss,
        }
        return total_loss, stats

# generate audio conditioned left and right features and then do clr loss; average them
# transformer attention, 3 tokens, and use one for aot one for clr
# separate xv_l and xv_r