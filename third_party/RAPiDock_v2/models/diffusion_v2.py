"""
diffusion_v2.py — V2 confidence model for HybriDock-Pep.

Subclasses CGTensorProductEquivariantModel with three targeted improvements
to the confidence head path only. Backbone weights are fully compatible with
the pretrained RAPiDock-Reloaded checkpoint.

Changes vs v1:
  1. Cross-attention pooling: peptide residues are weighted by the L2 norm of
     the receptor→peptide cross-message from the final cross_conv layer, so
     residues at the binding interface dominate the pooled representation.
  2. Sidechain orientation proxy: CA→tip distance per residue, projected to
     16 dims and pooled with the same attention weights.
  3. Receptor SS proxy: Ramachandran-region classification (α/β/other) of
     receptor residues (from stored φ/ψ), mean-pooled over the pocket and
     concatenated as a 3-dim context vector.

Combined head input: 96 (attn-pooled pep) + 16 (sidechain proj) + 3 (SS) = 115-dim.
All new parameters are randomly initialised; backbone is loaded from pretrained ckpt.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean, scatter_max

# ── make sure parent RAPiDock is importable ──────────────────────────────────
_RAPIDOCK_ROOT = Path(__file__).resolve().parent.parent.parent / "RAPiDock"
if str(_RAPIDOCK_ROOT) not in sys.path:
    sys.path.insert(0, str(_RAPIDOCK_ROOT))

from models.diffusion import CGTensorProductEquivariantModel   # noqa: E402
from dataset.peptide_feature import get_updated_peptide_feature  # noqa: E402


# ── helpers ───────────────────────────────────────────────────────────────────

def _batch_softmax(logits: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Batch-aware softmax: softmax(logits) within each sample in the batch."""
    # Subtract per-sample max for numerical stability
    max_vals = scatter_max(logits, batch, dim=0)[0][batch]
    exp = torch.exp(logits - max_vals)
    sum_exp = scatter_add(exp, batch, dim=0)[batch]
    return exp / (sum_exp + 1e-8)


def _ramachandran_class(phi: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """
    Soft 3-class Ramachandran classification per residue.
    Returns [n_res, 3] float tensor: [helix_score, sheet_score, other_score].
    Uses hard binary rules; output is one-hot-like [0/1 per residue].
    """
    # Alpha-helix region: phi in [-90, -40], psi in [-70, -10] (radians)
    import math
    h_phi = (phi > math.radians(-90)) & (phi < math.radians(-40))
    h_psi = (psi > math.radians(-70)) & (psi < math.radians(-10))
    helix = (h_phi & h_psi).float()

    # Beta-sheet region: phi in [-170, -50], psi in [100, 180]
    s_phi = (phi > math.radians(-170)) & (phi < math.radians(-50))
    s_psi = (psi > math.radians(100)) & (psi < math.radians(180))
    sheet = (s_phi & s_psi).float() * (1.0 - helix)  # no overlap

    other = 1.0 - helix - sheet
    return torch.stack([helix, sheet, other], dim=-1)  # [n_res, 3]


# ── V2 confidence model ───────────────────────────────────────────────────────

class V2ConfidenceModel(CGTensorProductEquivariantModel):
    """
    Drop-in replacement for CGTensorProductEquivariantModel in confidence mode.

    Load pretrained weights with strict=False: backbone weights load cleanly;
    the new v2 layers (chi1_proj, ss_proj, confidence_predictor_v2) are
    randomly initialised and trained from scratch.
    """

    V2_SIDECHAIN_DIM = 16   # projected sidechain-orientation feature dim
    V2_SS_DIM        = 3    # Ramachandran SS proxy dim

    def __init__(self, args):
        super().__init__(args, confidence_mode=True, num_confidence_outputs=1)

        ns = self.ns  # 48 by default
        v2_dim = ns * 2 + self.V2_SIDECHAIN_DIM + self.V2_SS_DIM  # 115

        # Sidechain orientation proxy: CA→tip distance → 16-dim
        self.chi1_proj = nn.Linear(1, self.V2_SIDECHAIN_DIM)

        # Receptor SS context: 3-dim mean-pooled one-hot → pass-through (no projection needed)
        # Concatenated directly.

        # Replaced confidence head (v1 head is 96-dim input; v2 is 115-dim)
        self.confidence_predictor_v2 = nn.Sequential(
            nn.Linear(v2_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        # Keep v1 confidence_predictor in place so pretrained weights load cleanly
        # (it won't be called in v2 forward, but its weights are ignored anyway)

    def forward(self, _data) -> torch.Tensor:
        data = copy.copy(_data)

        # ── noise schedule (confidence mode: sigma = t directly) ─────────────
        tr_t            = data.complex_t["tr"]
        rot_t           = data.complex_t["rot"]
        tor_backbone_t  = data.complex_t["tor_backbone"]
        tor_sidechain_t = data.complex_t["tor_sidechain"]
        tr_sigma, rot_sigma, tor_backbone_sigma, tor_sidechain_sigma = (
            tr_t, rot_t, tor_backbone_t, tor_sidechain_t
        )

        self.device = data["pep"].x.device

        # ── inject batch/ptr for single-graph (non-batched) inference ─────────
        for node_type in ("pep_a", "pep", "receptor"):
            store = data[node_type]
            if not hasattr(store, "batch") or store.batch is None:
                n = store.x.shape[0] if (hasattr(store, "x") and store.x is not None) else store.pos.shape[0]
                store.batch = torch.zeros(n, dtype=torch.long, device=self.device)
            if not hasattr(store, "ptr") or store.ptr is None:
                n = store.batch.shape[0]
                store.ptr = torch.tensor([0, n], dtype=torch.long, device=self.device)

        # ── receptor graph ────────────────────────────────────────────────────
        receptor_graph = self.build_rec_conv_graph(data)
        rec_node_attr  = self.rec_node_embedding(receptor_graph[0])
        rec_src, rec_dst = rec_edge_index = receptor_graph[1]
        rec_edge_attr  = self.rec_edge_embedding(receptor_graph[2])
        rec_edge_sh    = receptor_graph[3]

        # ── Receptor SS proxy (before embeddings overwrite raw features) ──────
        # data['receptor'].x shape: [n_rec, 1+9+(1280 optional)]
        # x[:,6] = phi, x[:,7] = psi (from get_protein_feature_mda dihedral order)
        rec_x_raw = data["receptor"].x
        if rec_x_raw.shape[1] >= 8:
            rec_phi = rec_x_raw[:, 6].float()
            rec_psi = rec_x_raw[:, 7].float()
            ss_onehot = _ramachandran_class(rec_phi, rec_psi).to(self.device)
            rec_batch = (
                data["receptor"].batch
                if hasattr(data["receptor"], "batch") and data["receptor"].batch is not None
                else torch.zeros(ss_onehot.shape[0], dtype=torch.long, device=self.device)
            )
            rec_ss_mean = scatter_mean(ss_onehot, rec_batch, dim=0)  # [batch, 3]
        else:
            n_graphs = data["pep"].batch.max().item() + 1
            rec_ss_mean = torch.zeros(n_graphs, self.V2_SS_DIM, device=self.device)

        # ── peptide graph ─────────────────────────────────────────────────────
        node_s_pep, ca_pep, tips_pep, edge_index_pep, edge_s_pep, edge_v_pep = (
            get_updated_peptide_feature(data, self.device, self.top_k)
        )
        data["pep"].x = (
            torch.cat([data["pep"].x[:, :1], node_s_pep, data["pep"].x[:, -1280:]], axis=1)
            if self.esm_embeddings_peptide
            else torch.cat([data["pep"].x[:, :1], node_s_pep], axis=1)
        )
        data["pep"].pos  = ca_pep.to(dtype=torch.float)
        data["pep"].tips = tips_pep.to(dtype=torch.float)
        data["pep", "pep_contact", "pep"].edge_index = edge_index_pep
        data["pep", "pep_contact", "pep"].edge_s     = edge_s_pep.to(dtype=torch.float)
        data["pep", "pep_contact", "pep"].edge_v     = edge_v_pep.to(dtype=torch.float)

        # ── sidechain orientation proxy: CA→tip distance ──────────────────────
        # Both tensors are [n_pep_res, 3] after get_updated_peptide_feature
        ca_tip_dist = (data["pep"].tips - data["pep"].pos).norm(
            dim=-1, keepdim=True
        )  # [n_pep_res, 1]

        pep_graph     = self.build_pep_conv_graph(data)
        pep_node_attr = self.pep_node_embedding(pep_graph[0])
        pep_src, pep_dst = pep_edge_index = pep_graph[1]
        pep_edge_attr = self.pep_edge_embedding(pep_graph[2])
        pep_edge_sh   = pep_graph[3]

        # ── cross graph ───────────────────────────────────────────────────────
        if self.dynamic_max_cross:
            cross_cutoff = (
                tr_sigma * self.cross_cutoff_weight + self.cross_cutoff_bias
            ).unsqueeze(1)
        else:
            cross_cutoff = self.cross_max_distance
        cross_edge_index, cross_edge_attr, cross_edge_sh = self.build_cross_conv_graph(
            data, cross_cutoff
        )
        cross_pep, cross_rec = cross_edge_index
        cross_edge_attr = self.cross_edge_embedding(cross_edge_attr)

        # ── message passing ───────────────────────────────────────────────────
        last_inter_update = None
        for idx in range(len(self.intra_convs)):
            pep_edge_attr_ = torch.cat(
                [pep_edge_attr,
                 pep_node_attr[pep_src,  :self.ns],
                 pep_node_attr[pep_dst,  :self.ns]], -1
            )
            pep_intra_update = self.intra_convs[idx](
                pep_node_attr, pep_edge_index, pep_edge_attr_, pep_edge_sh
            )

            rec_to_pep_edge_attr_ = torch.cat(
                [cross_edge_attr,
                 pep_node_attr[cross_pep, :self.ns],
                 rec_node_attr[cross_rec, :self.ns]], -1
            )
            pep_inter_update = self.cross_convs[idx](
                rec_node_attr, cross_edge_index,
                rec_to_pep_edge_attr_, cross_edge_sh,
                out_nodes=pep_node_attr.shape[0]
            )

            if idx == len(self.intra_convs) - 1:
                last_inter_update = pep_inter_update  # save for attention weights

            if idx != len(self.intra_convs) - 1:
                rec_edge_attr_ = torch.cat(
                    [rec_edge_attr,
                     rec_node_attr[rec_src, :self.ns],
                     rec_node_attr[rec_dst, :self.ns]], -1
                )
                rec_intra_update = self.intra_convs[idx](
                    rec_node_attr, rec_edge_index, rec_edge_attr_, rec_edge_sh
                )
                pep_to_rec_edge_attr_ = torch.cat(
                    [cross_edge_attr,
                     pep_node_attr[cross_pep, :self.ns],
                     rec_node_attr[cross_rec, :self.ns]], -1
                )
                rec_inter_update = self.cross_convs[idx](
                    pep_node_attr, torch.flip(cross_edge_index, dims=[0]),
                    pep_to_rec_edge_attr_, cross_edge_sh,
                    out_nodes=rec_node_attr.shape[0]
                )

            pep_node_attr = F.pad(
                pep_node_attr,
                (0, pep_intra_update.shape[-1] - pep_node_attr.shape[-1])
            )
            pep_node_attr = pep_node_attr + pep_intra_update + pep_inter_update

            if idx != len(self.intra_convs) - 1:
                rec_node_attr = F.pad(
                    rec_node_attr,
                    (0, rec_intra_update.shape[-1] - rec_node_attr.shape[-1])
                )
                rec_node_attr = rec_node_attr + rec_intra_update + rec_inter_update

        # ── V2 confidence head ────────────────────────────────────────────────

        # 1. Scalar peptide features [n_pep_res, ns*2]
        scalar_pep = (
            torch.cat([pep_node_attr[:, :self.ns], pep_node_attr[:, -self.ns:]], dim=1)
            if self.num_conv_layers >= 3
            else pep_node_attr[:, :self.ns]
        )  # [n_pep_res, 96]

        # 2. Cross-attention weights from last inter-update norms [n_pep_res]
        if last_inter_update is not None:
            cross_norm = last_inter_update[:, :self.ns].norm(dim=-1)  # [n_pep_res]
            attn = _batch_softmax(cross_norm, data["pep"].batch)       # [n_pep_res]
            attn_w = attn.unsqueeze(-1)                                 # [n_pep_res, 1]
        else:
            n_pep = scalar_pep.shape[0]
            attn_w = torch.ones(n_pep, 1, device=self.device)
            # Normalise per sample
            sum_attn = scatter_add(attn_w, data["pep"].batch, dim=0)[data["pep"].batch]
            attn_w = attn_w / (sum_attn + 1e-8)

        # 3. Attention-weighted pooling [batch, 96]
        pooled_pep = scatter_add(
            scalar_pep * attn_w, data["pep"].batch, dim=0
        )

        # 4. Sidechain proxy: project and pool [batch, 16]
        sc_feat  = self.chi1_proj(ca_tip_dist)              # [n_pep_res, 16]
        pooled_sc = scatter_add(sc_feat * attn_w, data["pep"].batch, dim=0)  # [batch, 16]

        # 5. Receptor SS proxy [batch, 3]  (already mean-pooled above)

        # 6. Concatenate and predict
        h = torch.cat([pooled_pep, pooled_sc, rec_ss_mean], dim=-1)  # [batch, 115]
        confidence = self.confidence_predictor_v2(h).squeeze(dim=-1)
        return confidence
