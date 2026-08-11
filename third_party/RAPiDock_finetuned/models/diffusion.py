##########################################################################
# File Name: diffusion.py
# Author: huifeng
# mail: huifengzhao@zju.edu.cn
# Created Time: Fri 20 Oct 2023 01:09:20 PM CST
#########################################################################

import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_cluster import radius
from torch_scatter import scatter, scatter_mean
from e3nn import o3
from e3nn.nn import BatchNorm
from utils.diffusion_utils import get_timestep_embedding
from utils.transform import NoiseSchedule
from utils.so3 import score_norm
from utils.torus import score_norm as torus_score
from dataset.peptide_feature import three2idx, allowable_features, atomname2idx, get_updated_peptide_feature

print("Loading diffusion model...")

feature_dims = [
            len(three2idx), max([len(res) for res in atomname2idx]) # Amino_idx_dim/ Atom_idx_dim
        ] + \
        [
            len(value) for key, value in allowable_features.items() # Atom_features_dim
        ] + [4] # Atom_charity_center_dim

class GaussianSmearing(nn.Module):
    # used to embed the edge dists
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))
    
class AminoEmbedding(nn.Module):
    """
        Embeddings for atom identity only (as of now)
    """
    def __init__(self, num_amino, emb_dim, sigma_embed_dim, intra_dim, dihediral_dim, lm_embedding_type = None):
        super(AminoEmbedding, self).__init__()
        self.amino_type_dim = 1
        self.sigma_embed_dim = sigma_embed_dim
        self.intra_dim = intra_dim
        self.dihediral_dim = dihediral_dim
        self.lm_embedding_type = lm_embedding_type
        
        self.amino_ebd = nn.Embedding(num_amino+1, emb_dim) # add 1 for padding
        self.sigma_ebd = nn.Linear(sigma_embed_dim, emb_dim)
        self.intra_dis_ebd = nn.Linear(intra_dim, emb_dim)
        self.dihediral_angles_ebd = nn.Linear(dihediral_dim, emb_dim)
        
        # LM embedding (ESM2)
        if self.lm_embedding_type is not None:
            if self.lm_embedding_type == 'esm':
                self.lm_embedding_dim = 1280
                self.lm_embedding_layer = nn.Linear(self.lm_embedding_dim + emb_dim, emb_dim)
            else: raise ValueError('LM Embedding type was not correctly determined. LM embedding type: ', self.lm_embedding_type)

    def forward(self, x):
        if self.lm_embedding_type is not None:
            assert x.shape[1] == self.amino_type_dim + self.sigma_embed_dim + self.intra_dim + self.dihediral_dim + self.lm_embedding_dim
        else:
            assert x.shape[1] == self.amino_type_dim + self.sigma_embed_dim + self.intra_dim + self.dihediral_dim
        
        x_embedding = 0
        # amino sequence
        x_embedding += self.amino_ebd(x[:,:self.amino_type_dim].long()).squeeze() # amino_ebd
        # sigma noise embedding
        x_embedding += self.sigma_ebd(x[:,-self.sigma_embed_dim:].to(torch.float32)) # sigma_ebd
        x_embedding += self.intra_dis_ebd(x[:,self.amino_type_dim:self.amino_type_dim+self.intra_dim].to(torch.float32)) # intra_dis_ebd
        x_embedding += self.dihediral_angles_ebd(x[:,self.amino_type_dim+self.intra_dim:self.amino_type_dim+self.intra_dim+self.dihediral_dim].to(torch.float32)) # dihediral_angles_ebd
        # # consider LM embedding here
        if self.lm_embedding_type is not None:
            x_embedding =  self.lm_embedding_layer(torch.cat([x_embedding, x[:, self.amino_type_dim+self.intra_dim+self.dihediral_dim:self.amino_type_dim+self.intra_dim+self.dihediral_dim+self.lm_embedding_dim].to(torch.float32)], dim=1))
        return x_embedding

class AtomEncoder(nn.Module):

    def __init__(self, emb_dim, sigma_embed_dim, pep_attr_dim):
        # first element of feature_dims tuple is a list with the lenght of each categorical feature and the second is the number of scalar features
        super(AtomEncoder, self).__init__()
        self.atom_embedding_list = torch.nn.ModuleList()
        self.num_categorical_features = len(feature_dims)
        self.num_scalar_features = sigma_embed_dim
        self.emb_dim = emb_dim
        self.pep_attr_dim = pep_attr_dim
        for i, dim in enumerate(feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

        if self.num_scalar_features > 0:
            self.linear = torch.nn.Linear(self.num_scalar_features, emb_dim)
        self.final_layer = torch.nn.Linear(emb_dim + pep_attr_dim, emb_dim)

    def forward(self, x):
        x_embedding = 0
        assert x.shape[1] == self.num_categorical_features + self.num_scalar_features + self.pep_attr_dim
        for i in range(self.num_categorical_features):
            x_embedding += self.atom_embedding_list[i](x[:, i].long())

        if self.num_scalar_features > 0:
            x_embedding += self.linear(x[:, self.num_categorical_features:self.num_categorical_features + self.num_scalar_features])
        x_embedding = self.final_layer(torch.cat([x_embedding, x[:, -self.pep_attr_dim:]], axis=1))
        return x_embedding
    
class EdgeEmbedding(nn.Module):
    """
        Embeddings for edge feature (as of now)
    """
    def __init__(self, ns,sigma_embed_dim=32,feature_embed_dim=103, dropout=0.0):
        self.connect_dim = 1
        self.sigma_embed_dim = sigma_embed_dim
        self.feature_embed_dim = feature_embed_dim
        
        super(EdgeEmbedding, self).__init__()
        self.connect_ebd = nn.Embedding(2, ns) # 0,1
        self.sigma_ebd = nn.Linear(self.sigma_embed_dim, ns)
        self.feature_ebd = nn.Linear(self.feature_embed_dim, ns)
        self.edge_embedding = nn.Sequential(nn.Linear(ns*3, ns),nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))
        # _init(self)

    def forward(self, x):
        # connect
        connect_ebd = self.connect_ebd(x[:,self.sigma_embed_dim:self.sigma_embed_dim+self.connect_dim].long()).squeeze()
        # sigma noise embedding
        sigma_ebd = self.sigma_ebd(x[:,:self.sigma_embed_dim].to(torch.float32))
        # 
        feature_ebd = self.feature_ebd(x[:,-self.feature_embed_dim:].to(torch.float32))
        # # add together
        x_embedding = torch.cat([connect_ebd, sigma_ebd, feature_ebd], 1)
        
        x_embedding = self.edge_embedding(x_embedding)
        return x_embedding
    
class CGTPEL(nn.Module):
    """
        Clebsch-Gordan tensor product equivariant layer
    """
    def __init__(self, 
                 in_irreps, sh_irreps, out_irreps, n_edge_features,
                 residual=True, batch_norm=True,hidden_features=None, is_last_layer=False,dropout=0.0):
        super(CGTPEL, self).__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual
        if hidden_features is None:
            hidden_features = n_edge_features

        self.tensor_prod = o3.FullyConnectedTensorProduct(
            in_irreps, sh_irreps, out_irreps, shared_weights=False
        )

        self.fc = nn.Sequential(
            nn.Linear(n_edge_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, self.tensor_prod.weight_numel),
        )
        self.batch_norm = BatchNorm(out_irreps) if (batch_norm and not is_last_layer) else None

    def forward(
        self,
        node_attr,
        edge_index,
        edge_attr,
        edge_sh,
        out_nodes=None,
        reduction="mean",
    ):
        """
        @param edge_index  [2, E]
        @param edge_sh  edge spherical harmonics
        """
        edge_src, edge_dst = edge_index
        tp = self.tensor_prod(node_attr[edge_dst], edge_sh, self.fc(edge_attr))
        # DTYPE FIX (May 28 2026): e3nn FullyConnectedTensorProduct returns int32 when
        # given empty input tensors (0 edges — happens when diffusion noise moves peptide
        # outside the cross-contact cutoff).  .float() is a no-op for float32 and costs
        # nothing in the normal (non-empty) case.
        tp = tp.float()

        out_nodes = out_nodes or node_attr.shape[0]
        out = scatter(tp, edge_src, dim=0, dim_size=out_nodes, reduce=reduction)

        if self.residual:
            new_shape = (0, out.shape[-1] - node_attr.shape[-1])
            padded = F.pad(node_attr, new_shape)
            out = out + padded

        if self.batch_norm:
            out = self.batch_norm(out)
        return out

class CGTensorProductEquivariantModel(nn.Module):
    """
        Clebsch-Gordan tensor product-based equivariant model
    """
    def __init__(self, args,
            confidence_mode=False,
            confidence_dropout=0,
            confidence_no_batchnorm=False,
            num_confidence_outputs=1,
        ):
        super(CGTensorProductEquivariantModel, self).__init__()
        
        self.dropout = args.dropout
        self.cross_cutoff_weight = args.cross_cutoff_weight
        self.cross_cutoff_bias = args.cross_cutoff_bias
        self.cross_max_dist = args.cross_max_distance
        self.dynamic_max_cross=args.dynamic_max_cross
        self.center_max_dist = args.center_max_distance
        
        self.ns, self.nv = args.ns, args.nv
        ns, nv = self.ns, self.nv
        self.num_conv_layers = args.num_conv_layers
        self.lig_max_radius = args.max_radius
        self.batch_norm = not args.no_batch_norm
        self.confidence_mode = confidence_mode
        self.scale_by_sigma = args.scale_by_sigma
        # M2 fix (optional): floor tr_sigma before the 1/sigma score scaling to cap
        # low-sigma amplification of the translation head. 0.0 = off (behaviour unchanged).
        # NB tr_sigma_min is already 0.1 in the schedule (1/sigma <= 10x), so this is a
        # weak lever here; the real collapse driver is weight drift growing the raw output.
        self.tr_sigma_floor = float(getattr(args, 'tr_sigma_floor', 0.0))
        self.rec_amino_dim = args.rec_amino_dim
        self.pep_amino_dim = args.pep_amino_dim 
        self.sigma_embed_dim = args.sigma_embed_dim
        self.intra_dim = args.intra_dim
        self.dihediral_dim = args.dihediral_dim
        self.edge_feature_dim = args.edge_feature_dim
        self.embedding_type = args.embedding_type
        self.embedding_scale = args.embedding_scale
        self.cross_dist_embed_dim = args.cross_distance_embed_dim
        self.use_second_order_repr = args.use_second_order_repr
        self.dist_embed_dim = args.distance_embed_dim
        self.esm_embeddings_peptide = args.esm_embeddings_peptide_train is not None
        self.noise_schedule = NoiseSchedule(args) if not confidence_mode else None
        self.timestep_emb_func = get_timestep_embedding(self.embedding_type,self.sigma_embed_dim,self.embedding_scale)
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=2)
        self.rec_node_embedding = AminoEmbedding(self.rec_amino_dim,ns,self.sigma_embed_dim,self.intra_dim, self.dihediral_dim, 'esm')
        self.rec_edge_embedding = EdgeEmbedding(ns,self.sigma_embed_dim, self.edge_feature_dim, self.dropout)
        self.pep_node_embedding = AminoEmbedding(self.pep_amino_dim,ns,self.sigma_embed_dim,self.intra_dim, 3, 'esm') if self.esm_embeddings_peptide else AminoEmbedding(self.pep_amino_dim,ns,self.sigma_embed_dim,self.intra_dim, 3)
        self.pep_edge_embedding = EdgeEmbedding(ns,self.sigma_embed_dim, self.edge_feature_dim, self.dropout)
        self.top_k = args.top_k
        self.cross_edge_embedding = nn.Sequential(
            nn.Linear(self.sigma_embed_dim + self.cross_dist_embed_dim, ns),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(ns, ns),
        )
        self.cross_distance_expansion = GaussianSmearing(
                0.0, self.cross_max_dist, self.cross_dist_embed_dim)
        
        if self.use_second_order_repr:
            irrep_seq = [
                f"{ns}x0e",
                f"{ns}x0e + {nv}x1o + {nv}x2e",
                f"{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o",
                f"{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o",
            ]
        else:
            irrep_seq = [
                f"{ns}x0e",
                f"{ns}x0e + {nv}x1o",
                f"{ns}x0e + {nv}x1o + {nv}x1e",
                f"{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o",
            ]
        
        intra_convs = []
        cross_convs = []
        for i in range(self.num_conv_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            params = {
                "in_irreps": in_irreps,
                "sh_irreps": self.sh_irreps,
                "out_irreps": out_irreps,
                "n_edge_features": 3 * ns,
                "hidden_features": 3 * ns,
                "residual": False,
                'batch_norm': self.batch_norm,
                'dropout': self.dropout
            }
            intra_convs.append(CGTPEL(**params))
            cross_convs.append(CGTPEL(**params))
            
        self.intra_convs = nn.ModuleList(intra_convs)
        self.cross_convs = nn.ModuleList(cross_convs)
        
        # compute confidence score
        if self.confidence_mode:
            self.confidence_predictor = nn.Sequential(
                nn.Linear(2 * self.ns if self.num_conv_layers >= 3 else self.ns, ns),
                nn.BatchNorm1d(ns) if not confidence_no_batchnorm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(confidence_dropout),
                nn.Linear(ns, ns),
                nn.BatchNorm1d(ns) if not confidence_no_batchnorm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(confidence_dropout),
                nn.Linear(ns, num_confidence_outputs),
            )
        else:
            # center of mass translation and rotation components
            self.center_distance_expansion = GaussianSmearing(
                0.0, self.center_max_dist, self.dist_embed_dim
            )
            self.center_edge_embedding = nn.Sequential(
                nn.Linear(self.dist_embed_dim + self.sigma_embed_dim, ns),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(ns, ns)
            )
            
            self.final_conv = CGTPEL(
                in_irreps=self.intra_convs[-1].out_irreps,
                sh_irreps=self.sh_irreps,
                out_irreps=f'2x1o + 2x1e',
                n_edge_features=2 * ns,
                residual=False,
                dropout=self.dropout,
                batch_norm= self.batch_norm,
                is_last_layer=True,
            )
            
            self.tr_final_layer = nn.Sequential(nn.Linear(1 + self.sigma_embed_dim, ns),nn.Dropout(self.dropout), nn.ReLU(), nn.Linear(ns, 1))
            self.rot_final_layer = nn.Sequential(nn.Linear(1 + self.sigma_embed_dim, ns),nn.Dropout(self.dropout), nn.ReLU(), nn.Linear(ns, 1))
            
            # Build Pep_atom_ebd
            self.pep_a_node_embedding = AtomEncoder(o3.Irreps(self.intra_convs[-1].out_irreps).dim,self.sigma_embed_dim, o3.Irreps(self.intra_convs[-1].out_irreps).dim)
            
            self.lig_distance_expansion = GaussianSmearing(
                0.0, self.lig_max_radius, self.dist_embed_dim)
            
            # torsion angles components
            self.final_edge_embedding = nn.Sequential(
                    nn.Linear(self.dist_embed_dim, ns),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(ns, ns),
                )
            self.final_tp_tor_bb = o3.FullTensorProduct(self.sh_irreps, "2e")
            self.final_tp_tor_sc = o3.FullTensorProduct(self.sh_irreps, "2e")
            self.tor_bb_bond_conv = CGTPEL(
                    in_irreps=self.intra_convs[-1].out_irreps,
                    sh_irreps=self.final_tp_tor_bb.irreps_out,
                    out_irreps=f'{ns}x0o + {ns}x0e',
                    n_edge_features=3 * ns,
                    residual=False,
                    dropout=self.dropout,
                    batch_norm=self.batch_norm
                )
            self.tor_sc_bond_conv = CGTPEL(
                    in_irreps=self.intra_convs[-1].out_irreps,
                    sh_irreps=self.final_tp_tor_sc.irreps_out,
                    out_irreps=f'{ns}x0o + {ns}x0e',
                    n_edge_features=3 * ns,
                    residual=False,
                    dropout=self.dropout,
                    batch_norm=self.batch_norm
                )
            self.tor_bb_final_layer = nn.Sequential(
                    nn.Linear(2 * ns, ns, bias=False),
                    nn.SiLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(ns, 1, bias=False)
                )
            self.tor_sc_final_layer = nn.Sequential(
                    nn.Linear(2 * ns, ns, bias=False),
                    nn.SiLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(ns, 1, bias=False)
                )
            # INJECT FIX: cross-edge contact type embedding (additive, zero-init → neutral start)
            # Encodes which of 4 contact types (CA-CA, tip-CA, CA-tip, tip-tip) gives
            # the minimum distance per cross edge.  Zero-init means existing checkpoints
            # load correctly and the embedding learns from scratch during inject finetune.
            self.cross_type_embedding = nn.Embedding(4, ns)
            nn.init.zeros_(self.cross_type_embedding.weight)

            # POSITION ADAPTER (Jul 30 2026): zero-init equivariant translation adapter.
            # Frozen-base + ControlNet-style zero-conv: fresh capacity DEDICATED TO PLACEMENT.
            # Zeroing the fc final layer makes its output EXACTLY 0 at init, so the model
            # reproduces pretrained bit-for-bit; gradients still flow into fc, so it learns an
            # ADDITIVE long-peptide translation correction without ever touching the pretrained
            # placement weights (which diverged when finetuned directly).  Registered LAST so its
            # params sit at the tail of model.parameters(): pretrained EMA/state_dict (which lack
            # them) load by position/zip and leave the adapter at its zero-init.
            self.use_tr_adapter = getattr(args, 'use_tr_adapter', False)
            if self.use_tr_adapter:
                # Aug 2026 cliff-v2 fix: hidden_features widened by tr_adapter_hidden_mult
                # (default 1.0 = old behaviour, n_edge_features=2*ns). More capacity for the
                # adapter to learn a nuanced, well-calibrated SMALL correction rather than
                # being capacity-starved into coarse over/under-shoots.
                _hidden_mult = float(getattr(args, 'tr_adapter_hidden_mult', 1.0))
                self.tr_adapter_conv = CGTPEL(
                    in_irreps=self.intra_convs[-1].out_irreps,
                    sh_irreps=self.sh_irreps,
                    out_irreps='1x1o + 1x1e',
                    n_edge_features=2 * ns,
                    hidden_features=int(round(2 * ns * _hidden_mult)),
                    residual=False,
                    dropout=self.dropout,
                    batch_norm=self.batch_norm,
                    is_last_layer=True,
                )
                nn.init.zeros_(self.tr_adapter_conv.fc[-1].weight)
                nn.init.zeros_(self.tr_adapter_conv.fc[-1].bias)
                # Max magnitude (pre-sigma score units) of the additive translation correction.
                self.tr_adapter_cap = float(getattr(args, 'tr_adapter_cap', 1.0))
                # Aug 2026 cliff-v2 fix (outlier-pose root cause): the adapter's PHYSICAL
                # displacement = tr_sigma^2 * (score-space correction) -- traced empirically to
                # reach 20+A per step at high sigma even though the raw score-space output is
                # tiny there, because sigma^2 amplification (up to ~900x at sigma=30 vs sigma=1)
                # outpaces the output's own shrinkage. That regime (tr_sigma above roughly where
                # the x0 direct-placement loss stops supervising) is where the adapter has never
                # been taught what a good correction looks like, yet its potential impact is
                # LARGEST there -- measured as a 33% vs 2% (pretrained) rate of 80-190A outlier
                # poses across the 52-complex benchmark. Fix: taper tr_corr linearly to zero
                # between tr_adapter_gate_hi and tr_sigma_max, so the model falls back to the
                # base's own (well-calibrated, full-pretraining-trained) coarse placement in
                # exactly the sigma range the adapter was never supervised in. Defaults to
                # tr_sigma_max (no gate = old behaviour) if not set; train_lastlayer.py sets it
                # to match --x0-sigma-hi by default so the applied range always matches the
                # supervised range.
                self.tr_adapter_gate_hi = float(getattr(args, 'tr_adapter_gate_hi', args.tr_sigma_max))
                self.tr_sigma_max_for_gate = float(getattr(args, 'tr_sigma_max', 30.0))
                # Aug 2026 cliff-v3 fix (LOW-sigma outlier root cause, found via cliffmixed
                # forensics): the high-sigma gate above only protects the EARLY/coarse part of
                # the trajectory. tr_corr is ALSO divided by tr_sigma (a few lines below, mirrors
                # the frozen base's own scaling) -- at LOW sigma (late, fine-refinement steps)
                # this is completely ungated and can amplify a score-space-capped correction by
                # 10-30x in physical terms, on the very last steps where there's no time left for
                # the base's own signal to correct it back. Empirically: full-150-complex sweep,
                # cliffmixed_final checkpoint, poses >20A from crystal correlate with peptide
                # length at r=0.699 (very_long: 36% of ALL poses outliers; short: 0.5%) --
                # consistent with a low-sigma, ungated, length-correlated bias, not the
                # already-fixed high-sigma mechanism (worst individual outliers reached 263A with
                # near-perfect conformation, i.e. rmsd_superposed 2-16A -- a pure placement error).
                # Fix: a SECOND cap applied AFTER the /tr_sigma amplification, bounding the
                # physical displacement directly in Angstroms regardless of sigma or peptide
                # length. Default 4.0 A: generous enough to matter (the x0 loss's own --x0-cap
                # implies placement errors up to ~20A are being corrected for during training) but
                # small enough that even every one of the 16 sampling steps hitting the cap in the
                # same direction (worst case, no cancellation) caps total possible adapter-driven
                # drift at 64A -- and in practice steps partially cancel, not compound linearly.
                self.tr_adapter_physical_cap = float(getattr(args, 'tr_adapter_physical_cap', 4.0))
                # Length-scaled cap decay rate r (see forward() for the formula). 0.0 = old flat
                # cap (backward compatible). Small positive r shrinks the cap for longer peptides
                # specifically, targeting the r=0.699 length-correlated outlier mechanism directly
                # instead of a single global bound that helps short peptides (already fine) as
                # much as long ones (the ones actually failing).
                self.tr_adapter_cap_length_scale = float(getattr(args, 'tr_adapter_cap_length_scale', 0.0))
                # Aug 2026 cliff-v4 hypothesis (cheap-tested before retrain, per lesson from the
                # v2/v3 magnitude-cap failures): the outliers aren't a magnitude problem at all --
                # capping tr_corr flat OR length-scaled left the 4 worst-offender complexes at
                # 21-30A best-of-32 either way (one even got WORSE: 26.3A -> 29.7A on a 40-mer with
                # the cap shrunk to ~0.02A/step). That rules out per-step amplitude as the driver:
                # the drift must be SIGN-CONSISTENT across the ~16 sampling steps (a capped-small
                # step repeated in the same direction 16x still compounds to the same wrong place).
                # A magnitude cap can never fix a directional bias. This param instead ZEROES
                # tr_corr completely below a sigma threshold (hard cutoff, no taper -- deliberately
                # blunt for a quick falsifiability test) so the adapter can influence early/coarse
                # placement but cannot contribute anything during late refinement, where the
                # ungated /tr_sigma amplification (see two comments above) does its damage. 0.0 =
                # disabled (old behaviour).
                self.tr_adapter_zero_below_sigma = float(
                    getattr(args, 'tr_adapter_zero_below_sigma', 0.0))

    def forward(self, _data):
        data = copy.copy(_data)
        # get noise schedule
        tr_t = data.complex_t["tr"]
        rot_t = data.complex_t["rot"]
        tor_backbone_t = data.complex_t["tor_backbone"]
        tor_sidechain_t = data.complex_t["tor_sidechain"]
        if not self.confidence_mode:
            tr_sigma, rot_sigma, tor_backbone_sigma,tor_sidechain_sigma = self.noise_schedule(tr_t, rot_t, tor_backbone_t,tor_sidechain_t)
        else:
            tr_sigma, rot_sigma, tor_backbone_sigma,tor_sidechain_sigma = tr_t, rot_t, tor_backbone_t,tor_sidechain_t
            
        self.device = data['pep'].x.device

        # build receptor graph
        receptor_graph = self.build_rec_conv_graph(data)
        rec_node_attr = self.rec_node_embedding(receptor_graph[0])
        rec_src, rec_dst = rec_edge_index = receptor_graph[1]
        rec_edge_attr = self.rec_edge_embedding(receptor_graph[2])
        rec_edge_sh = receptor_graph[3]
        
        # build pep graph
        node_s_pep, ca_pep, tips_pep, edge_index_pep, edge_s_pep, edge_v_pep = get_updated_peptide_feature(data, self.device, self.top_k)
        data['pep'].x = torch.cat([data['pep'].x[:,:1], node_s_pep, data['pep'].x[:,-1280:]], axis=1) if self.esm_embeddings_peptide else torch.cat([data['pep'].x[:,:1], node_s_pep], axis=1)  # [num_res, 1+5+3(+1280)]
        data['pep'].pos = ca_pep.to(dtype=torch.float)
        data['pep'].tips = tips_pep.to(dtype=torch.float)
        data['pep', 'pep_contact', 'pep'].edge_index = edge_index_pep
        data['pep', 'pep_contact', 'pep'].edge_s = edge_s_pep.to(dtype=torch.float)
        data['pep', 'pep_contact', 'pep'].edge_v = edge_v_pep.to(dtype=torch.float)
        
        pep_graph = self.build_pep_conv_graph(data)
        pep_node_attr = self.pep_node_embedding(pep_graph[0])
        pep_src, pep_dst = pep_edge_index = pep_graph[1]
        pep_edge_attr = self.pep_edge_embedding(pep_graph[2])
        pep_edge_sh = pep_graph[3]
        
        # build cross graph
        if self.dynamic_max_cross:
            cross_cutoff = (tr_sigma * self.cross_cutoff_weight + self.cross_cutoff_bias).unsqueeze(1)
        else:
            cross_cutoff = self.cross_max_distance
        cross_edge_index, cross_edge_attr, cross_edge_sh, cross_type_indices = self.build_cross_conv_graph(
            data, cross_cutoff
        )
        cross_pep, cross_rec = cross_edge_index
        cross_edge_attr = self.cross_edge_embedding(cross_edge_attr)
        # INJECT FIX: add contact-type signal (zero-init → no-op until trained)
        if not self.confidence_mode:
            cross_edge_attr = cross_edge_attr + self.cross_type_embedding(cross_type_indices.long())
        
        for idx in range(len(self.intra_convs)):
            # message passing within pep graph (intra)
            pep_edge_attr_ = torch.cat([pep_edge_attr, pep_node_attr[pep_src, :self.ns], pep_node_attr[pep_dst, :self.ns]], -1)
            pep_intra_update = self.intra_convs[idx](pep_node_attr, pep_edge_index, pep_edge_attr_, pep_edge_sh)
            
            # message passing between two graphs (inter)
            rec_to_pep_edge_attr_ = torch.cat([cross_edge_attr, pep_node_attr[cross_pep, :self.ns], rec_node_attr[cross_rec, :self.ns]], -1)
            pep_inter_update = self.cross_convs[idx](rec_node_attr, cross_edge_index, rec_to_pep_edge_attr_, cross_edge_sh, out_nodes=pep_node_attr.shape[0])
            
            # message passing within receptor graph (intra)
            if idx != len(self.intra_convs) - 1:
                rec_edge_attr_ = torch.cat([rec_edge_attr, rec_node_attr[rec_src, :self.ns], rec_node_attr[rec_dst, :self.ns]], -1)
                rec_intra_update = self.intra_convs[idx](rec_node_attr, rec_edge_index, rec_edge_attr_, rec_edge_sh)
                
                pep_to_rec_edge_attr_ = torch.cat([cross_edge_attr, pep_node_attr[cross_pep, :self.ns], rec_node_attr[cross_rec, :self.ns]], -1)
                rec_inter_update = self.cross_convs[idx](pep_node_attr, torch.flip(cross_edge_index, dims=[0]), pep_to_rec_edge_attr_, cross_edge_sh, out_nodes=rec_node_attr.shape[0])
            
            # padding original features
            pep_node_attr = F.pad(
                pep_node_attr,
                (0, pep_intra_update.shape[-1] - pep_node_attr.shape[-1]))
            # update features with residual updates
            pep_node_attr = pep_node_attr + pep_intra_update + pep_inter_update
            
            if idx != len(self.intra_convs) - 1:
                rec_node_attr = F.pad(rec_node_attr, (0, rec_intra_update.shape[-1] - rec_node_attr.shape[-1]))
                rec_node_attr = rec_node_attr + rec_intra_update + rec_inter_update
                
        # compute confidence score
        if self.confidence_mode:
            scalar_pep_attr = (
                torch.cat(
                    [pep_node_attr[:, : self.ns], pep_node_attr[:, -self.ns :]], dim=1
                )
                if self.num_conv_layers >= 3
                else pep_node_attr[:, : self.ns]
            )
            confidence = self.confidence_predictor(scatter_mean(scalar_pep_attr, data['pep'].batch, dim=0)).squeeze(dim=-1)
            return confidence
        
        # compute translational and rotational score vectors
        center_edge_index, center_edge_attr, center_edge_sh = self.build_center_conv_graph(data)
        center_edge_attr = self.center_edge_embedding(center_edge_attr)
        center_edge_attr = torch.cat([center_edge_attr, pep_node_attr[center_edge_index[1], :self.ns]], -1)
        global_pred = self.final_conv(pep_node_attr, center_edge_index, center_edge_attr, center_edge_sh, out_nodes=data.num_graphs)
        
        tr_pred = global_pred[:, :3] + global_pred[:, 6:9]
        rot_pred = global_pred[:, 3:6] + global_pred[:, 9:]
        data.graph_sigma_emb = self.timestep_emb_func(data.complex_t['tr'])
        
        # fix the magnitude of tr and rot score vectors
        tr_norm = torch.linalg.vector_norm(tr_pred, dim=1).unsqueeze(1)
        tr_pred = tr_pred / tr_norm * self.tr_final_layer(torch.cat([tr_norm, data.graph_sigma_emb], dim=1))
        
        rot_norm = torch.linalg.vector_norm(rot_pred, dim=1).unsqueeze(1)
        rot_pred = rot_pred / rot_norm * self.rot_final_layer(torch.cat([rot_norm, data.graph_sigma_emb], dim=1))
        if self.scale_by_sigma:
            _tr_sigma = tr_sigma
            if getattr(self, 'tr_sigma_floor', 0.0) > 0.0:
                _tr_sigma = tr_sigma.clamp(min=self.tr_sigma_floor)
            tr_pred = tr_pred / _tr_sigma.unsqueeze(1)
            rot_pred = rot_pred * score_norm(rot_sigma.cpu()).unsqueeze(1).to(data['pep'].x.device)

        # POSITION ADAPTER: additive zero-init translation correction (see __init__).
        # At init the adapter conv outputs exactly 0, so tr_pred is unchanged (reproduces
        # pretrained).  It learns an additive placement correction in the same (sigma-scaled)
        # score space as the frozen pretrained translation.
        if getattr(self, 'use_tr_adapter', False):
            adapter_pred = self.tr_adapter_conv(
                pep_node_attr, center_edge_index, center_edge_attr, center_edge_sh,
                out_nodes=data.num_graphs)
            tr_corr = adapter_pred[:, :3] + adapter_pred[:, 3:6]
            # BOUNDED correction (soft-clip): scale the MAGNITUDE (a rotation-invariant scalar,
            # so equivariance is preserved) so ||tr_corr|| < tr_adapter_cap, but stay IDENTITY
            # near 0 so the gradient FLOWS at init (a norm-normalised tanh has a zero Jacobian at
            # the origin and freezes the zero-init adapter forever — verified failure).  The
            # factor cap/sqrt(cap^2+||x||^2) -> 1 as ||x||->0 (gradient=I) and -> cap/||x|| as
            # ||x||->inf (||tr_corr||->cap).  Still exactly 0 at init (x=0 -> tr_corr=0).
            cap = getattr(self, 'tr_adapter_cap', 1.0)
            cnorm = torch.linalg.vector_norm(tr_corr, dim=1, keepdim=True)
            tr_corr = tr_corr * (cap / torch.sqrt(cap * cap + cnorm * cnorm))
            if self.scale_by_sigma:
                tr_corr = tr_corr / tr_sigma.unsqueeze(1)
                # LOW-SIGMA PHYSICAL CAP (see __init__ comment): the /tr_sigma division above
                # amplifies without bound as tr_sigma -> 0 (late, fine-refinement steps), where
                # the high-sigma gate below does nothing (gate=1, fully active) and there's no
                # remaining trajectory for the base's own signal to correct an overshoot back.
                # Same soft-clip shape as the score-space cap above, applied to the PHYSICAL
                # (post-scaling) magnitude this time so it bounds what actually reaches the
                # sampler regardless of how small sigma or how long/asymmetric the peptide is.
                _pcap = getattr(self, 'tr_adapter_physical_cap', None)
                if _pcap is not None:
                    # Aug 2026 length-scaled cap: a FLAT cap doesn't stop the correction from
                    # compounding across many low-sigma steps in the same direction -- verified
                    # empirically (full-150-complex sweep, physical_cap=4.0 fixed): outlier rate
                    # by peptide length was IDENTICAL before/after adding the flat cap (very_long
                    # 36.0% -> 37.1%, no change). The length-correlation itself (r=0.699) implies
                    # a SYSTEMATIC (not random-cancelling) bias that grows with peptide size, so
                    # the cap should shrink correspondingly, not stay constant. Exponential decay,
                    # anchored so cap(8 residues) == tr_adapter_physical_cap (backward compatible
                    # with the un-scaled default when tr_adapter_cap_length_scale=0):
                    #   cap(L) = physical_cap * exp(-r * (L - 8))
                    # e.g. r=0.15: cap(8)=4.0A (unchanged), cap(20)=0.66A (6x tighter),
                    # cap(40)=0.02A (near-fully suppressed) -- directly targets the peptides
                    # that were actually failing, leaves short/medium (already fine) unchanged.
                    _r = getattr(self, 'tr_adapter_cap_length_scale', 0.0)
                    if _r > 0.0 and hasattr(data['pep'], 'batch'):
                        _pep_len = torch.bincount(
                            data['pep'].batch, minlength=data.num_graphs
                        ).float().to(tr_corr.device)
                        _pcap_eff = _pcap * torch.exp(-_r * (_pep_len - 8.0)).unsqueeze(1)
                    else:
                        _pcap_eff = _pcap
                    _pnorm = torch.linalg.vector_norm(tr_corr, dim=1, keepdim=True)
                    tr_corr = tr_corr * (_pcap_eff / torch.sqrt(_pcap_eff * _pcap_eff + _pnorm * _pnorm))
                # LOW-SIGMA ZERO-OUT (see __init__ comment, cliff-v4 hypothesis): unlike the
                # physical cap above, this doesn't bound the magnitude -- it removes the
                # correction's ability to contribute at all once sigma drops below the
                # threshold, which is the only thing that can stop a sign-consistent drift from
                # compounding across steps regardless of how tightly each step is capped.
                _zero_below = getattr(self, 'tr_adapter_zero_below_sigma', 0.0)
                if _zero_below > 0.0:
                    tr_corr = tr_corr * (tr_sigma > _zero_below).float().unsqueeze(1)
            # SIGMA GATE (see __init__ comment): linearly taper tr_corr to zero between
            # tr_adapter_gate_hi and tr_sigma_max_for_gate. Applied AFTER the sigma-scaling
            # above so it gates the PHYSICAL-displacement-relevant quantity, not just the raw
            # pre-scale output. No-op (gate=1 everywhere) if tr_adapter_gate_hi >= sigma_max.
            _gate_hi = getattr(self, 'tr_adapter_gate_hi', None)
            _sigma_max_g = getattr(self, 'tr_sigma_max_for_gate', 30.0)
            if _gate_hi is not None and _gate_hi < _sigma_max_g:
                _gate = torch.where(
                    tr_sigma > _gate_hi,
                    torch.clamp(1.0 - (tr_sigma - _gate_hi) / (_sigma_max_g - _gate_hi), min=0.0),
                    torch.ones_like(tr_sigma),
                )
                tr_corr = tr_corr * _gate.unsqueeze(1)
            tr_pred = tr_pred + tr_corr

        # torsional components
        
        pep_a_node_attr_from_pep = torch.cat([pep_node_attr[data['pep'].batch == i][data['pep_a'].atom2res_index[data['pep_a'].batch == i]] for i in range(len(data.name))],dim=0)
        data['pep_a'].node_sigma_emb = self.timestep_emb_func(data['pep_a'].node_t['tr']) # tr rot and tor noise is all the same
        # DTYPE FIX (May 28 2026): atom2resid_index, atom2atomid_index, and pep_a.x
        # are stored as int64 in graph caches (torch.tensor(int_list) defaults to int64).
        # torch.cat([int64, float32]) auto-promotes to float32, but CUDA scatter ops
        # downstream can still see integer dtypes if the cache was written before this fix.
        # Explicit .float() on every int tensor before concat eliminates the issue entirely.
        pep_a_node_attr = torch.cat(
                [data['pep_a'].atom2resid_index.unsqueeze(-1).float(),
                 data['pep_a'].atom2atomid_index.unsqueeze(-1).float(),
                 data['pep_a'].x.float(),
                 data['pep_a'].node_sigma_emb,
                 pep_a_node_attr_from_pep], 1
            ) # 1+1+19+32+??84
        pep_a_node_attr = self.pep_a_node_embedding(pep_a_node_attr)
        
        
        if not data['pep_a'].mask_edges_backbone.squeeze().sum() == 0:
            tor_bonds_backbone, tor_edge_index_backbone, tor_edge_attr_backbone, tor_edge_sh_backbone = self.build_backbone_bond_conv_graph(data)
            tor_bond_vec_backbone = data['pep_a'].pos[tor_bonds_backbone[1]] - data['pep_a'].pos[tor_bonds_backbone[0]]
            tor_bond_attr_backbone = pep_a_node_attr[tor_bonds_backbone[0]] + pep_a_node_attr[tor_bonds_backbone[1]]
            tor_bonds_sh_backbone = o3.spherical_harmonics(
                "2e", tor_bond_vec_backbone, normalize=True, normalization="component"
            )
            tor_edge_sh_backbone = self.final_tp_tor_bb(tor_edge_sh_backbone, tor_bonds_sh_backbone[tor_edge_index_backbone[0]])
            tor_edge_attr_backbone = torch.cat(
                [
                    tor_edge_attr_backbone,
                    pep_a_node_attr[tor_edge_index_backbone[1], : self.ns],
                    tor_bond_attr_backbone[tor_edge_index_backbone[0], : self.ns],
                ],
                -1,
            )
            tor_pred_backbone = self.tor_bb_bond_conv(
                pep_a_node_attr,
                tor_edge_index_backbone,
                tor_edge_attr_backbone,
                tor_edge_sh_backbone,
                out_nodes=data['pep_a'].mask_edges_backbone.sum(),
                reduction="mean",
            )
            tor_pred_backbone = self.tor_bb_final_layer(tor_pred_backbone).squeeze(1)
            edge_sigma_backbone = tor_backbone_sigma[data["pep_a"].batch][
                data["pep_a", "pep_a"].edge_index[0]
            ][data['pep_a'].mask_edges_backbone.squeeze()]
            if self.scale_by_sigma:
                tor_pred_backbone = tor_pred_backbone * torch.sqrt(
                    torch.tensor(torus_score(edge_sigma_backbone.cpu().numpy()))
                    .float()
                    .to(data["pep_a"].x.device)
                )
        else: tor_pred_backbone = torch.empty(0, device=self.device)
        
        if not data['pep_a'].mask_edges_sidechain.squeeze().sum() == 0:
            tor_bonds_sidechain, tor_edge_index_sidechain, tor_edge_attr_sidechain, tor_edge_sh_sidechain = self.build_sidechain_bond_conv_graph(data)
            tor_bond_vec_sidechain = data['pep_a'].pos[tor_bonds_sidechain[1]] - data['pep_a'].pos[tor_bonds_sidechain[0]]
            tor_bond_attr_sidechain = pep_a_node_attr[tor_bonds_sidechain[0]] + pep_a_node_attr[tor_bonds_sidechain[1]]
            tor_bonds_sh_sidechain = o3.spherical_harmonics(
                "2e", tor_bond_vec_sidechain, normalize=True, normalization="component"
            )
            tor_edge_sh_sidechain = self.final_tp_tor_sc(tor_edge_sh_sidechain, tor_bonds_sh_sidechain[tor_edge_index_sidechain[0]])
            tor_edge_attr_sidechain = torch.cat(
                [
                    tor_edge_attr_sidechain,
                    pep_a_node_attr[tor_edge_index_sidechain[1], : self.ns],
                    tor_bond_attr_sidechain[tor_edge_index_sidechain[0], : self.ns],
                ],
                -1,
            )
            tor_pred_sidechain = self.tor_sc_bond_conv(
                pep_a_node_attr,
                tor_edge_index_sidechain,
                tor_edge_attr_sidechain,
                tor_edge_sh_sidechain,
                out_nodes=data['pep_a'].mask_edges_sidechain.sum(),
                reduction="mean",
            )
            tor_pred_sidechain = self.tor_sc_final_layer(tor_pred_sidechain).squeeze(1)
            edge_sigma_sidechain = tor_sidechain_sigma[data["pep_a"].batch][
                data["pep_a", "pep_a"].edge_index[0]
            ][data['pep_a'].mask_edges_sidechain.squeeze()]
            if self.scale_by_sigma:
                tor_pred_sidechain = tor_pred_sidechain * torch.sqrt(
                    torch.tensor(torus_score(edge_sigma_sidechain.cpu().numpy()))
                    .float()
                    .to(data["pep_a"].x.device)
                )
        else: tor_pred_sidechain = torch.empty(0, device=self.device)
        return tr_pred, rot_pred, tor_pred_backbone, tor_pred_sidechain
            
    def build_rec_conv_graph(self,data):
        data['receptor'].node_sigma_emb = self.timestep_emb_func(data['receptor'].node_t['tr']) # tr rot and tor noise is all the same
        if len(data['receptor'].x.shape) == 1:
            data['receptor'].x = data['receptor'].x[:,None]
        node_attr = torch.cat(
                [data['receptor'].x, data['receptor'].node_sigma_emb], 1
            ) # 1+5+4+1280+32
        # this assumes the edges were already created in preprocessing since protein's structure is fixed
        edge_index = data['receptor', 'receptor'].edge_index
        edge_vec = data['receptor', 'receptor'].edge_v.squeeze()
        edge_sigma_emb = data['receptor'].node_sigma_emb[edge_index[0]]
        edge_attr = torch.cat([edge_sigma_emb, data['receptor', 'receptor'].edge_s], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return node_attr, edge_index, edge_attr, edge_sh

    def build_pep_conv_graph(self,data):
        data['pep'].node_sigma_emb = self.timestep_emb_func(data['pep'].node_t['tr']) # tr rot and tor noise is all the same
        if len(data['pep'].x.shape) == 1:
            data['pep'].x = data['pep'].x[:,None]
        # DTYPE FIX (May 28 2026): pep.x may be int64 (seq_pep is int when no ESM embeddings).
        # .float() is a no-op for float32 and safely casts int64 → float32.
        node_attr = torch.cat(
                [data['pep'].x.float(), data['pep'].node_sigma_emb], 1
            ) # 1+5+4+32
        # this assumes the edges were already created in preprocessing since protein's structure is fixed
        edge_index = data['pep', 'pep'].edge_index
        edge_vec = data['pep', 'pep'].edge_v.squeeze()
        edge_sigma_emb = data['pep'].node_sigma_emb[edge_index[0]]
        edge_attr = torch.cat([edge_sigma_emb, data['pep', 'pep'].edge_s], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return node_attr, edge_index, edge_attr, edge_sh

    def build_cross_conv_graph(self, data, cross_distance_cutoff):
        # builds the cross edges between pep and receptor
        if torch.is_tensor(cross_distance_cutoff):

            # Convert to CPU then back to MPS/GPU to avoid Tensor core issues
            device = data['receptor'].pos.device

            x_pos = (data['receptor'].pos / cross_distance_cutoff[data['receptor'].batch]).cpu()
            y_pos = (data['pep'].pos / cross_distance_cutoff[data['pep'].batch]).cpu()

            # If-else statement to avoid tensor shape change issues
            if data['receptor'].tips.shape[0] == 0:
                x_tips = data['receptor'].pos.cpu()
            else:
                x_tips = (
                    data['receptor'].tips /
                    cross_distance_cutoff[data['receptor'].batch][:data['receptor'].tips.shape[0]]
                ).cpu()

            # If-else statement to avoid tensor shape change issues
            if data['pep'].tips.shape[0] == 0:
                y_tips = data['pep'].pos.cpu()
            else:
                y_tips = (
                    data['pep'].tips /
                    cross_distance_cutoff[data['pep'].batch][:data['pep'].tips.shape[0]]
                ).cpu()

            batch_x = data['receptor'].batch.cpu()
            batch_y = data['pep'].batch.cpu()

            # different cutoff for every graph (depends on the diffusion time)
            edge_index_ca_ca = radius(x_pos, y_pos, 1, batch_x, batch_y, max_num_neighbors=10000).to(device)
            edge_index_tips_ca = radius(x_tips, y_pos, 1, batch_x[:x_tips.shape[0]], batch_y, max_num_neighbors=10000).to(device)
            edge_index_ca_tips = radius(x_pos, y_tips, 1, batch_x, batch_y[:y_tips.shape[0]], max_num_neighbors=10000).to(device)
            edge_index_tips_tips = radius(x_tips, y_tips, 1, batch_x[:x_tips.shape[0]], batch_y[:y_tips.shape[0]], max_num_neighbors=10000).to(device)
        else:
            # Avoid tensor issues
            device = data['receptor'].pos.device

            rec_pos = data['receptor'].pos.cpu()
            if data['receptor'].tips.shape[0] == 0:
                rec_tips = rec_pos
            else:
                rec_tips = data['receptor'].tips.cpu()

            pep_pos = data['pep'].pos.cpu()

            if data['pep'].tips.shape[0] == 0:
                pep_tips = pep_pos
            else:
                pep_tips = data['pep'].tips.cpu()

            batch_rec = data['receptor'].batch.cpu()
            batch_pep = data['pep'].batch.cpu()
            
            edge_index_ca_ca = radius(rec_pos, pep_pos, 1, batch_rec, batch_pep, max_num_neighbors=10000).to(device)
            edge_index_tips_ca = radius(rec_tips, pep_pos, 1, batch_rec[:rec_tips.shape[0]], batch_pep, max_num_neighbors=10000).to(device)
            edge_index_ca_tips = radius(rec_pos, pep_tips, 1, batch_rec, batch_pep[:pep_tips.shape[0]], max_num_neighbors=10000).to(device)
            edge_index_tips_tips = radius(rec_tips, pep_tips, 1, batch_rec[:rec_tips.shape[0]], batch_pep[:pep_tips.shape[0]], max_num_neighbors=10000).to(device)
        
        edge_index = torch.cat((edge_index_ca_ca, edge_index_tips_ca, edge_index_ca_tips, edge_index_tips_tips), dim=1)
        edge_index = torch.unique(edge_index,dim=1).reshape(2,-1)
        
        src, dst = edge_index

        # Better support for CPU tensors and MPS tensors by performing calculations on CPU and then moving back to the original device
        edge_vec_ca_ca = data['receptor'].pos[dst.long()] - data['pep'].pos[src.long()]

        if data['receptor'].tips.shape[0] == 0:
            rec_tips = data['receptor'].pos
        else:
            rec_tips = data['receptor'].tips

        if data['pep'].tips.shape[0] == 0:
            pep_tips = data['pep'].pos
        else:
            pep_tips = data['pep'].tips

        edge_vec_tips_ca = rec_tips[dst.long()] - data['pep'].pos[src.long()]
        edge_vec_ca_tips = data['receptor'].pos[dst.long()] - pep_tips[src.long()]
        edge_vec_tips_tips = rec_tips[dst.long()] - pep_tips[src.long()]
        
        edge_vec_min, indices = torch.cat((edge_vec_ca_ca.norm(dim=-1)[:,None],edge_vec_tips_ca.norm(dim=-1)[:,None],edge_vec_ca_tips.norm(dim=-1)[:,None],edge_vec_tips_tips.norm(dim=-1)[:,None]),dim=1).min(dim=1)

        edge_length_emb = self.cross_distance_expansion(edge_vec_min)
        # cross_edge_type_emb = self.cross_edge_type(indices)
        edge_sigma_emb = data['pep'].node_sigma_emb[src.long()]
        # edge_attr = torch.cat([cross_edge_type_emb, edge_sigma_emb, edge_length_emb], 1)
        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], 1)
        edge_vec_final = torch.where(
            (indices == 0).unsqueeze(-1),
            edge_vec_ca_ca,
            torch.where(
                (indices == 1).unsqueeze(-1),
                edge_vec_tips_ca,
                torch.where(
                    (indices == 2).unsqueeze(-1),
                    edge_vec_ca_tips,
                    edge_vec_tips_tips
                )
            )
        )

        edge_sh = o3.spherical_harmonics(
            self.sh_irreps,
            edge_vec_final,
            normalize=True,
            normalization='component'
        )

        return edge_index, edge_attr, edge_sh, indices

    def build_center_conv_graph(self, data):
        edge_index = torch.cat([data['pep'].batch.unsqueeze(0), torch.arange(len(data['pep'].batch)).to(data['pep'].x.device).unsqueeze(0)], dim=0)
        center_pos, count = torch.zeros((data.num_graphs, 3)).to(data['pep'].x.device), torch.zeros((data.num_graphs, 3)).to(data['pep'].x.device)
        center_pos.index_add_(0, index=data['pep'].batch, source=data['pep'].pos)
        center_pos = center_pos / torch.bincount(data['pep'].batch).unsqueeze(1)

        edge_vec = data['pep'].pos[edge_index[1]] - center_pos[edge_index[0]]
        edge_attr = self.center_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data['pep'].node_sigma_emb[edge_index[1].long()]
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return edge_index, edge_attr, edge_sh
    
    def build_backbone_bond_conv_graph(self, data):
        # builds the graph for the convolution between the center of the rotatable bonds and the neighbouring nodes
        bonds = data['pep_a', 'pep_a'].edge_index[:, data['pep_a'].mask_edges_backbone.squeeze()].long()
        bond_pos = (data['pep_a'].pos[bonds[0]] + data['pep_a'].pos[bonds[1]]) / 2
        bond_batch = data['pep_a'].batch[bonds[0]]
        edge_index = radius((data['pep_a'].pos).cpu(), (bond_pos).cpu(), self.lig_max_radius, batch_x=(data['pep_a'].batch).cpu(), batch_y=(bond_batch).cpu()).to(data['pep_a'].pos.device)

        # Conversion back to proper device in order to avoid tensor core issues on MPS
        edge_index = edge_index.long()

        edge_vec = (
            data['pep_a'].pos[edge_index[1]].to(data['pep_a'].pos.device)
            - bond_pos[edge_index[0]].to(data['pep_a'].pos.device)
        )
        edge_attr = self.lig_distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = self.final_edge_embedding(edge_attr)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return bonds, edge_index, edge_attr, edge_sh
    
    def build_sidechain_bond_conv_graph(self, data):
        # builds the graph for the convolution between the center of the rotatable bonds and the neighbouring nodes
        bonds = data['pep_a', 'pep_a'].edge_index[:, data['pep_a'].mask_edges_sidechain.squeeze()].long()
        bond_pos = (data['pep_a'].pos[bonds[0]] + data['pep_a'].pos[bonds[1]]) / 2
        bond_batch = data['pep_a'].batch[bonds[0]]
        edge_index = radius((data['pep_a'].pos).cpu(), (bond_pos).cpu(), self.lig_max_radius, batch_x=(data['pep_a'].batch).cpu(), batch_y=(bond_batch).cpu()).to(data['pep_a'].pos.device)

        # Conversion back to proper device in order to avoid tensor core issues on MPS
        edge_index = edge_index.long()

        edge_vec = (
            data['pep_a'].pos[edge_index[1]].to(data['pep_a'].pos.device)
            - bond_pos[edge_index[0]].to(data['pep_a'].pos.device)
        )
        edge_attr = self.lig_distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = self.final_edge_embedding(edge_attr)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return bonds, edge_index, edge_attr, edge_sh
