##########################################################################
# File Name: inference.py
# Author: huifeng
# mail: huifengzhao@zju.edu.cn
# Created Time: Thu 19 Oct 2023 09:31:15 AM CST
#########################################################################


import os
import sys

# ── macOS env-var guards ────────────────────────────────────────────────────
# Must be set BEFORE torch is imported so the flags are visible when
# PyTorch/OpenMP shared libraries are loaded.
#
# KMP_DUPLICATE_LIB_OK  – silences the crash caused by two copies of
#   libiomp5 being loaded (common when conda PyTorch meets a system OpenMP).
#
# PYTORCH_ENABLE_MPS_FALLBACK – lets ops not yet ported to Metal (e.g.
#   torch.unique with a dim argument, torch_scatter) fall back to CPU
#   automatically instead of raising a NotImplementedError.
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# ────────────────────────────────────────────────────────────────────────────

import copy
import yaml
import torch
import MDAnalysis
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
from io import StringIO
from argparse import Namespace
from MDAnalysis.coordinates.memory import MemoryReader
from torch_geometric.loader import DataListLoader
from utils.inference_parsing import get_parser
from utils.utils import get_model, ExponentialMovingAverage
from utils.inference_utils import InferenceDataset, set_nones
from utils.peptide_updater import randomize_position
from utils.sampling import sampling
import multiprocessing

warnings.filterwarnings("ignore")


def load_model(score_model_args, ckpt_path, device):
    state_dict = torch.load(
        ckpt_path, map_location=torch.device("cpu")
    )
    # BUG FIX (Aug 2026): a fine-tuned ADAPTER checkpoint carries encoder.tr_adapter_conv.*
    # weights, but the model is built from model_parameters.yml which has no use_tr_adapter key
    # (defaults False) — so the adapter layer was never created and its weights were SILENTLY
    # DROPPED by the strict=False load below. Result: adapter checkpoints ran as the frozen
    # base (== pretrained) at inference, making every adapter eval meaningless. Fix: detect the
    # adapter weights in the checkpoint and enable use_tr_adapter (+ cap) BEFORE building the
    # model, so the layer exists and the trained adapter weights actually load and take effect.
    _sd = state_dict["model"] if isinstance(state_dict, dict) and "model" in state_dict else state_dict
    if any("tr_adapter" in k for k in _sd):
        score_model_args.use_tr_adapter = True
        # BUG FIX (Aug 2026 cliff-run postmortem): this used to hardcode cap=1.0 regardless of
        # what --tr-adapter-cap the checkpoint was actually TRAINED with. A checkpoint trained
        # under a tighter cap (to bound worst-case correction magnitude, since the adapter's
        # physical correction scales with tr_sigma up to ~tr_sigma_max*cap) would silently run
        # at inference under the WIDER default cap instead — defeating the tightened cap's whole
        # point right where it matters (rare high-sigma spikes). Checkpoints saved after this fix
        # store their own cap (save_checkpoint() in train_lastlayer.py); only fall back to the
        # 1.0 default for older checkpoints that predate this field.
        if isinstance(state_dict, dict) and "tr_adapter_cap" in state_dict:
            score_model_args.tr_adapter_cap = float(state_dict["tr_adapter_cap"])
        elif not hasattr(score_model_args, "tr_adapter_cap"):
            score_model_args.tr_adapter_cap = 1.0
        # Aug 2026 (widened-adapter fix): hidden_mult changes the ARCHITECTURE (fc.0.weight
        # shape), so it can't be a stored config value that might drift out of sync with the
        # actual weights -- infer it directly from the checkpoint's own fc.0.weight shape,
        # which is authoritative. hidden_features = 2*ns*mult (see models/diffusion.py).
        _fc0_key = next((k for k in _sd if k.endswith("tr_adapter_conv.fc.0.weight")), None)
        if _fc0_key is not None:
            _hidden_features = _sd[_fc0_key].shape[0]
            score_model_args.tr_adapter_hidden_mult = _hidden_features / (2 * score_model_args.ns)
        # Sigma-gate (outlier-pose fix): same self-describing rationale as tr_adapter_cap.
        if isinstance(state_dict, dict) and "tr_adapter_gate_hi" in state_dict:
            score_model_args.tr_adapter_gate_hi = float(state_dict["tr_adapter_gate_hi"])
        # Low-sigma physical cap (Aug 2026 cliffmixed outlier forensics): same self-describing
        # rationale. Older checkpoints without this field fall back to the models/diffusion.py
        # __init__ default (4.0 A) via getattr there, not here.
        if isinstance(state_dict, dict) and "tr_adapter_physical_cap" in state_dict:
            score_model_args.tr_adapter_physical_cap = float(state_dict["tr_adapter_physical_cap"])
        if isinstance(state_dict, dict) and "tr_adapter_cap_length_scale" in state_dict:
            score_model_args.tr_adapter_cap_length_scale = float(state_dict["tr_adapter_cap_length_scale"])
        # Debug/experimentation override -- lets an already-trained checkpoint (predating this
        # field) be evaluated under a length-scaled cap WITHOUT retraining, to validate the idea
        # cheaply before committing GPU time to a real training run with it baked in.
        if os.environ.get("TR_ADAPTER_CAP_LENGTH_SCALE") is not None:
            score_model_args.tr_adapter_cap_length_scale = float(os.environ["TR_ADAPTER_CAP_LENGTH_SCALE"])
        if isinstance(state_dict, dict) and "tr_adapter_zero_below_sigma" in state_dict:
            score_model_args.tr_adapter_zero_below_sigma = float(state_dict["tr_adapter_zero_below_sigma"])
        # Same cheap-validation override as above, for the cliff-v4 zero-below-sigma hypothesis.
        if os.environ.get("TR_ADAPTER_ZERO_BELOW_SIGMA") is not None:
            score_model_args.tr_adapter_zero_below_sigma = float(os.environ["TR_ADAPTER_ZERO_BELOW_SIGMA"])
        print(f"[load_model] adapter checkpoint detected -> use_tr_adapter=True "
              f"(cap={getattr(score_model_args, 'tr_adapter_cap', 1.0)}"
              f"{' [from checkpoint]' if isinstance(state_dict, dict) and 'tr_adapter_cap' in state_dict else ' [legacy default]'}, "
              f"hidden_mult={getattr(score_model_args, 'tr_adapter_hidden_mult', 1.0):.2f} [from checkpoint shape], "
              f"gate_hi={getattr(score_model_args, 'tr_adapter_gate_hi', 'unset (no gate)')}, "
              f"physical_cap={getattr(score_model_args, 'tr_adapter_physical_cap', 4.0)}, "
              f"zero_below_sigma={getattr(score_model_args, 'tr_adapter_zero_below_sigma', 0.0)})")
    model = get_model(score_model_args, no_parallel=True)
    # strict=False: tolerates new params added to the architecture (e.g. cross_type_embedding)
    # that are absent in older checkpoints — they keep their zero-init from __init__.
    missing, unexpected = model.load_state_dict(state_dict["model"], strict=False)
    _adapter_dropped = [k for k in unexpected if "tr_adapter" in k]
    if _adapter_dropped:
        raise RuntimeError(f"adapter weights STILL dropped at inference ({len(_adapter_dropped)}) "
                           f"— use_tr_adapter not wired; refusing to run a silent no-op adapter.")
    if unexpected:
        print(f"[load_model] unexpected keys ignored: {len(unexpected)}")
    model = model.to(device)

    ema_weights = ExponentialMovingAverage(
        model.parameters(), decay=score_model_args.ema_rate
    )
    ema_weights.load_state_dict(state_dict["ema_weights"], device=device)
    ema_weights.copy_to(model.parameters())
    return model

def load_config(args):
    if args.config:
        # content in config file will cover the cmd input content
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in config_dict.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value

def prepare_data(args, score_model_args):
    if args.protein_peptide_csv is not None:
        df = pd.read_csv(args.protein_peptide_csv)
        complex_name_list = set_nones(df["complex_name"].tolist())
        protein_description_list = set_nones(df["protein_description"].tolist())
        peptide_description_list = set_nones(df["peptide_description"].tolist())
    else:
        complex_name_list = [args.complex_name]
        protein_description_list = [args.protein_description]
        peptide_description_list = [args.peptide_description]
    
    complex_name_list = [
        name if name is not None else f"complex_{i}"
        for i, name in enumerate(complex_name_list)
    ]
    for name in complex_name_list:
        write_dir = f"{args.output_dir}/{name}"
        os.makedirs(write_dir, exist_ok=True)
    
    # preprocessing of initial proteins and peptides into geometric graphs
    return InferenceDataset(
        output_dir=args.output_dir,
        complex_name_list=complex_name_list,
        protein_description_list=protein_description_list,
        peptide_description_list=peptide_description_list,
        lm_embeddings=score_model_args.esm_embeddings_path_train is not None,
        lm_embeddings_pep=score_model_args.esm_embeddings_peptide_train is not None,
        conformation_type=args.conformation_type,
        conformation_partial=args.conformation_partial,
    )

def prepare_data_list(original_complex_graph, N):
    data_list = []
    nums = []
    if len(original_complex_graph["peptide_inits"]) == 1:
        data_list = [copy.deepcopy(original_complex_graph) for _ in range(N)]
    elif len(original_complex_graph["peptide_inits"]) > 1:
         for i, peptide_init in enumerate(
                    original_complex_graph["peptide_inits"]
                ):
            if i !=0:
                original_complex_graph["pep_a"].pos = (
                torch.from_numpy(
                    MDAnalysis.Universe(peptide_init).atoms.positions
                )
                - original_complex_graph.original_center
            )
            num = N - sum(nums) if i == len(original_complex_graph["peptide_inits"]) - 1 else round(
                original_complex_graph["partials"][i] / sum(original_complex_graph["partials"]) * N
            )
            nums.append(num)
            data_list.extend([copy.deepcopy(original_complex_graph) for _ in range(num)])
    return data_list

def save_predictions(write_dir, predict_pos, original_complex_graph, args, confidence):
    raw_pdb = MDAnalysis.Universe(StringIO(original_complex_graph["pep"].noh_mda), format="pdb")
    peptide_unrelaxed_files = []
    
    re_order = None
    # reorder predictions based on confidence output
    if confidence is not None:
        confidence = confidence.cpu().numpy()
        re_order = np.argsort(confidence)[::-1]
        confidence = confidence[re_order]
        predict_pos = predict_pos[re_order]

    for rank, pos in enumerate(predict_pos):
        raw_pdb.atoms.positions = pos
        file_name = f"rank{rank+1}_{args.scoring_function}.pdb" if confidence is not None else f"rank{rank+1}.pdb"
        peptide_unrelaxed_file = os.path.join(write_dir, file_name)
        peptide_unrelaxed_files.append(peptide_unrelaxed_file)
        raw_pdb.atoms.write(peptide_unrelaxed_file)

    if args.scoring_function == "ref2015" or args.fastrelax:
        from utils.pyrosetta_utils import relax_score, PYROSETTA_AVAILABLE
        if not PYROSETTA_AVAILABLE:
            print(
                "WARNING: PyRosetta is not available on this system "
                "(requires ≥32 GB RAM and a PyRosetta installation). "
                "Skipping ref2015 scoring — unrelaxed poses saved as rank*.pdb."
            )
        else:
            relaxed_poses = [peptide.replace(".pdb", "_relaxed.pdb") for peptide in peptide_unrelaxed_files]
            protein_raw_file = f"{write_dir}/{os.path.basename(write_dir)}_protein_raw.pdb"

            with multiprocessing.Pool(args.cpu) as pool:
                ref2015_scores = pool.map(
                    relax_score,
                    zip(
                        [protein_raw_file] * len(peptide_unrelaxed_files),
                        peptide_unrelaxed_files,
                        relaxed_poses,
                        [args.scoring_function == "ref2015"] * len(peptide_unrelaxed_files),
                    ),
                )
            if ref2015_scores and ref2015_scores[0] is not None:
                re_order = np.argsort(ref2015_scores)
                score_results = [['file','ref2015score']]
                for rank, order in enumerate(re_order):
                    os.rename(relaxed_poses[order], os.path.join(write_dir, f"rank{rank+1}_{args.scoring_function}.pdb"))
                    score_results.append([f"rank{rank+1}_{args.scoring_function}", f"{ref2015_scores[order]:.2f}"])
                open(os.path.join(write_dir, "ref2015_score.csv"),'w').write('\n'.join([','.join(i) for i in score_results]))
    
    if re_order is not None:
        return re_order
    else:
        return 0
    
def process_complex(model, confidence_model, score_model_args, args, original_complex_graph, write_dir):
    # data_list_prepare
    N = args.N
    data_list = prepare_data_list(original_complex_graph, N)
    randomize_position(data_list, False, score_model_args.tr_sigma_max)

    visualization_list = None
    if args.save_visualisation:
        visualization_list = [
            np.asarray([g["pep_a"].pos.cpu().numpy() + original_complex_graph.original_center.cpu().numpy() for g in data_list])
        ]

    data_list, confidence, visualization_list = sampling(
        data_list=data_list,
        model=model,
        args=score_model_args,
        batch_size=args.batch_size,
        no_final_step_noise=args.no_final_step_noise,
        inference_steps=args.inference_steps,
        actual_steps=(
            args.actual_steps
            if args.actual_steps is not None
            else args.inference_steps
        ),
        visualization_list=visualization_list,
        confidence_model=confidence_model,
    )

    predict_pos = np.asarray(
        [
            complex_graph["pep_a"].pos.cpu().numpy()
            + original_complex_graph.original_center.cpu().numpy()
            for complex_graph in data_list
        ]
    )

    # save predictions
    re_order = save_predictions(write_dir, predict_pos, original_complex_graph, args, confidence)

    # save visualisation frames
    if args.save_visualisation:
        raw_pdb = MDAnalysis.Universe(
            StringIO(original_complex_graph["pep"].noh_mda), format="pdb"
        )
        visualization_list = list(
            np.transpose(np.array(visualization_list), (1, 0, 2, 3))
        )
        if args.scoring_function in ["confidence", "ref2015"]:
            for rank, batch_idx in enumerate(re_order):
                raw_pdb.load_new(
                    visualization_list[batch_idx], format=MemoryReader
                )
                with MDAnalysis.Writer(
                    os.path.join(write_dir, f"rank{rank+1}_reverseprocess.pdb"),
                    multiframe=True,
                    bonds=None,
                    n_atoms=raw_pdb.atoms.n_atoms,
                ) as pdb_writer:
                    for ts in raw_pdb.trajectory:
                        pdb_writer.write(raw_pdb)
        else:
            for rank in range(len(predict_pos)):
                raw_pdb.load_new(visualization_list[rank], format=MemoryReader)
                with MDAnalysis.Writer(
                    os.path.join(write_dir, f"rank{rank+1}_reverseprocess.pdb"),
                    multiframe=True,
                    bonds=None,
                    n_atoms=raw_pdb.atoms.n_atoms,
                ) as pdb_writer:
                    for ts in raw_pdb.trajectory:
                        pdb_writer.write(raw_pdb)

def main(args):
    # Input parameters by config file
    load_config(args)
    os.makedirs(args.output_dir, exist_ok=True)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif hasattr(torch.backends, "xpu") and torch.backends.xpu.is_available():
        device = torch.device("xpu")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    with open(f"{args.model_dir}/model_parameters.yml") as f:
        score_model_args = Namespace(**yaml.full_load(f))

    inference_dataset = prepare_data(args, score_model_args)
    inference_loader = DataListLoader(
        dataset=inference_dataset, batch_size=1, shuffle=False
    )

    model = load_model(score_model_args, f"{args.model_dir}/{args.ckpt}", device)

    # load confidence model
    confidence_model = None
    confidence_args = None
    if args.scoring_function == "confidence":
        with open(f"{args.confidence_model_dir}/model_parameters.yml") as f:
            confidence_args = Namespace(**yaml.full_load(f))

        confidence_model = get_model(
            confidence_args, no_parallel=True, confidence_mode=True
        )
        state_dict = torch.load(
            f"{args.confidence_model_dir}/{args.confidence_ckpt}",
            map_location=torch.device("cpu"),
        )
        confidence_model.load_state_dict(state_dict["model"], strict=True)
        confidence_model = confidence_model.to(device)
        confidence_model.eval()

    failures, skipped = 0, 0
    print("Size of test dataset: ", len(inference_dataset))

    for idx, original_complex_graph in tqdm(enumerate(inference_loader)):
        if not original_complex_graph[0].success:
            skipped += 1
            print(
                f"HAPPENING | The inference dataset did not contain {inference_dataset.complex_names[idx]} for {inference_dataset.peptide_descriptions[idx]} and {inference_dataset.protein_descriptions[idx]}. We are skipping this complex."
            )
            continue
        try:
            process_complex(
                model, confidence_model, score_model_args, args, original_complex_graph[0],
                f"{args.output_dir}/{inference_dataset.complex_names[idx]}"
            )
        except Exception as e:
            import traceback
            print("\nFULL ERROR TRACE:")
            traceback.print_exc()

    print(f"Failed for {failures} complexes")
    print(f"Skipped {skipped} complexes")
    print(f"Results are in {args.output_dir}")


if __name__ == "__main__":
    _args = get_parser().parse_args()
    main(_args)
