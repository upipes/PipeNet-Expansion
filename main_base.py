
import argparse
import os
import random
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import itertools
import utility.data_preparation as data_preparation
import dual_align
import datetime
import hashlib
from rich import print
import pandas as pd

from utility.train_base_base import BASECLASSIFIER
import baselines.conse as conse
import baselines.vgse as vgse
import baselines.costa as costa
import baselines.sourceonly as sonly
import baselines.dann as dann
import baselines.adda as adda
import baselines.tpds as tpds
import baselines.g2kd as g2kd

from datetime import datetime
import json
from pathlib import Path
import time

import subprocess
import sys
from pathlib import Path


parser = argparse.ArgumentParser()

# base settings
parser.add_argument('--cuda', action='store_true', default=False, help='Enables cuda')
parser.add_argument('--dataroot', default='***', help='Path to datasets folder')
parser.add_argument('--rootpath', default='***', help='Path for saving model checkpoints and results')
parser.add_argument('--numSeeds', type=int, default=1, help='Number of (randomly selected) seeds to experiment on')
parser.add_argument('--manualSeed', nargs='+', type=int, default=None, help='Specify manual seed(s)')
parser.add_argument('--dataset', default='SD', choices=['SD', 'Road', 'domain1', 'domain2', 'domain3', 'domain4'],help='Dataset for GPR classification')
parser.add_argument('--zst', action='store_true', default=False, help='Perform experiment of model transfer from one dataset to another')
parser.add_argument('--zstfrom', default='Road', help='Transfer from which dataset (GPR-SD)')


# dataloader, process and save args
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--preprocessing', action='store_true', default=False, help='Enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--strict_eval', action='store_true', default=False, help='When running on test set, only validate after final epoch')
parser.add_argument('--early_stopping_slope', action='store_true', default=False, help='Enable early stopping heuristic')
parser.add_argument('--norm_scale_heuristic', action='store_true', default=False, help='Scale the predicted classifier weights (heuristic for bias correction)')
parser.add_argument('--calc_entropy', action='store_true', default=False, help='Calculate output distribution on test set of original and new classes')
parser.add_argument('--save_path', default='result.csv', help='Path to save the results.')
parser.add_argument('--avg_save_path', default='result_avg.csv', help='Path to save the averaged results.')
parser.add_argument('--accuracy_only_csv', action='store_true', default=False, help='Save only the accuracy fields needed by compact experiments.')
parser.add_argument('--train_sample_missing_rate', nargs='+', type=int, default=[0], help='Percentage of source-domain training samples removed uniformly per original class.')
parser.add_argument('--save_domain_classifier_weights', action='store_true', default=False, help='Save source-domain and target-domain classifier weights for each run.')
parser.add_argument('--overwrite_base_classifier_cache', action='store_true', default=False, help='Retrain and overwrite cached source-domain base classifiers.')
parser.add_argument('--cache_tag', default='', help='Optional tag appended to base-classifier cache names for isolated experiments.')

# Baselines
parser.add_argument('--conse_benchmark', action='store_true', default=False, help='Run ConSE benchmark')
parser.add_argument('--costa_benchmark', action='store_true', default=False, help='Run COSTA benchmark')
parser.add_argument('--subspace_proj', action='store_true', default=False, help='Adapted baseline from Akyürek et al. Project predicted weights unto subspace spanned by original class weights')
parser.add_argument('--source_only_benchmark', action='store_true', default=False, help='Run SourceOnly benchmark')
parser.add_argument('--dann_benchmark', action='store_true', default=False, help='Run DANN benchmark')
parser.add_argument('--adda_benchmark', action='store_true', default=False, help='Run ADDA benchmark')
parser.add_argument('--tpds_benchmark', action='store_true', default=False, help='Run TPDS benchmark')
parser.add_argument('--g2kd_benchmark', action='store_true', default=False, help='Run G2KD benchmark')
parser.add_argument('--vgse_baseline', default=None, help='Run VGSE CRM baseline (choices: wavg or smo)')
parser.add_argument('--vgse_nbs', default=5, help='Number of VGSE CRM WAvg neighbours')
parser.add_argument('--vgse_eta', default=5, help='eta hyperparameter for VGSE CRM WAvg')
parser.add_argument('--vgse_alpha', type=float, default=1, help='alpha hyperparameter for VGSE CRM SMO')
parser.add_argument('--daegnn', action='store_true', default=False, help='Run wDAE-GNN benchmark')

# Training args
parser.add_argument('--num_layers', nargs='+', type=int, default=[2], help='Number of layers in weight prediction MLP')
parser.add_argument('--embed_dim', nargs='+', type=int, default=[1000], help='Set the dimensionality of the hidden layers')
parser.add_argument('--batch_size', nargs='+', type=int, default=[16], help='input batch size')
parser.add_argument('--nepoch', nargs='+', type=int, default=[500], help='Max number of epochs to train for')
parser.add_argument('--classifier_nepoch', type=int, default=100, help='Max number of epochs to train for')
parser.add_argument('--classifier_lr', type=float, default=0.0001, help='Learning rate to train softmax classifier')
parser.add_argument('--classifier_beta1', type=float, default=0.9, help='beta1 for adam to train classifier. default=0.5')
parser.add_argument('--lr', nargs='+', type=float, default=[0.0001], help='Learning rate(s) to train weight regressor network')
parser.add_argument('--beta1', nargs='+', type=float, default=[0.9], help='beta1 parameter(s) for adam to train weight regressor network. default=0.5')

parser.add_argument('--dann_hidden_dim', type=int, default=512, help='Hidden dimension of DANN feature extractor')
parser.add_argument('--dann_domain_weight', type=float, default=0.1, help='Weight for DANN domain loss')

parser.add_argument('--adda_hidden_dim', type=int, default=512, help='Hidden dimension of ADDA feature extractor')
parser.add_argument('--adda_adv_weight', type=float, default=0.1, help='Weight for ADDA adversarial loss')
parser.add_argument('--adda_pretrain_epochs', type=int, default=50, help='Number of source pretraining epochs for ADDA')

parser.add_argument('--tpds_align_weight', type=float, default=1.0, help='Weight for TPDS pairwise alignment')
parser.add_argument('--tpds_cc_weight', type=float, default=0.5, help='Weight for TPDS category consistency')
parser.add_argument('--tpds_im_weight', type=float, default=0.1, help='Weight for TPDS mutual information regularization')
parser.add_argument('--tpds_neighbors', type=int, default=5, help='Number of neighbors used by TPDS pair matching')
parser.add_argument('--tpds_steps', type=int, default=3, help='Number of TPDS proxy-distribution refinement steps')

parser.add_argument('--g2kd_dis_weight', type=float, default=1.0, help='Weight for G2KD distillation loss')
parser.add_argument('--g2kd_stu_weight', type=float, default=0.3, help='Weight for G2KD student pseudo-label loss')
parser.add_argument('--g2kd_ent_weight', type=float, default=0.1, help='Weight for G2KD entropy regularization')
parser.add_argument('--g2kd_neighbors', type=int, default=5, help='Number of neighbors used to build geometry-guided knowledge')


# Method args
parser.add_argument('--image_embedding', default='pretrained_resnet50', help='Whether base classifier was finetuned on original classes or generic imagenet model')
parser.add_argument('--conclude_inv', action='store_true', default=False, help='Conclude intervention classes or not')
parser.add_argument('--factual_branch', default='attention', choices=['attention', 'mean', 'none'], help='Factual refinement branch')
parser.add_argument('--intervention_branch', default='attention', choices=['attention', 'mean', 'none'], help='Intervention refinement branch')
parser.add_argument('--sep_param', default=-0.5, type=float, help='Separation loss threshold')
parser.add_argument('--concatenation', action='store_true', default=False, help='Dual description concatenation')
parser.add_argument('--att_dim', default=0, type=int, help='att dim')

# Semantic quantity and quality args
parser.add_argument('--view_num', default=50, type=int, help='Number of views in LLM descriptions')
parser.add_argument('--view_percent', default=100, type=int, help='Percentage of the original non-global views used in the experiment.')
parser.add_argument('--random_view_subset', action='store_true', default=False, help='Randomly sample view_num non-global views while always keeping the global view.')
parser.add_argument('--view_error_percent', default=0, type=int, help='Percentage of non-global views corrupted by class permutation replacement.')
parser.add_argument('--class_permute_error_views', action='store_true', default=False, help='Apply deranged class-swap replacement to selected erroneous views.')

# Applicability args
parser.add_argument('--class_embedding', default='clip', choices=['clip', 'sbert', 'llama', 'qwen'], help='Text-to-embedding representation generation')
parser.add_argument('--llm', default='gpt4o', choices=['gpt4o', 'gpt4omini', 'gemini2.5', 'llama70b', 'qwen_plus'], help='LLMs to use')

# Ablation args
parser.add_argument('--single_autoencoder_baseline', action='store_true', default=False, help='Train a single autoencoder predicting weights from attributes')
parser.add_argument('--cos_sim_loss', action='store_true', default=False, help='Enable cosine similarity loss')
parser.add_argument('--single_modal_ablation', action='store_true', default=False, help='Remove Weight to Attribute mapping')
parser.add_argument('--include_new', action='store_true', default=False, help='Whether to include new attributes during training')
parser.add_argument('--seperate_loss', action='store_true', default=False, help='Use seperate loss or not')
parser.add_argument('--inv_merge', action='store_true', default=False, help='Use inv_merge or not')

# Empirical args
parser.add_argument('--method', default='ours_with_inv', choices=['ICIS', 'ours_without_inv', 'ours_with_inv'], help='Conduct (further) empirical analysis on three methods: the SOTA baseline (ICIS), our method without interventions, and our method with interventions.')

# Confusion boundary args
parser.add_argument('--save_pred_matrix', action='store_true', default=False, help='Save matrices with predictions after evaluation for confusion boundary analysis')
parser.add_argument('--pred_matrix_output_dir', default='', help='Optional directory for saved per-class accuracies and prediction matrices.')
parser.add_argument('--save_method_checkpoint', action='store_true', default=False, help='Save adapted baseline modules for downstream visualization such as Grad-CAM')


start_time = datetime.now()
start_timestamp = time.time()

opt = parser.parse_args()

# PROJECT_ROOT = Path(__file__).resolve().parent
#
# def run_feature_extraction():
#     feature_scripts = [
#         PROJECT_ROOT / "feature_extraction_50.py",
#         PROJECT_ROOT / "feature_extraction_50_simulation.py",
#         PROJECT_ROOT / "feature_extraction_101.py",
#     ]
#
#     for script_path in feature_scripts:
#         if not script_path.exists():
#             raise FileNotFoundError(
#                 f"Feature extraction script not found: {script_path}"
#             )
#
#         print(f"Running feature extraction: {script_path.name}")
#
#         completed = subprocess.run(
#             [sys.executable, str(script_path)],
#             cwd=str(PROJECT_ROOT),
#             env=os.environ.copy(),
#         )
#
#         if completed.returncode != 0:
#             raise RuntimeError(
#                 f"Feature extraction failed: {script_path.name}"
#             )
#
#
# run_feature_extraction()

def classifier_cache_path(opt, seed):
    source_dataset = opt.zstfrom if opt.zst else opt.dataset
    missing_rate = getattr(opt, "current_train_sample_missing_rate", opt.train_sample_missing_rate[0])
    suffix = f"_miss{missing_rate}" if missing_rate else ""
    tag = f"_{opt.cache_tag}" if getattr(opt, "cache_tag", "") else ""
    return f'{opt.rootpath}/models/base-classifiers/{source_dataset}_{opt.image_embedding}_seed{seed}_clr{opt.classifier_lr}_nep{opt.classifier_nepoch}{suffix}{tag}'


def _linear_state_from_weight_bias(weight, bias, classes, note=None):
    return {
        "weight": weight.detach().cpu().clone(),
        "bias": bias.detach().cpu().clone(),
        "classes": classes.detach().cpu().clone() if torch.is_tensor(classes) else classes,
        "note": note,
    }


def _classifier_module_state(classifier, classes, note=None):
    return _linear_state_from_weight_bias(
        classifier.weight.data,
        classifier.bias.data,
        classes,
        note=note,
    )


def _short_tensor_hash(tensor):
    if tensor is None:
        return ""
    if not torch.is_tensor(tensor):
        tensor = torch.as_tensor(tensor)
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha1(array.tobytes()).hexdigest()[:12]


def _classifier_weight_hash(classifier_state):
    if not classifier_state:
        return ""
    weight_hash = _short_tensor_hash(classifier_state.get("weight"))
    bias_hash = _short_tensor_hash(classifier_state.get("bias"))
    return f"{weight_hash}:{bias_hash}"


def ensure_source_train_scarcity(data, opt, missing_rate):
    applied_rate = getattr(data, "source_missing_rate_applied", None)
    if applied_rate == missing_rate:
        return

    before_size = int(data.train_feature.size(0))
    print(
        f"DATA_LOADER did not apply source scarcity for missing_rate={missing_rate}; "
        f"forcing source-domain subsampling in main.py. before_ntrain={before_size}"
    )
    data.train_feature, data.train_label = data.subsample_train_per_class(
        data.train_feature,
        data.train_label,
        missing_rate,
    )
    data.source_missing_rate_applied = missing_rate
    data.originalclasses = torch.from_numpy(np.unique(data.train_label.numpy()))
    if opt.zst:
        data.originalclasses = torch.arange(len(data.originalclasses))
    data.nclass = len(data.originalclasses) + len(data.newclasses)
    data.ntrain = data.train_feature.size(0)
    data.train_mapped_label = data_preparation.map_label(data.train_label, data.originalclasses)
    print(f"Forced source scarcity complete. after_ntrain={int(data.ntrain)}")


def save_domain_classifier_weights(experiment, method_name, opt, seed, missing_rate):
    output_dir = os.path.join(opt.rootpath, "models", "cexp5_classifier_weights")
    os.makedirs(output_dir, exist_ok=True)

    source_classes = experiment.originalclasses.detach().cpu().clone()
    target_classes = experiment.newclasses.detach().cpu().clone()
    if opt.zst:
        target_classes = target_classes - len(experiment.originalclasses)

    source_state = getattr(experiment, "source_domain_classifier_state", None)
    target_state = getattr(experiment, "target_domain_classifier_state", None)

    if hasattr(experiment, "target_weights"):
        source_state = _linear_state_from_weight_bias(
            experiment.target_weights[:, :-1],
            experiment.target_weights[:, -1],
            source_classes,
            note="source-domain base classifier",
        )

    if hasattr(experiment, "new_model"):
        target_state = _linear_state_from_weight_bias(
            experiment.new_model.fc.weight.data,
            experiment.new_model.fc.bias.data,
            target_classes,
            note="target-domain predicted classifier",
        )

    classifier = None
    if hasattr(experiment, "model"):
        if hasattr(experiment.model, "fc"):
            classifier = experiment.model.fc
        elif hasattr(experiment.model, "label_classifier"):
            classifier = experiment.model.label_classifier
        elif hasattr(experiment.model, "classifier"):
            classifier = experiment.model.classifier
    if classifier is None and hasattr(experiment, "classifier"):
        if hasattr(experiment.classifier, "fc"):
            classifier = experiment.classifier.fc
        else:
            classifier = experiment.classifier

    if classifier is not None and source_state is None:
        source_state = _classifier_module_state(
            classifier,
            source_classes,
            note="classifier used for source-domain labels",
        )

    if classifier is not None and target_state is None:
        target_state = _classifier_module_state(
            classifier,
            target_classes,
            note="classifier used when evaluating target-domain samples; shared with source when the method has no separate target head",
        )

    save_obj = {
        "method": method_name,
        "dataset": opt.dataset,
        "zstfrom": opt.zstfrom,
        "source_dataset": opt.zstfrom if opt.zst else opt.dataset,
        "target_dataset": opt.dataset,
        "zst": opt.zst,
        "seed": seed,
        "train_sample_missing_rate": missing_rate,
        "train_sample_keep_rate": 100 - missing_rate,
        "image_embedding": opt.image_embedding,
        "class_embedding": opt.class_embedding,
        "llm": opt.llm,
        "source_classifier": source_state,
        "target_classifier": target_state,
        "source_train_indices": getattr(experiment, "source_train_indices", None),
        "source_train_counts": getattr(experiment, "source_train_counts", None),
    }
    filename = f"{opt.dataset}_from_{opt.zstfrom}_{method_name}_miss{missing_rate}_seed{seed}.pt"
    torch.save(save_obj, os.path.join(output_dir, filename))
    return save_obj, os.path.join(output_dir, filename)

# assert 
assert opt.factual_branch in ['attention', 'mean', 'none']
assert opt.intervention_branch in ['attention', 'mean', 'none']
assert opt.class_embedding in ["clip", "sbert", "llama", "qwen"]
assert opt.llm in ['gpt4o', 'gpt4omini', 'gemini2.5', 'llama70b', 'qwen_plus']
assert opt.method in ['ICIS', 'ours_without_inv', 'ours_with_inv']
if opt.save_pred_matrix:
    assert opt.strict_eval, "If saving prediction matrices, run with strict_eval to not overwrite"
if opt.vgse_baseline:
    assert opt.vgse_baseline in ['wavg', 'smo']

# path and seed
if not os.path.exists(opt.rootpath):
    os.makedirs(opt.rootpath)

if opt.manualSeed is None:
    seedlist = [random.randint(1, 10000) for _ in range(opt.numSeeds)]
else:
    opt.numSeeds = len(opt.manualSeed)
    seedlist = opt.manualSeed

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# params selection
args = [opt.lr, opt.beta1, opt.nepoch, opt.batch_size, opt.embed_dim, opt.num_layers, opt.new_rate, opt.train_sample_missing_rate]
params = [x if type(x) == list else [x] for x in args]
params = list(itertools.product(*params))

# full model
accs_new_only, accs_gzsl, accs_new, accs_original, hs = [], [], [], [], []
accs_new_only_std, accs_gzsl_std, accs_new_std, accs_original_std, hs_std = [], [], [], [], []
hparam_avg_mses, hparams_min_mses, epoch_min_idx_argmax, epoch_min_idx_mean, hparam_mse_idxs, hparam_loss_idxs, hparams_min_losses, hparam_cos_idxs, hparams_max_cos, hparam_avg_cos_list = [], [], [], [], [], [], [], [], [], []
start_time = datetime.datetime.now()
print(f"Start time: {start_time}")
for _lr, _beta1, _nepoch, _batch_size, _embed_dim, _num_layers, _new_rate, _train_sample_missing_rate in params:
    opt.current_train_sample_missing_rate = _train_sample_missing_rate
    acc_gzsl_seeds_avg, acc_original_seeds_avg, acc_new_seeds_avg, H_seeds_avg, new_zsl_seeds_avg = [], [], [], [], []
    seed_source_train_sizes, seed_source_train_counts, seed_source_index_hashes = [], [], []
    seed_source_classifier_hashes, seed_target_classifier_hashes = [], []
    seed_mse_list, seed_min_mse, seed_min_idx, seed_mse_idx, seed_min_loss, seed_loss_idx, seed_cos_idx, seed_max_cos, seed_avg_cos_list = [], [], [], [], [], [], [], [], []

    for seed in seedlist:
        split_mse_list, split_min_idx_list, split_loss_list, split_loss_idx_list, split_cos_list, split_cos_idx_list, split_cos_full_list = [], [], [], [], [], [], []
        # setting the seed 
        print("Random Seed: ", seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if opt.cuda:
            torch.cuda.manual_seed_all(seed)
        cudnn.benchmark = True
        acc_val_original = 0

        # load data
        data = data_preparation.DATA_LOADER(opt)
        ensure_source_train_scarcity(data, opt, _train_sample_missing_rate)

        # load or train base classification model
        if not os.path.exists(opt.rootpath + '/models/base-classifiers/'):
            os.makedirs(opt.rootpath + '/models/base-classifiers/')
        model_path = classifier_cache_path(opt, seed)
        print(
            f"Source-domain training split: source_dataset={opt.zstfrom if opt.zst else opt.dataset}, "
            f"target_dataset={opt.dataset}, missing_rate={_train_sample_missing_rate}, "
            f"ntrain={data.ntrain}, counts={getattr(data, 'train_sample_counts', {})}"
        )
        print(f"Base classifier cache path: {model_path}")
        if os.path.isfile(model_path) and not opt.overwrite_base_classifier_cache:
            print(f"Existing base classifier at {model_path} detected. Loading model and skipping training.")
            opt.current_base_classifier_model = torch.load(model_path)
        else:
            if os.path.isfile(model_path):
                print(f"Overwriting cached base classifier at {model_path}.")
            base_model, acc_val_original = BASECLASSIFIER(data.train_feature, data_preparation.map_label(data.train_label, data.originalclasses), data, data.nclass, opt.cuda, seedinfo=seed, _lr=_lr, _beta1=_beta1, _nepoch=opt.classifier_nepoch, _batch_size=_batch_size, _embed_dim=_embed_dim, _num_layers=_num_layers, opt=opt).fit()
            torch.save(base_model, model_path)
            opt.current_base_classifier_model = base_model
            print(f"Saved base classifier for dataset {opt.dataset} trained on seed {seed}.")

        value = (opt.image_embedding or "").lower()
        if "resnet101" in value or "resnet-101" in value:
            backbone = "ResNet-101"
        elif "resnet50" in value or "resnet-50" in value:
            backbone = "ResNet-50"
        else:
            backbone = "Unknown"

        if "pretrained" in value or "pre_trained" in value:
            training_type = "pre_trained"
        elif "finetuned" in value or "fine_tuned" in value:
            training_type = "fine_tuned"
        else:
            training_type = "unknown"

        end_time = datetime.now()
        end_timestamp = time.time()


        result = {
            "model_name": backbone,
            "domain_name": opt.zstfrom,
            "training_type": training_type,
            "accuracy": acc_val_original.item(),
            "created_at": start_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
            "updated_at": end_time.strftime("%Y-%m-%d %H:%M:%S.%f")
        }

        with open(opt.save_path + opt.zstfrom + '_' + opt.image_embedding + '.json', "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
