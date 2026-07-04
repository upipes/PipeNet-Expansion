import argparse

from LLM_desc_gen.prompt_suite import PhysicsAwarePromptSuite
from LLM_desc_gen.semantic_prior_generator import SemanticPriorGenerator
from utility.utils import read_lines

def _safe_read(path):
    lines = read_lines(path)
    return [] if lines is None else [line for line in lines if line and line.strip()]


def _classnames(rootpath, dataset):
    names = _safe_read(f"{rootpath}/data/{dataset}/idx2name.txt")
    if names:
        return names
    return _safe_read(f"{rootpath}/prompt/aux_info/classnames_{dataset}.txt")


def _class_descriptions(rootpath, dataset):
    desc_lines = _safe_read(f"{rootpath}/prompt/aux_info/classes_description_{dataset}.txt")
    out = {}
    for line in desc_lines:
        if ":" in line:
            name, desc = line.split(":", 1)
            out[name.strip().lower()] = desc.strip()
    return out


def _views(rootpath, dataset):
    view_lines = _safe_read(f"{rootpath}/prompt/aux_info/views_{dataset}.txt")
    views = []
    for line in view_lines:
        views.append(line.split(". ", 1)[1].strip() if ". " in line else line.strip())
    return views


def _domain(rootpath, dataset):
    return " ".join(_safe_read(f"{rootpath}/prompt/aux_info/domain_{dataset}.txt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rootpath", default=".", help="Workspace root")
    parser.add_argument("--source_dataset", default="SD", help="Source dataset for the current transfer task")
    parser.add_argument("--target_dataset", default="Road", help="Target dataset for the current transfer task")
    args = parser.parse_args()

    suite = PhysicsAwarePromptSuite(args.rootpath)
    prior_generator = SemanticPriorGenerator(args.rootpath)
    for dataset in [args.source_dataset, args.target_dataset]:
        prior_generator.ensure_priors(dataset, force=False)
    suite.build_dataset_prompts(
        dataset=args.source_dataset,
        domain=_domain(args.rootpath, args.source_dataset),
        classes=_classnames(args.rootpath, args.source_dataset),
        class_descriptions=_class_descriptions(args.rootpath, args.source_dataset),
        views=_views(args.rootpath, args.source_dataset),
        role="source",
    )
    suite.build_dataset_prompts(
        dataset=args.target_dataset,
        domain=_domain(args.rootpath, args.target_dataset),
        classes=_classnames(args.rootpath, args.target_dataset),
        class_descriptions=_class_descriptions(args.rootpath, args.target_dataset),
        views=_views(args.rootpath, args.target_dataset),
        role="target",
    )
    suite.build_transfer_prompts(
        source_dataset=args.source_dataset,
        target_dataset=args.target_dataset,
        source_domain=_domain(args.rootpath, args.source_dataset),
        target_domain=_domain(args.rootpath, args.target_dataset),
        classes=_classnames(args.rootpath, args.target_dataset),
        class_descriptions=_class_descriptions(args.rootpath, args.target_dataset),
        views=_views(args.rootpath, args.target_dataset),
    )
    suite.build_target_internal_prompts(
        target_dataset=args.target_dataset,
        target_domain=_domain(args.rootpath, args.target_dataset),
        classes=_classnames(args.rootpath, args.target_dataset),
        class_descriptions=_class_descriptions(args.rootpath, args.target_dataset),
    )
