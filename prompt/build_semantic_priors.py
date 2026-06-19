import argparse

from LLM_desc_gen.semantic_prior_generator import SemanticPriorGenerator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rootpath", default=".", help="Workspace root")
    parser.add_argument("--source_dataset", default="SD", help="Source dataset for the current transfer task")
    parser.add_argument("--target_dataset", default="Road", help="Target dataset for the current transfer task")
    parser.add_argument("--llm", default="gpt4o", help="LLM name")
    parser.add_argument("--force", action="store_true", default=False, help="Regenerate priors even if files already exist")
    args = parser.parse_args()

    generator = SemanticPriorGenerator(rootpath=args.rootpath, llm_name=args.llm)
    for dataset in [args.source_dataset, args.target_dataset]:
        generator.ensure_priors(dataset, force=args.force)
