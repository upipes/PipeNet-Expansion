import argparse

from LLM_desc_gen.semantic_prompt_builder import rebuild_semantics


parser = argparse.ArgumentParser()
parser.add_argument("--rootpath", default=".", help="Workspace root")
parser.add_argument("--source_dataset", default="SD", help="Source dataset for the current transfer task")
parser.add_argument("--target_dataset", default="Road", help="Target dataset for the current transfer task")
parser.add_argument("--llm", default="gpt4o", help="Semantic source name used in output file naming")
parser.add_argument("--backend", default="sbert", choices=["clip", "sbert", "llama", "qwen"], help="Embedding backend name")


if __name__ == "__main__":
    args = parser.parse_args()
    rebuild_semantics(
        rootpath=args.rootpath,
        source_dataset=args.source_dataset,
        target_dataset=args.target_dataset,
        llm_name=args.llm,
        backend=args.backend,
    )
