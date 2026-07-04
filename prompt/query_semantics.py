import argparse

from LLM_desc_gen.semantic_query_runner import SemanticQueryRunner

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rootpath", default=".", help="Workspace root")
    parser.add_argument("--source_dataset", default="SD", help="Source dataset for the current transfer task")
    parser.add_argument("--target_dataset", default="Road", help="Target dataset for the current transfer task")
    parser.add_argument("--llm", default="gpt4o", help="LLM backend name")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--delay_seconds", type=float, default=0, help="Delay between queries")
    parser.add_argument("--overwrite", action="store_true", help="Re-query even if outputs already exist")
    args = parser.parse_args()

    runner = SemanticQueryRunner(
        rootpath=args.rootpath,
        llm_name=args.llm,
        temperature=args.temperature,
        delay_seconds=args.delay_seconds,
    )
    runner.run_task(
        source_dataset=args.source_dataset,
        target_dataset=args.target_dataset,
        overwrite=args.overwrite,
    )
