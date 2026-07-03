from .domains import serialize_domain_summary
from .semantics import serialize_semantic_description

def serialize_generation_run(run):
    return {
        "id": run.id,
        "domain": serialize_domain_summary(run.domain),
        "llmName": run.llm_name,
        "useExpertKnowledge": run.use_expert_knowledge,
        "useImageAssist": run.use_image_assist,
        "status": run.status,
        "generatedCount": run.generated_count,
        "createdAt": run.generated_at.isoformat() if run.generated_at else None,
        "descriptions": [
            serialize_semantic_description(item)
            for item in run.descriptions.select_related("category").all()
        ],
    }


