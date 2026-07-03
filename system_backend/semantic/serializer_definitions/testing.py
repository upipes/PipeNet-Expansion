def _ordered_class_accuracy(run):
    source = run.source_dataset or ""
    target = run.target_dataset or ""
    order = ["Cavity", "Crack", "Metal Pipeline"] if any(
        marker in f"{source} {target}"
        for marker in ["Sandy", "Silty", "Backfill", "Layered", "a1", "a2", "a3", "a4"]
    ) else ["Cavity", "Crack", "Loose", "Normal", "Pipeline"]
    values = run.class_accuracy or {}
    ordered = {name: values[name] for name in order if name in values}
    for key, value in values.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def serialize_model_training_run(run):
    return {
        "id": run.id,
        "methodName": run.method_name,
        "modelName": run.model_name,
        "sourceDataset": run.source_dataset,
        "targetDataset": run.target_dataset,
        "backbone": run.backbone,
        "semanticGenerator": run.semantic_generator,
        "embeddingModel": run.embedding_model,
        "optimizer": run.optimizer,
        "knowledgeItems": run.knowledge_items,
        "refinementIterations": run.refinement_iterations,
        "learningRate": run.learning_rate,
        "batchSize": run.batch_size,
        "epochs": run.epochs,
        "accuracy": float(run.accuracy),
        "acc": f"{float(run.accuracy):.1f}%",
        "classAccuracy": _ordered_class_accuracy(run),
        "methodCheckpointPath": run.method_checkpoint_path or "",
        "modelDescription": run.model_description or "",
        "description": run.model_description or "",
        "createdAt": run.created_at.isoformat() if run.created_at else None,
        "updatedAt": run.updated_at.isoformat() if run.updated_at else None,
    }

