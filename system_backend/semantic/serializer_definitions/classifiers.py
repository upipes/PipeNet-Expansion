def classifier_type_label(value):
    return {
        "fine_tuned": "Fine-tuned",
        "pre_trained": "Pre-trained",
    }.get(value, value)


def serialize_original_classifier(classifier):
    return {
        "id": classifier.id,
        "name": classifier.model_name,
        "modelName": classifier.model_name,
        "source": classifier.domain_name,
        "original": classifier.domain_name,
        "sourceCode": classifier.domain.code if classifier.domain else classifier.domain_name,
        "domainId": classifier.domain_id,
        "type": classifier_type_label(classifier.training_type),
        "trainingType": classifier.training_type,
        "acc": f"{float(classifier.accuracy):.1f}%",
        "accuracy": float(classifier.accuracy),
        "createdAt": classifier.created_at.isoformat() if classifier.created_at else None,
        "updatedAt": classifier.updated_at.isoformat() if classifier.updated_at else None,
        "description": classifier.model_description,
        "modelDescription": classifier.model_description,
        "modelFilePath": classifier.model_file_path or "",
        "model_file_path": classifier.model_file_path or "",
    }

