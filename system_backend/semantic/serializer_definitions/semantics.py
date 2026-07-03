def serialize_semantic_description(description):
    brief_map = description.all_view_brief_descriptions or {}
    detailed_map = description.all_view_detailed_descriptions or {}
    ordered_keys = list(brief_map.keys())
    for key in detailed_map.keys():
        if key not in ordered_keys:
            ordered_keys.append(key)
    views = [
        {
            "key": key,
            "briefDescription": brief_map.get(key, ""),
            "description": detailed_map.get(key, brief_map.get(key, "")),
            "detailedDescription": detailed_map.get(key, brief_map.get(key, "")),
        }
        for key in ordered_keys
    ]

    return {
        "id": description.id,
        "descId": description.id,
        "categoryId": description.category_id,
        "cls": description.category.name,
        "view": description.primary_view,
        "text": description.primary_brief_description,
        "briefDescription": description.primary_brief_description,
        "allViewBriefDescriptions": brief_map,
        "allViewDetailedDescriptions": detailed_map,
        "score": float(description.llm_confidence),
        "status": "Generated",
        "details": views,
        "views": views,
        "createdAt": description.generated_at.isoformat() if description.generated_at else None,
    }


def serialize_semantic_annotation(annotation):
    return {
        "id": annotation.id,
        "descId": annotation.description_id,
        "view": annotation.view_name,
        "viewName": annotation.view_name,
        "viewText": annotation.view_text,
        "text": annotation.annotated_text,
        "annotatedText": annotation.annotated_text,
        "type": annotation.annotation_effect,
        "effect": annotation.annotation_effect,
        "note": annotation.annotation_content or "",
        "annotationContent": annotation.annotation_content or "",
        "updateRevise": getattr(annotation, "update_revise", "") or "",
        "update_revise": getattr(annotation, "update_revise", "") or "",
        "annotatedAt": annotation.annotated_at.isoformat() if annotation.annotated_at else None,
    }

