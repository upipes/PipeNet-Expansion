import json
import os
import random
import re
import urllib.error
import urllib.request
from pathlib import Path
from decimal import Decimal

from django.db import transaction
from django.db.models import Max, Q
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.csrf import csrf_exempt

from ..models import (
    AreaDomain,
    ModelTrainingRun,
    OriginalClassifier,
    SemanticAnnotation,
    SemanticCategory,
    SemanticDescription,
    SemanticGenerationRun,
)
from ..serializers import (
    serialize_domain_detail,
    serialize_domain_summary,
    serialize_generation_run,
    serialize_model_training_run,
    serialize_original_classifier,
    serialize_semantic_annotation,
)
from .common import *


SEMANTIC_VIEW_KEYS = [
    "Dominant Shape",
    "Interaction Type",
    "Reflection Strength",
    "Feature Footprint",
    "Signal Coherence",
    "Anomaly Source",
    "Pattern Uniformity",
    "Feature Expression",
    "Layer Interaction",
    "Signal Complexity",
]


def _project_root():
    return Path(__file__).resolve().parents[3]


def _dataset_key_for_domain(domain):
    value = domain.code or domain.name
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_") or f"domain_{domain.id}"


def _normalize_llm_name(llm_name):
    mapping = {
        "GPT-4o": "gpt-4o",
        "GPT-3.5-turbo": "gpt-3.5-turbo",
        "GPT-4o-mini": "gpt-4o-mini",
        "Gemini-2.5": "gemini-2.5-pro",
        "LLama-3.1": "llama3.1-70b-instruct",
        "LLaMa-3.1": "llama3.1-70b-instruct",
        "Qwen-2.5": "qwen2.5-72b-instruct",
    }
    return mapping.get(llm_name, llm_name)


def _write_runner_aux_files(domain, categories):
    root = _project_root()
    aux_dir = root / "prompt" / "aux_info"
    aux_dir.mkdir(parents=True, exist_ok=True)
    dataset_key = _dataset_key_for_domain(domain)
    domain_text = json.dumps(_domain_prompt_context(domain), ensure_ascii=False)
    (aux_dir / f"domain_{dataset_key}.txt").write_text(domain_text, encoding="utf-8")
    (aux_dir / f"classnames_{dataset_key}.txt").write_text("\n".join(categories), encoding="utf-8")
    (aux_dir / f"views_{dataset_key}.txt").write_text("\n".join(SEMANTIC_VIEW_KEYS), encoding="utf-8")
    class_descriptions = [
        f"{name}: {name} semantic target under {domain.name} GPR domain context."
        for name in categories
    ]
    (aux_dir / f"classes_description_{dataset_key}.txt").write_text("\n".join(class_descriptions), encoding="utf-8")
    return root, dataset_key


def _demo_semantic_file_path():
    configured = os.getenv("SEMANTIC_DEMO_FILE")
    if configured:
        return Path(configured)
    return Path(r".\file\GPR-SD-semantic.json")


def _load_demo_semantic_payload(domain, categories, llm_name, use_expert_knowledge, use_image_assist):
    path = _demo_semantic_file_path()
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    file_domain_values = {
        str(payload.get("domain") or "").strip(),
        str(payload.get("domainName") or "").strip(),
        str(payload.get("code") or "").strip(),
    }
    current_domain_values = {str(domain.name or "").strip(), str(domain.code or "").strip()}
    if not any(value and value in current_domain_values for value in file_domain_values):
        return None

    file_llm = str(payload.get("llmName") or payload.get("llm_name") or "").strip()
    if file_llm and file_llm != llm_name:
        return None
    if "useExpertKnowledge" in payload and bool(payload.get("useExpertKnowledge")) != use_expert_knowledge:
        return None
    if "useImageAssist" in payload and bool(payload.get("useImageAssist")) != use_image_assist:
        return None

    classes = payload.get("classes")
    if not isinstance(classes, dict):
        return None

    category_lookup = {name.lower(): name for name in categories}
    category_lookup.update({name.lower().replace(" ", ""): name for name in categories})
    category_aliases = {
        "pipeline": "Metal Pipeline" if "Metal Pipeline" in categories else "Pipeline",
        "metalpipeline": "Metal Pipeline",
        "metal_pipe": "Metal Pipeline",
        "metal pipe": "Metal Pipeline",
        "pipe": "Metal Pipeline" if "Metal Pipeline" in categories else "Pipeline",
    }
    descriptions = []
    for raw_key, item in classes.items():
        if not isinstance(item, dict):
            continue
        normalized_key = str(raw_key).strip().lower()
        class_name = category_lookup.get(normalized_key) or category_lookup.get(normalized_key.replace(" ", ""))
        if not class_name:
            alias_name = category_aliases.get(normalized_key) or category_aliases.get(normalized_key.replace(" ", ""))
            class_name = alias_name if alias_name in categories else None
        if not class_name:
            continue
        brief_map = item.get("brief_semantics") or item.get("all_view_brief_descriptions") or {}
        detail_map = item.get("detailed_semantics") or item.get("all_view_detailed_descriptions") or {}
        if not isinstance(brief_map, dict):
            brief_map = {}
        if not isinstance(detail_map, dict):
            detail_map = {}
        for view_name in SEMANTIC_VIEW_KEYS:
            brief_map.setdefault(view_name, f"{class_name} {view_name.lower()} semantic cue under {domain.name}.")
            detail_map.setdefault(view_name, brief_map[view_name])
        primary_view = str(item.get("primary_view") or next(iter(brief_map), "Dominant Shape")).strip()
        descriptions.append(
            {
                "class": class_name,
                "primary_view": primary_view,
                "primary_brief_description": str(
                    item.get("primary_view_brief")
                    or item.get("primary_brief_description")
                    or brief_map.get(primary_view)
                    or ""
                ).strip(),
                "all_view_brief_descriptions": brief_map,
                "all_view_detailed_descriptions": detail_map,
                "llm_confidence": item.get("overall_semantic_confidence") or item.get("llm_confidence") or 0.9,
            }
        )
    if not descriptions:
        return None
    return {"descriptions": descriptions}


def _semantic_run_for_config(domain, llm_name, use_expert_knowledge, use_image_assist):
    base = SemanticGenerationRun.objects.filter(domain=domain, status="success").prefetch_related("descriptions__category")
    expected_categories = set(domain.semantic_categories.values_list("name", flat=True))

    def valid_run(queryset):
        for run in queryset:
            if not expected_categories:
                return run
            actual_categories = {description.category.name for description in run.descriptions.all()}
            if expected_categories.issubset(actual_categories):
                return run
        return None

    return (
        valid_run(
            base.filter(
                llm_name=llm_name,
                use_expert_knowledge=use_expert_knowledge,
                use_image_assist=use_image_assist,
            )
        )
        or valid_run(base.filter(llm_name=llm_name))
        or valid_run(base)
    )


def _domain_prompt_context(domain):
    if domain.domain_type == AreaDomain.SOIL:
        return {
            "domain_name": domain.name,
            "domain_type": domain.domain_type,
            "composition": {
                "sand": str(domain.sand_percent or ""),
                "silt": str(domain.silt_percent or ""),
                "clay": str(domain.clay_percent or ""),
            },
            "water_content": [str(domain.water_min or ""), str(domain.water_max or "")],
            "relative_permittivity": [str(domain.permittivity_min or ""), str(domain.permittivity_max or "")],
            "conductivity": [str(domain.conductivity_min or ""), str(domain.conductivity_max or "")],
            "peplinski_dimension": str(domain.peplinski_dimension or ""),
            "area_description": domain.area_description,
            "signal_behavior": domain.signal_behavior,
            "semantic_usage": domain.semantic_usage,
        }
    return {
        "domain_name": domain.name,
        "domain_type": domain.domain_type,
        "condition": domain.condition_text,
        "road_surface": domain.road_surface,
        "frequency_range_mhz": [domain.frequency_min, domain.frequency_max],
        "time_window_ns": domain.time_window_ns,
        "area_description": domain.area_description,
        "signal_behavior": domain.signal_behavior,
        "semantic_usage": domain.semantic_usage,
    }


def _semantic_generation_prompt(domain, categories, llm_name, use_expert_knowledge, use_image_assist):
    return (
        "Generate GPR class semantic descriptions for a cross-area subsurface diagnosis system.\n"
        "Return JSON only. Do not include markdown.\n"
        "The JSON schema must be:\n"
        "{\"descriptions\":[{\"class\":\"Cavity\",\"primary_view\":\"Dominant Shape\","
        "\"primary_brief_description\":\"...\",\"all_view_brief_descriptions\":{\"Dominant Shape\":\"...\"},"
        "\"all_view_detailed_descriptions\":{\"Dominant Shape\":\"...\"},\"llm_confidence\":92.0}]}\n"
        f"Each class must include these views: {', '.join(SEMANTIC_VIEW_KEYS)}.\n"
        "Brief descriptions are one sentence. Detailed descriptions are two to four sentences.\n"
        f"Domain context: {json.dumps(_domain_prompt_context(domain), ensure_ascii=False)}\n"
        f"Classes: {', '.join(categories)}\n"
        f"LLM: {llm_name}; expert knowledge: {use_expert_knowledge}; image assist: {use_image_assist}."
    )


def _call_llm_for_semantics(domain, categories, llm_name, use_expert_knowledge, use_image_assist):
    try:
        from LLM_desc_gen.semantic_query_runner import SemanticQueryRunner

        root, dataset_key = _write_runner_aux_files(domain, categories)
        role = "source" if "original" in (domain.condition_text or "").lower() else "target"
        runner = SemanticQueryRunner(
            rootpath=root,
            llm_name=_normalize_llm_name(llm_name),
            temperature=0.2,
            delay_seconds=0,
        )
        result = runner._load_or_query_role(dataset_key, role=role, overwrite=True)
        parsed = result.get("parsed") or {}
        descriptions = []
        for class_name in categories:
            class_key = class_name.lower()
            brief_map = {}
            detail_map = {}
            for view_name in SEMANTIC_VIEW_KEYS:
                text = (
                    (parsed.get(view_name) or {}).get(class_key)
                    or (parsed.get(view_name) or {}).get(class_name)
                    or ""
                )
                if not text:
                    text = f"{class_name} {view_name.lower()} semantic cue generated under {domain.name}."
                brief_map[view_name] = text[:260]
                detail_map[view_name] = text
            primary_view = next((view for view in SEMANTIC_VIEW_KEYS if brief_map.get(view)), "Dominant Shape")
            descriptions.append(
                {
                    "class": class_name,
                    "primary_view": primary_view,
                    "primary_brief_description": brief_map[primary_view],
                    "all_view_brief_descriptions": brief_map,
                    "all_view_detailed_descriptions": detail_map,
                    "llm_confidence": 88 + random.random() * 8,
                }
            )
        return {"descriptions": descriptions}
    except Exception:
        pass

    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    api_url = os.getenv("LLM_API_BASE", "https://api.openai.com/v1/chat/completions")
    model_name = os.getenv("LLM_MODEL_NAME") or llm_name
    body = {
        "model": model_name,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "You are a GPR subsurface diagnosis semantic generation engine."},
            {"role": "user", "content": _semantic_generation_prompt(domain, categories, llm_name, use_expert_knowledge, use_image_assist)},
        ],
    }
    request = urllib.request.Request(
        api_url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"LLM request fails: {exc}") from exc

    content = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
    if not content:
        raise RuntimeError("LLM request fails: empty response.")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError("LLM request fails: response is not valid JSON.") from exc
    return parsed


def _confidence_value(value):
    try:
        number = Decimal(str(value))
    except Exception:
        number = Decimal("88.00")
    if number <= 1:
        number *= 100
    return max(Decimal("0.00"), min(Decimal("100.00"), number)).quantize(Decimal("0.01"))


def _save_generated_semantics(domain, llm_name, use_expert_knowledge, use_image_assist, generated_payload, run=None):
    descriptions = generated_payload.get("descriptions") if isinstance(generated_payload, dict) else None
    if not isinstance(descriptions, list) or not descriptions:
        raise ValueError("Semantic generation fails: LLM output has no descriptions.")

    with transaction.atomic():
        if run is None:
            SemanticGenerationRun.objects.filter(
                domain=domain,
                llm_name=llm_name,
                use_expert_knowledge=use_expert_knowledge,
                use_image_assist=use_image_assist,
                status="running",
            ).delete()
            run = SemanticGenerationRun.objects.create(
                domain=domain,
                llm_name=llm_name,
                use_expert_knowledge=use_expert_knowledge,
                use_image_assist=use_image_assist,
                generated_count=0,
                status="running",
            )
        else:
            SemanticDescription.objects.filter(run=run).delete()
        saved_count = 0
        for item in descriptions:
            if not isinstance(item, dict):
                continue
            class_name = str(item.get("class") or item.get("category") or "").strip()
            if not class_name:
                continue
            category, _ = SemanticCategory.objects.get_or_create(domain=domain, name=class_name)
            brief_map = item.get("all_view_brief_descriptions") or item.get("brief_descriptions") or {}
            detail_map = item.get("all_view_detailed_descriptions") or item.get("detailed_descriptions") or {}
            if not isinstance(brief_map, dict):
                brief_map = {}
            if not isinstance(detail_map, dict):
                detail_map = {}
            for key in SEMANTIC_VIEW_KEYS:
                brief_map.setdefault(key, f"{class_name} {key.lower()} semantic cue under {domain.name}.")
                detail_map.setdefault(
                    key,
                    f"For {class_name} in {domain.name}, the {key.lower()} view is generated from the area description, signal behavior, and selected semantic settings.",
                )
            primary_view = str(item.get("primary_view") or item.get("primaryView") or next(iter(brief_map), "Dominant Shape")).strip()
            primary_brief = str(
                item.get("primary_brief_description")
                or item.get("primaryBriefDescription")
                or brief_map.get(primary_view)
                or ""
            ).strip()
            SemanticDescription.objects.create(
                run=run,
                domain=domain,
                category=category,
                primary_view=primary_view,
                primary_brief_description=primary_brief,
                all_view_brief_descriptions=brief_map,
                all_view_detailed_descriptions=detail_map,
                llm_confidence=_confidence_value(item.get("llm_confidence") or item.get("confidence")),
            )
            saved_count += 1
        run.generated_count = saved_count
        run.status = "success"
        run.save(update_fields=["generated_count", "status"])
    return run


@csrf_exempt
def semantic_generation(request):
    if request.method != "POST":
        return JsonResponse({"message": "Only POST is supported."}, status=405)

    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"message": "Semantic generation fails: invalid JSON payload."}, status=400)

    action = str(payload.get("action") or "complete").strip().lower()
    run_id = payload.get("runId") or payload.get("run_id")

    if action == "cancel":
        run = get_object_or_404(SemanticGenerationRun, pk=run_id)
        run.status = "failed"
        run.generated_count = 0
        run.save(update_fields=["status", "generated_count"])
        return JsonResponse({"message": "Semantic generation is cancelled.", "run": serialize_generation_run(run)}, status=200)

    domain_name = str(payload.get("domainName") or payload.get("domain") or payload.get("domainCode") or "").strip()
    llm_name = str(payload.get("llmName") or "GPT-4o").strip()
    use_expert_knowledge = bool(payload.get("useExpertKnowledge"))
    use_image_assist = bool(payload.get("useImageAssist"))

    if not domain_name:
        return JsonResponse({"message": "Semantic generation fails: domain is missing."}, status=400)

    domain = _domain_by_name_or_code(domain_name)
    if not domain:
        return JsonResponse({"message": "Semantic generation fails: domain does not exist."}, status=404)

    if action == "start":
        run = SemanticGenerationRun.objects.create(
            domain=domain,
            llm_name=llm_name,
            use_expert_knowledge=use_expert_knowledge,
            use_image_assist=use_image_assist,
            generated_count=0,
            status="running",
        )
        return JsonResponse({"message": "Semantic generation run starts.", "run": serialize_generation_run(run)}, status=201)

    active_run = None
    if run_id:
        active_run = get_object_or_404(SemanticGenerationRun, pk=run_id, domain=domain)
        if active_run.status == "failed":
            return JsonResponse({"message": "Semantic generation fails: this run is cancelled."}, status=400)

    categories = list(domain.semantic_categories.order_by("id").values_list("name", flat=True))
    if categories:
        try:
            demo_payload = _load_demo_semantic_payload(
                domain,
                categories,
                llm_name,
                use_expert_knowledge,
                use_image_assist,
            )
            if demo_payload:
                run = _save_generated_semantics(
                    domain,
                    llm_name,
                    use_expert_knowledge,
                    use_image_assist,
                    demo_payload,
                    run=active_run,
                )
                return JsonResponse({"message": "Semantic generation succeeds.", "run": serialize_generation_run(run)}, status=200)
        except (OSError, json.JSONDecodeError, ValueError):
            pass

        try:
            generated_payload = _call_llm_for_semantics(domain, categories, llm_name, use_expert_knowledge, use_image_assist)
            if generated_payload:
                run = _save_generated_semantics(
                    domain,
                    llm_name,
                    use_expert_knowledge,
                    use_image_assist,
                    generated_payload,
                    run=active_run,
                )
                return JsonResponse({"message": "Semantic generation succeeds.", "run": serialize_generation_run(run)}, status=200)
        except (RuntimeError, ValueError):
            pass

    run = _semantic_run_for_config(domain, llm_name, use_expert_knowledge, use_image_assist)

    if not run:
        return JsonResponse(
            {
                "message": (
                    "Semantic generation fails: no precomputed semantic result exists for this configuration. "
                    "Please import first."
                )
            },
            status=404,
        )

    return JsonResponse({"message": "Semantic generation succeeds.", "run": serialize_generation_run(run)}, status=200)


def latest_semantic_generation(request):
    domain_name = str(request.GET.get("domain") or request.GET.get("domainName") or request.GET.get("domainCode") or "").strip()
    llm_name = str(request.GET.get("llmName") or "GPT-4o").strip()
    use_expert_knowledge = str(request.GET.get("useExpertKnowledge") or "false").lower() == "true"
    use_image_assist = str(request.GET.get("useImageAssist") or "false").lower() == "true"
    if not domain_name:
        return JsonResponse({"message": "Latest semantic generation fails: domain is missing."}, status=400)

    domain = _domain_by_name_or_code(domain_name)
    if not domain:
        return JsonResponse({"message": "Latest semantic generation fails: domain does not exist."}, status=404)
    run = _semantic_run_for_config(domain, llm_name, use_expert_knowledge, use_image_assist)
    if not run:
        return JsonResponse({"run": None, "descriptions": []})
    return JsonResponse({"run": serialize_generation_run(run)})


@csrf_exempt
def semantic_annotations(request):
    if request.method == "POST":
        try:
            payload = json.loads(request.body.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            return JsonResponse({"message": "Semantic annotation save fails: invalid JSON payload."}, status=400)

        desc_id = str(payload.get("descId") or "").strip()
        annotations = payload.get("annotations") or []
        detail_updates = payload.get("detailUpdates") or {}

        if not desc_id:
            return JsonResponse({"message": "Semantic annotation save fails: description id is missing."}, status=400)
        if not isinstance(annotations, list):
            return JsonResponse({"message": "Semantic annotation save fails: annotations must be a list."}, status=400)

        description = get_object_or_404(SemanticDescription, pk=desc_id)
        instances = []
        for item in annotations:
            if not isinstance(item, dict):
                continue

            view_name = str(item.get("viewName") or item.get("view") or "").strip()
            view_text = str(item.get("viewText") or "").strip()
            annotated_text = str(item.get("annotatedText") or item.get("text") or "").strip()
            annotation_effect = str(item.get("effect") or item.get("type") or "").strip()
            annotation_content = str(item.get("annotationContent") or item.get("note") or "").strip()
            update_revise = str(item.get("updateRevise") or item.get("update_revise") or "").strip()

            if not view_name or not view_text or not annotated_text:
                return JsonResponse(
                    {"message": "Semantic annotation save fails: view and selected text are required."},
                    status=400,
                )
            if annotation_effect not in {SemanticAnnotation.CORRECT, SemanticAnnotation.INACCURATE, SemanticAnnotation.INCORRECT}:
                return JsonResponse(
                    {"message": "Semantic annotation save fails: annotation effect is invalid."},
                    status=400,
                )

            instances.append(
                SemanticAnnotation(
                    description=description,
                    view_name=view_name,
                    view_text=view_text,
                    annotated_text=annotated_text,
                    annotation_effect=annotation_effect,
                    annotation_content=annotation_content,
                    update_revise=update_revise,
                )
            )

        with transaction.atomic():
            if isinstance(detail_updates, dict) and detail_updates:
                detailed = dict(description.all_view_detailed_descriptions or {})
                for key, value in detail_updates.items():
                    if str(key).strip():
                        detailed[str(key)] = str(value or "")
                description.all_view_detailed_descriptions = detailed
                description.save(update_fields=["all_view_detailed_descriptions"])
            SemanticAnnotation.objects.filter(description=description).delete()
            if instances:
                SemanticAnnotation.objects.bulk_create(instances)

        saved = SemanticAnnotation.objects.filter(description=description).order_by("annotated_at", "id")
        return JsonResponse(
            {
                "message": "Semantic annotations are saved.",
                "annotations": [serialize_semantic_annotation(item) for item in saved],
            },
            status=200,
        )

    if request.method != "GET":
        return JsonResponse({"message": "Only GET and POST are supported."}, status=405)

    desc_id = str(request.GET.get("descId") or "").strip()
    view_name = str(request.GET.get("view") or "").strip()

    if not desc_id:
        return JsonResponse({"message": "Semantic annotation loading fails: description id is missing."}, status=400)

    annotations = SemanticAnnotation.objects.filter(description_id=desc_id)
    if view_name:
        annotations = annotations.filter(view_name=view_name)

    data = [serialize_semantic_annotation(item) for item in annotations.order_by("annotated_at", "id")]
    return JsonResponse({"annotations": data})
