import json
import random
from decimal import Decimal

from django.db import transaction
from django.db.models import Max, Min, Q
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
@csrf_exempt
def domain_list(request):
    if request.method == "POST":
        try:
            payload = json.loads(request.body.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            return JsonResponse({"message": "Dataset input save fails: invalid JSON payload."}, status=400)

        dataset_name = str(payload.get("datasetName") or payload.get("name") or "").strip()
        dataset_code = str(payload.get("datasetCode") or payload.get("code") or dataset_name).strip()
        domain_type = str(payload.get("domainType") or payload.get("type") or "road").strip().lower()
        condition_text = str(payload.get("roleCondition") or payload.get("areaCondition") or "").strip()
        area_description = str(payload.get("areaDescription") or "").strip()
        class_text = str(
            payload.get("classList")
            or payload.get("classes")
            or payload.get("supportedClasses")
            or payload.get("Class")
            or ""
        ).strip()

        if not dataset_name or not domain_type or not condition_text or not area_description or not class_text:
            return JsonResponse(
                {"message": "Dataset input save fails: required fields are missing."},
                status=400,
            )

        if domain_type not in {AreaDomain.ROAD, AreaDomain.SOIL}:
            return JsonResponse(
                {"message": "Dataset input save fails: domain type is invalid."},
                status=400,
            )

        try:
            frequency_min, frequency_max = _range_values(payload, "frequency")
            water_min, water_max = _range_values(payload, "water", decimal=True)
            permittivity_min, permittivity_max = _range_values(payload, "permittivity", decimal=True)
            conductivity_min, conductivity_max = _range_values(payload, "conductivity", decimal=True)

            class_names = []
            for raw_name in class_text.split(","):
                class_name = raw_name.strip()
                if not class_name:
                    continue
                if class_name.lower() == "noraml":
                    class_name = "Normal"
                if class_name not in class_names:
                    class_names.append(class_name)

            if not class_names:
                return JsonResponse(
                    {"message": "Dataset input save fails: class list is invalid."},
                    status=400,
                )

            existing = AreaDomain.objects.filter(code=dataset_code).first()
            next_order = (
                existing.display_order
                if existing
                else (AreaDomain.objects.aggregate(min_order=Min("display_order"))["min_order"] or 1) - 1
            )

            with transaction.atomic():
                domain, _created = AreaDomain.objects.update_or_create(
                    code=dataset_code or dataset_name,
                    defaults={
                        "name": dataset_name,
                        "domain_type": domain_type,
                        "display_order": next_order,
                        "is_active": True,
                        "condition_text": condition_text,
                        "road_surface": str(payload.get("roadSurface") or "").strip(),
                        "frequency_min": frequency_min,
                        "frequency_max": frequency_max,
                        "time_window_ns": _int_or_none(payload.get("timeWindow")),
                        "sand_percent": _decimal_or_none(payload.get("sandPercent")),
                        "silt_percent": _decimal_or_none(payload.get("siltPercent")),
                        "clay_percent": _decimal_or_none(payload.get("clayPercent")),
                        "water_min": water_min,
                        "water_max": water_max,
                        "permittivity_min": permittivity_min,
                        "permittivity_max": permittivity_max,
                        "conductivity_min": conductivity_min,
                        "conductivity_max": conductivity_max,
                        "peplinski_dimension": _decimal_or_none(payload.get("peplinskiDimension")),
                        "area_description": area_description,
                        "signal_behavior": str(payload.get("signalBehavior") or "").strip(),
                        "semantic_usage": str(payload.get("semanticUsage") or "").strip(),
                    },
                )
                SemanticCategory.objects.filter(domain=domain).delete()
                SemanticCategory.objects.bulk_create(
                    [SemanticCategory(domain=domain, name=class_name) for class_name in class_names]
                )
        except ValueError as exc:
            return JsonResponse({"message": f"Dataset input save fails: {exc}"}, status=400)

        return JsonResponse(
            {
                "message": "Dataset input is saved.",
                "domain": serialize_domain_detail(domain),
            },
            status=201,
        )

    domains = AreaDomain.objects.filter(is_active=True)
    data = [serialize_domain_summary(domain) for domain in domains]
    return JsonResponse(
        {
            "domains": data,
            "defaults": {
                "originalCode": "a1",
                "newCode": "a3",
                "originalName": _domain_by_name_or_code("a1 Sandy Loam").name if _domain_by_name_or_code("a1 Sandy Loam") else "a1 Sandy Loam",
                "newName": _domain_by_name_or_code("a3 Urban Backfill Soil").name if _domain_by_name_or_code("a3 Urban Backfill Soil") else "a3 Urban Backfill Soil",
            },
        }
    )


def domain_detail(request, domain_id):
    domain = get_object_or_404(AreaDomain, pk=domain_id, is_active=True)
    return JsonResponse(serialize_domain_detail(domain))


def domain_detail_by_code(request, code):
    domain = _domain_by_name_or_code(code)
    if not domain:
        return JsonResponse({"message": "Domain detail loading fails: domain does not exist."}, status=404)
    return JsonResponse(serialize_domain_detail(domain))
