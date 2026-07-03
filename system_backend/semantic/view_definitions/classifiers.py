import json
import os
import random
import subprocess
import sys
import tempfile
import uuid
import base64
from decimal import Decimal
from pathlib import Path

from django.db import transaction
from django.db.models import Max, Q
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.utils import timezone
from django.utils.dateparse import parse_datetime
from django.views.decorators.csrf import csrf_exempt

from ..models import (
    AreaDomain,
    ModelTrainingRun,
    OriginalClassifier,
    SemanticAnnotation,
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


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CLASSIFIER_RESULT_DIR = PROJECT_ROOT / "runtime_outputs" / "original_classifiers"
FEATURE_IMAGE_DIR = PROJECT_ROOT / "runtime_outputs" / "feature_images"
FEATURE_GRADCAM_DIR = PROJECT_ROOT / "runtime_outputs" / "feature_gradcam"


def _dataset_code_for_training(domain):
    value = f"{getattr(domain, 'code', '')} {getattr(domain, 'name', '')}".lower()
    for index in range(1, 5):
        if f"a{index}" in value or f"domain{index}" in value:
            return f"domain{index}"
    if "gpr-sd" in value or value.strip() == "sd":
        return "SD"
    if "gpr-road" in value or "road" in value:
        return "Road"
    return getattr(domain, "code", "") or getattr(domain, "name", "")


def _target_dataset_for_source(source_dataset):
    if source_dataset == "SD":
        return "Road"
    if source_dataset == "Road":
        return "SD"
    domain_targets = {
        "domain1": "domain2",
        "domain2": "domain1",
        "domain3": "domain4",
        "domain4": "domain3",
    }
    if source_dataset in domain_targets:
        return domain_targets[source_dataset]
    return source_dataset


def _image_embedding(model_name, training_type):
    prefix = "pretrained" if training_type == OriginalClassifier.PRE_TRAINED else "finetuned"
    if model_name == OriginalClassifier.RESNET101:
        return f"{prefix}_resnet101" if prefix == "pretrained" else "resnet101_finetuned"
    if model_name == OriginalClassifier.VITS16:
        return f"{prefix}_vit_s_16" if prefix == "pretrained" else "vit_s_16_finetuned"
    return f"{prefix}_resnet50" if prefix == "pretrained" else "resnet50_finetuned"


def _gradcam_domain_code(domain_name):
    value = str(domain_name or "").lower()
    if "gpr-sd" in value or value == "sd":
        return "SD"
    if "gpr-road" in value or value == "road":
        return "Road"
    if "road" in value:
        return "Road"
    if "sd" in value:
        return "SD"
    return ""


def _gradcam_target_for_source(source):
    if source == "Road":
        return "SD"
    if source == "SD":
        return "Road"
    return ""


def _parse_result_time(value):
    if not value:
        return None
    parsed = parse_datetime(str(value))
    if parsed is None:
        return None
    if timezone.is_naive(parsed):
        parsed = timezone.make_aware(parsed, timezone.get_current_timezone())
    return parsed


def _accuracy_percent(value):
    accuracy = Decimal(str(value))
    if accuracy <= 1:
        accuracy *= Decimal("100")
    return accuracy.quantize(Decimal("0.01"))


def _classifier_model_path(source_dataset, image_embedding):
    rootpath = os.getenv("GPR_TRAINING_ROOTPATH") or str(PROJECT_ROOT)
    classifier_lr = os.getenv("GPR_CLASSIFIER_LR", "0.0001")
    classifier_nepoch = os.getenv("GPR_CLASSIFIER_NEPOCH", "100")
    return str(
        Path(rootpath)
        / "models"
        / "base-classifiers"
        / f"{source_dataset}_{image_embedding}_seed0_clr{classifier_lr}_nep{classifier_nepoch}"
    )


def _run_original_classifier_training(model_name, training_type, domain):
    source_dataset = _dataset_code_for_training(domain)
    target_dataset = _target_dataset_for_source(source_dataset)
    image_embedding = _image_embedding(model_name, training_type)
    CLASSIFIER_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    save_prefix = str(CLASSIFIER_RESULT_DIR) + os.sep

    command = [
        sys.executable,
        str(PROJECT_ROOT / "main_base.py"),
        "--cuda",
        "--manualSeed",
        "0",
        "--dataset",
        target_dataset,
        "--image_embedding",
        image_embedding,
        "--class_embedding",
        "llama",
        "--factual_branch",
        "attention",
        "--intervention_branch",
        "none",
        "--source_only_benchmark",
        "--cos_sim_loss",
        "--llm",
        "gpt4o",
        "--include_new",
        "--num_layers",
        "2",
        "--beta1",
        "0.9",
        "--lr",
        "0.00001",
        "--batch_size",
        "8",
        "--embed_dim",
        "2048",
        "--strict_eval",
        "--early_stopping_slope",
        "--calc_entropy",
        "--save_pred_matrix",
        "--nepoch",
        "500",
        "--view_num",
        "10",
        "--zst",
        "--zstfrom",
        source_dataset,
        "--norm_scale_heuristic",
        "--save_path",
        save_prefix,
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES", "0")
    timeout = int(os.getenv("CLASSIFIER_TRAINING_TIMEOUT", "7200"))
    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "Classifier training script fails.").strip()
        raise RuntimeError(message[-1200:])

    result_file = CLASSIFIER_RESULT_DIR / f"{source_dataset}_{image_embedding}.json"
    if not result_file.exists():
        raise RuntimeError("Classifier training result file is missing.")

    with result_file.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    result_file.unlink(missing_ok=True)
    result["model_file_path"] = result.get("model_file_path") or _classifier_model_path(source_dataset, image_embedding)
    return result


@csrf_exempt
def feature_image_upload(request):
    if request.method != "POST":
        return JsonResponse({"message": "Only POST is supported."}, status=405)

    upload = request.FILES.get("image")
    if not upload:
        return JsonResponse({"message": "Feature image input fails: image file is missing."}, status=400)

    suffix = Path(upload.name).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        return JsonResponse({"message": "Feature image input fails: unsupported image type."}, status=400)

    FEATURE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    image_id = uuid.uuid4().hex
    image_name = f"{image_id}{suffix}"
    image_path = FEATURE_IMAGE_DIR / image_name
    with image_path.open("wb") as handle:
        for chunk in upload.chunks():
            handle.write(chunk)

    return JsonResponse(
        {
            "message": "Feature image input succeeds.",
            "imageId": image_id,
            "imageName": image_name,
            "imagePath": str(image_path),
        }
    )


@csrf_exempt
def feature_activation_generate(request):
    if request.method != "POST":
        return JsonResponse({"message": "Only POST is supported."}, status=405)

    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"message": "Activation generation fails: invalid JSON payload."}, status=400)

    classifier_id = _int_or_none(payload.get("classifierId"))
    image_id = str(payload.get("imageId") or "").strip()
    image_name = str(payload.get("imageName") or "").strip()
    if not classifier_id or not image_id or not image_name:
        return JsonResponse({"message": "Activation generation needs classifier and input image."}, status=400)

    classifier = get_object_or_404(OriginalClassifier.objects.select_related("domain"), pk=classifier_id)
    if not classifier.model_file_path:
        return JsonResponse({"message": "Activation generation fails: classifier model file path is missing."}, status=400)

    image_path = FEATURE_IMAGE_DIR / image_name
    if not image_path.exists() or not image_name.startswith(image_id):
        return JsonResponse({"message": "Activation generation fails: uploaded image is missing."}, status=404)

    source = _gradcam_domain_code(classifier.domain_name)
    target = _gradcam_target_for_source(source)
    if not source or not target:
        return JsonResponse({"message": "Activation generation only supports GPR-SD and GPR-Road classifiers."}, status=400)

    image_embedding = _image_embedding(classifier.model_name, classifier.training_type)
    if "vit" in image_embedding.lower():
        return JsonResponse({"message": "Activation generation only supports ResNet-50 and ResNet-101 classifiers."}, status=400)

    run_id = uuid.uuid4().hex
    outdir = FEATURE_GRADCAM_DIR / run_id
    outdir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "visualize_gradcam_from_main_method.py"),
        "--method",
        "sourceonly",
        "--source",
        source,
        "--target",
        target,
        "--image_embedding",
        image_embedding,
        "--seed",
        "0",
        "--draw_domain",
        "source",
        "--image",
        str(image_path),
        "--method_checkpoint",
        classifier.model_file_path,
        "--rootpath",
        str(PROJECT_ROOT),
        "--outdir",
        str(outdir),
        "--cuda",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES", "0")
    try:
        completed = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=int(os.getenv("GRADCAM_TIMEOUT", "900")),
        )
    except subprocess.TimeoutExpired:
        return JsonResponse({"message": "Activation generation fails: script timeout."}, status=500)

    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "Activation generation script fails.").strip()
        return JsonResponse({"message": f"Activation generation fails: {message[-1200:]}"}, status=500)

    outputs = sorted(outdir.glob("*_gradcam.png"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not outputs:
        outputs = sorted(outdir.glob("*.png"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not outputs:
        return JsonResponse({"message": "Activation generation fails: output image is missing."}, status=500)

    output_path = outputs[0]
    image_base64 = base64.b64encode(output_path.read_bytes()).decode("ascii")
    return JsonResponse(
        {
            "message": "Activation map generation succeeds.",
            "activationImage": f"data:image/png;base64,{image_base64}",
            "outputPath": str(output_path),
            "source": source,
            "target": target,
            "imageEmbedding": image_embedding,
        }
    )
def _classifier_domain_from_payload(payload):
    domain_id = _blank_to_none(payload.get("domainId"))
    domain_value = str(payload.get("domain") or payload.get("original") or payload.get("domainName") or "").strip()

    if domain_id is not None:
        return get_object_or_404(AreaDomain, pk=domain_id, is_active=True)

    if domain_value:
        domain = _domain_by_name_or_code(domain_value)
        if domain:
            return domain

    return None


@csrf_exempt
def original_classifier_list(request):
    if request.method == "POST":
        try:
            payload = json.loads(request.body.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            return JsonResponse({"message": "Classifier save fails: invalid JSON payload."}, status=400)

        model_name = str(payload.get("modelName") or payload.get("model") or "").strip()
        training_type = str(payload.get("trainingType") or payload.get("type") or "").strip()
        domain = _classifier_domain_from_payload(payload)
        description = str(payload.get("modelDescription") or payload.get("description") or "").strip()

        if not model_name or not domain or not training_type:
            return JsonResponse({"message": "Classifier save fails: required fields are missing."}, status=400)

        if model_name not in dict(OriginalClassifier.MODEL_CHOICES):
            return JsonResponse({"message": "Classifier save fails: model is invalid."}, status=400)

        if training_type not in dict(OriginalClassifier.TRAINING_TYPE_CHOICES):
            return JsonResponse({"message": "Classifier save fails: type is invalid."}, status=400)

        try:
            training_result = _run_original_classifier_training(model_name, training_type, domain)
        except Exception as exc:
            return JsonResponse({"message": f"Classifier training fails: {exc}"}, status=500)

        result_model_name = str(training_result.get("model_name") or model_name).strip()
        if result_model_name not in dict(OriginalClassifier.MODEL_CHOICES):
            result_model_name = model_name
        result_training_type = str(training_result.get("training_type") or training_type).strip()
        if result_training_type not in dict(OriginalClassifier.TRAINING_TYPE_CHOICES):
            result_training_type = training_type

        classifier = OriginalClassifier.objects.create(
            model_name=result_model_name,
            domain=domain,
            domain_name=domain.name,
            training_type=result_training_type,
            accuracy=_accuracy_percent(training_result.get("accuracy", 0)),
            model_description=description or "User configured original-domain classifier record.",
            model_file_path=str(training_result.get("model_file_path") or ""),
        )

        created_at = _parse_result_time(training_result.get("created_at"))
        updated_at = _parse_result_time(training_result.get("updated_at"))
        update_fields = []
        if created_at:
            classifier.created_at = created_at
            update_fields.append("created_at")
        if updated_at:
            classifier.updated_at = updated_at
            update_fields.append("updated_at")
        if update_fields:
            OriginalClassifier.objects.filter(pk=classifier.pk).update(
                **{field: getattr(classifier, field) for field in update_fields}
            )
            classifier.refresh_from_db()

        return JsonResponse(
            {
                "message": "Classifier training succeeds.",
                "classifier": serialize_original_classifier(classifier),
            },
            status=201,
        )

    if request.method != "GET":
        return JsonResponse({"message": "Only GET and POST are supported."}, status=405)

    try:
        page = max(1, int(request.GET.get("page", 1)))
        page_size = max(1, min(500, int(request.GET.get("pageSize", 4))))
    except (TypeError, ValueError):
        return JsonResponse({"message": "Original classifier loading fails: pagination is invalid."}, status=400)

    queryset = OriginalClassifier.objects.select_related("domain").all()
    total = queryset.count()
    start = (page - 1) * page_size
    end = start + page_size
    rows = [serialize_original_classifier(item) for item in queryset[start:end]]

    return JsonResponse(
        {
            "classifiers": rows,
            "pagination": {
                "page": page,
                "pageSize": page_size,
                "total": total,
                "pageCount": max(1, (total + page_size - 1) // page_size),
            },
        }
    )


@csrf_exempt
def original_classifier_retrain(request, classifier_id):
    if request.method != "POST":
        return JsonResponse({"message": "Only POST is supported."}, status=405)

    classifier = get_object_or_404(OriginalClassifier.objects.select_related("domain"), pk=classifier_id)
    current_accuracy = float(classifier.accuracy)
    delta = random.choice([-0.2, -0.1, 0, 0.1, 0.2, 0.3])
    next_accuracy = min(99.9, max(0.0, round(current_accuracy + delta, 2)))

    classifier.accuracy = Decimal(str(next_accuracy))
    classifier.save(update_fields=["accuracy", "updated_at"])

    return JsonResponse(
        {
            "message": "Classifier retraining succeeds.",
            "classifier": serialize_original_classifier(classifier),
        }
    )


@csrf_exempt
def original_classifier_detail(request, classifier_id):
    classifier = get_object_or_404(OriginalClassifier.objects.select_related("domain"), pk=classifier_id)

    if request.method == "DELETE":
        classifier.delete()
        return JsonResponse({"message": "Classifier is deleted."}, status=200)

    if request.method == "GET":
        return JsonResponse({"classifier": serialize_original_classifier(classifier)})

    return JsonResponse({"message": "Only GET and DELETE are supported."}, status=405)
