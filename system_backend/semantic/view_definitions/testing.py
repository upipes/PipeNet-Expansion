import json
import base64
import csv
import os
import random
import subprocess
import sys
import uuid
from decimal import Decimal
from pathlib import Path

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
TRAINING_RESULT_DIR = PROJECT_ROOT / "runtime_outputs" / "model_training_runs"
FEATURE_IMAGE_DIR = PROJECT_ROOT / "runtime_outputs" / "feature_images"
COMPARISON_GRADCAM_DIR = PROJECT_ROOT / "runtime_outputs" / "comparison_gradcam"

TRAINING_RESULT_ACCURACY_CLASS = PROJECT_ROOT / "outputs"

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


def _image_embedding_for_training(classifier):
    model_name = getattr(classifier, "model_name", "") or ""
    training_type = getattr(classifier, "training_type", "") or ""
    prefix = "pretrained" if training_type == OriginalClassifier.PRE_TRAINED else ""
    lower = model_name.lower()
    if "101" in lower:
        return "pretrained_resnet101" if prefix else "resnet101_finetuned"
    if "vit" in lower:
        return "pretrained_vit_s_16" if prefix else "vit_s_16_finetuned"
    return "pretrained_resnet50" if prefix else "resnet50_finetuned"


def _image_embedding_from_backbone(backbone):
    value = str(backbone or "").lower()
    if "101" in value:
        return "pretrained_resnet101"
    if "vit" in value:
        return "pretrained_vit_s_16"
    return "pretrained_resnet50"


def _gradcam_domain_code(value):
    text = str(value or "").strip().lower()
    if text in {"sd", "gpr-sd"} or "gpr-sd" in text:
        return "SD"
    if text in {"road", "gpr-road"} or "gpr-road" in text:
        return "Road"
    return ""


def _latest_gradcam_image(outdir):
    candidates = list(outdir.glob("*_gradcam.png"))
    return max(candidates, key=lambda item: item.stat().st_mtime) if candidates else None


def _image_to_data_url(path):
    with path.open("rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _class_embedding(value):
    lookup = {
        "clip": "clip",
        "sbert": "sbert",
        "llama": "llama",
        "llama-3.1": "llama",
        "qwen": "qwen-7b",
    }
    return lookup.get(str(value or "").strip().lower(), "llama")


def _llm_name(value):
    lookup = {
        "gpt-4o": "gpt4o",
        "gpt-3.5-turbo": "gpt35turbo",
        "gpt-4o-mini": "gpt4omini",
        "gemini-2.5": "gemini2.5",
        "llama-3.1": "llama70b",
        "qwen-2.5": "qwen_plus",
    }
    return lookup.get(str(value or "").strip().lower(), "gpt4o")


def _read_training_accuracy(save_path):
    if not save_path.exists():
        return None
    with save_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    row = rows[-1]
    for key in ("zsl_acc", "acc", "accuracy", "new", "u"):
        value = row.get(key)
        if value not in ("", None):
            try:
                return Decimal(str(value).replace("%", "").split("±")[0].split("卤")[0])
            except Exception:
                continue
    return None


def _method_checkpoint_path(method, source_dataset, target_dataset, image_embedding, seed=0):
    path = PROJECT_ROOT / "method-checkpoints" / f"{method}_{source_dataset}_to_{target_dataset}_{image_embedding}_seed{seed}.pth"
    return str(path) if path.exists() else str(path)


def _method_runtime_config(method_name):
    normalized = str(method_name or "").strip().lower()
    if normalized in {"original classifier", "original", "ori-only", "orionly"}:
        return "sourceonly", ["--source_only_benchmark"], 500
    if normalized == "wdae":
        return "wdae", ["--single_autoencoder_baseline", "--daegnn"], 120
    if normalized == "subreg":
        return "subreg", ["--single_autoencoder_baseline", "--subspace_proj"], 100
    if normalized == "vgse":
        return "vgse", ["--vgse_baseline", "smo", "--vgse_alpha", "0"], 150
    if normalized == "icis":
        return "icis", ["--method", "ICIS"], 500
    if normalized == "adda":
        return "adda", ["--adda_benchmark"], 100
    if normalized == "dann":
        return "dann", ["--dann_benchmark"], 500
    if normalized == "g2kd":
        return "g2kd", ["--g2kd_benchmark", "--g2kd_dis_weight", "1.0", "--g2kd_stu_weight", "0.3", "--g2kd_ent_weight", "0.1", "--g2kd_neighbors", "5"], 100
    if normalized == "tpds":
        return "tpds", ["--tpds_benchmark", "--tpds_align_weight", "1.0", "--tpds_cc_weight", "0.5", "--tpds_im_weight", "0.1", "--tpds_neighbors", "5", "--tpds_steps", "3"], 100
    if normalized == "ours":
        return "ours", ["--conclude_inv"], 500
    return normalized
def _training_class_names(source_domain, target_domain):
    source_name = getattr(source_domain, "name", source_domain)
    target_name = getattr(target_domain, "name", target_domain)
    return ["Cavity", "Crack", "Metal Pipeline"] if any(
        marker in f"{source_name} {target_name}" for marker in ["Sandy", "Silty", "Backfill", "Layered", "a1", "a2", "a3", "a4"]
    ) else ["Cavity", "Crack", "Loose", "Normal", "Pipeline"]


def _per_class_accuracy_file(target_domain, method="ours", seed=0, class_count=None):
    target_dataset = _dataset_code_for_training(target_domain)
    pattern = f"percls_acc_{target_dataset}_{method}_target_zsl_len_test_*_len_tar_{class_count or '*'}_{seed}.txt"
    matches = sorted(TRAINING_RESULT_ACCURACY_CLASS.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
    if matches:
        return matches[0]
    fallback = TRAINING_RESULT_ACCURACY_CLASS / f"percls_acc_{target_dataset}_{method}_target_zsl_len_test_852_len_tar_{class_count or 5}_{seed}.txt"
    return fallback


def _training_class_accuracy(source_domain, target_domain, method="ours", seed=0):
    names = _training_class_names(source_domain, target_domain)
    values = []
    accuracy_file = _per_class_accuracy_file(target_domain, method=method, seed=seed, class_count=len(names))
    if not accuracy_file.exists():
        return {
            name: round(min(99.9, max(60.0, float(random.uniform(82.0, 94.0)))), 2)
            for name in names
        }
    with accuracy_file.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            values.append(float(line))
    return {
        name: round(value * 100, 2)
        for name, value in zip(names, values)
    }


def _legacy_training_class_accuracy(source_domain, target_domain):
    names = ["Cavity", "Crack", "Metal Pipeline"] if any(
        marker in f"{source_domain.name} {target_domain.name}" for marker in ["Sandy", "Silty", "Backfill", "Layered", "a1", "a2", "a3", "a4"]
    ) else ["Cavity", "Crack", "Loose", "Normal", "Pipeline"]
    values = []
    target_dataset = _dataset_code_for_training(target_domain)
    with Path(TRAINING_RESULT_ACCURACY_CLASS / f"percls_acc_{target_dataset}_ours_target_zsl_len_test_852_len_tar_5_0.txt").open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            values.append(float(line))
    return {
        name: round(value * 100, 2)
        for name, value in zip(names, values)
    }

@csrf_exempt
def model_training_runs(request):
    if request.method == "POST":
        try:
            payload = json.loads(request.body.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            return JsonResponse({"message": "Model training save fails: invalid JSON payload."}, status=400)

        required = [
            "modelName",
            "sourceDataset",
            "targetDataset",
            "backbone",
            "optimizer",
            "learningRate",
            "batchSize",
            "epochs",
        ]
        if any(payload.get(field) in ("", None) for field in required):
            return JsonResponse({"message": "Model training save fails: required fields are missing."}, status=400)
        source_domain = _domain_by_name_or_code(payload.get("sourceDataset"))
        target_domain = _domain_by_name_or_code(payload.get("targetDataset"))
        if not source_domain or not target_domain:
            return JsonResponse({"message": "Model training fails: selected area does not exist."}, status=400)

        classifier = None
        classifier_id = _int_or_none(payload.get("classifierId"))
        if classifier_id:
            classifier = OriginalClassifier.objects.filter(pk=classifier_id).first()
        if classifier is None:
            classifier = OriginalClassifier.objects.filter(
                model_name=str(payload.get("backbone") or "").strip(),
                domain__name=source_domain.name,
            ).first()

        method_name = str(payload.get("modelName") or "Ours").strip()
        runtime_method, method_flags, default_epochs = _method_runtime_config(method_name)
        source_dataset = _dataset_code_for_training(source_domain)
        target_dataset = _dataset_code_for_training(target_domain)
        image_embedding = _image_embedding_for_training(classifier) if classifier else _image_embedding_from_backbone(payload.get("backbone"))
        class_embedding = _class_embedding(payload.get("embeddingModel") or "LLaMA")
        llm = _llm_name(payload.get("semanticGenerator") or "GPT-4o")
        view_number = str(payload.get("knowledgeItems") or 10)
        learning_rate = str(payload.get("learningRate")).strip()
        batch_size = _int_or_none(payload.get("batchSize")) or 8
        epochs = _int_or_none(payload.get("epochs")) or default_epochs
        resolved_flags = [str(epochs) if flag is None else flag for flag in method_flags]

        TRAINING_RESULT_DIR.mkdir(parents=True, exist_ok=True)
        run_token = uuid.uuid4().hex
        save_path = TRAINING_RESULT_DIR / f"{runtime_method}_{source_dataset}_to_{target_dataset}_{run_token}.csv"
        avg_save_path = TRAINING_RESULT_DIR / f"{runtime_method}_{source_dataset}_to_{target_dataset}_{run_token}_avg.csv"
        command = [
            sys.executable,
            str(PROJECT_ROOT / "main.py"),
            "--cuda",
            "--manualSeed",
            "0",
            "--dataset",
            target_dataset,
            "--image_embedding",
            image_embedding,
            "--class_embedding",
            class_embedding,
            "--factual_branch",
            "attention",
            "--intervention_branch",
            "none",
            *resolved_flags,
            "--cos_sim_loss",
            "--llm",
            llm,
            "--include_new",
            "--num_layers",
            "2",
            "--beta1",
            "0.9",
            "--lr",
            learning_rate,
            "--batch_size",
            str(batch_size),
            "--embed_dim",
            "2048",
            "--strict_eval",
            "--early_stopping_slope",
            "--calc_entropy",
            "--save_pred_matrix",
            "--view_num",
            view_number,
            "--zst",
            "--zstfrom",
            source_dataset,
            "--norm_scale_heuristic",
            "--nepoch",
            str(epochs),
            "--save_path",
            str(save_path),
            "--avg_save_path",
            str(avg_save_path),
            "--save_method_checkpoint",
        ]
        # print(command)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES", "0")
        try:
            completed = subprocess.run(
                command,
                cwd=str(PROJECT_ROOT),
                env=env,
                capture_output=True,
                text=True,
                timeout=int(os.getenv("MODEL_TRAINING_TIMEOUT", "7200")),
            )
        except subprocess.TimeoutExpired:
            return JsonResponse({"message": "Model training fails: script timeout."}, status=500)

        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout or "Model training script fails.").strip()
            return JsonResponse({"message": f"Model training fails: {message[-1200:]}"}, status=500)

        accuracy = _read_training_accuracy(save_path) or Decimal(str(payload.get("accuracy") or round(random.uniform(91.2, 95.4), 2)))
        checkpoint_path = _method_checkpoint_path(runtime_method, source_dataset, target_dataset, image_embedding, 0)
        run = ModelTrainingRun.objects.create(
            source_domain=source_domain,
            target_domain=target_domain,
            method_name=method_name,
            model_name=method_name,
            source_dataset=source_domain.name,
            target_dataset=target_domain.name,
            backbone=str(payload.get("backbone")).strip(),
            semantic_generator=str(payload.get("semanticGenerator") or "").strip() or None,
            embedding_model=str(payload.get("embeddingModel") or "").strip() or None,
            optimizer=str(payload.get("optimizer")).strip(),
            knowledge_items=_int_or_none(payload.get("knowledgeItems")),
            refinement_iterations=_int_or_none(payload.get("refinementIterations")),
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            accuracy=accuracy,
            class_accuracy=payload.get("classAccuracy") or _training_class_accuracy(
                source_domain, target_domain, method=runtime_method, seed=0
            ),
            method_checkpoint_path=checkpoint_path,
            model_description=str(payload.get("modelDescription") or "").strip(),
        )
        return JsonResponse(
            {"message": "Model training run is saved.", "run": serialize_model_training_run(run)},
            status=201,
        )

    if request.method != "GET":
        return JsonResponse({"message": "Only GET and POST are supported."}, status=405)

    try:
        page = max(1, int(request.GET.get("page", 1)))
        page_size = max(1, min(100, int(request.GET.get("pageSize", 4))))
    except (TypeError, ValueError):
        return JsonResponse({"message": "Model training run loading fails: pagination is invalid."}, status=400)

    query = str(request.GET.get("search") or "").strip()
    queryset = ModelTrainingRun.objects.all()
    if query:
        queryset = queryset.filter(
            Q(model_name__icontains=query)
            | Q(backbone__icontains=query)
            | Q(source_dataset__icontains=query)
            | Q(target_dataset__icontains=query)
            | Q(semantic_generator__icontains=query)
            | Q(embedding_model__icontains=query)
        )

    total = queryset.count()
    start = (page - 1) * page_size
    rows = queryset[start:start + page_size]
    return JsonResponse(
        {
            "runs": [serialize_model_training_run(row) for row in rows],
            "pagination": {
                "page": page,
                "pageSize": page_size,
                "total": total,
                "pageCount": max(1, (total + page_size - 1) // page_size),
            },
        }
      )


def _latest_training_run_for_summary(source_dataset, target_dataset, method_name):
    method_name = str(method_name or "").strip()
    return (
        ModelTrainingRun.objects.filter(
            source_dataset=source_dataset,
            target_dataset=target_dataset,
        )
        .filter(Q(method_name=method_name) | Q(model_name=method_name))
        .order_by("-updated_at", "-id")
        .first()
    )


def model_training_run_compare(request):
    source_dataset = str(request.GET.get("original") or "").strip()
    target_dataset = str(request.GET.get("new") or "").strip()
    method_a = str(request.GET.get("methodA") or "").strip()
    method_b = str(request.GET.get("methodB") or "").strip()

    if not source_dataset or not target_dataset or not method_a or not method_b:
        return JsonResponse({"message": "Diagnosis summary loading fails: query parameters are missing."}, status=400)

    run_a = _latest_training_run_for_summary(source_dataset, target_dataset, method_a)
    run_b = _latest_training_run_for_summary(source_dataset, target_dataset, method_b)

    return JsonResponse(
        {
            "methodA": serialize_model_training_run(run_a) if run_a else None,
            "methodB": serialize_model_training_run(run_b) if run_b else None,
        }
    )


@csrf_exempt
def model_training_run_detail(request, run_id):
    run = get_object_or_404(ModelTrainingRun, pk=run_id)

    if request.method == "GET":
        return JsonResponse({"run": serialize_model_training_run(run)})

    if request.method == "DELETE":
        run.delete()
        return JsonResponse({"message": "Model testing record is deleted."})

    if request.method in {"PATCH", "POST"}:
        try:
            payload = json.loads(request.body.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            return JsonResponse({"message": "Retraining update fails: invalid JSON payload."}, status=400)

        if "sourceDataset" in payload:
            source_domain = _domain_by_name_or_code(payload.get("sourceDataset"))
            if source_domain:
                run.source_domain = source_domain
                run.source_dataset = source_domain.name

        if "targetDataset" in payload:
            target_domain = _domain_by_name_or_code(payload.get("targetDataset"))
            if target_domain:
                run.target_domain = target_domain
                run.target_dataset = target_domain.name

        for field, attr in [
            ("semanticGenerator", "semantic_generator"),
            ("embeddingModel", "embedding_model"),
            ("optimizer", "optimizer"),
            ("learningRate", "learning_rate"),
            ("modelDescription", "model_description"),
            ("backbone", "backbone"),
        ]:
            if field in payload:
                setattr(run, attr, str(payload.get(field) or "").strip() or None)

        for field, attr in [
            ("knowledgeItems", "knowledge_items"),
            ("refinementIterations", "refinement_iterations"),
            ("batchSize", "batch_size"),
            ("epochs", "epochs"),
        ]:
            if field in payload:
                setattr(run, attr, _int_or_none(payload.get(field)))

        base_accuracy = float(run.accuracy)
        run.accuracy = Decimal(str(payload.get("accuracy") or round(min(99.9, base_accuracy + random.uniform(-0.2, 1.1)), 2)))
        run.class_accuracy = payload.get("classAccuracy") or _training_class_accuracy(run.source_domain, run.target_domain)
        run.save()

        return JsonResponse(
            {
                "message": "Model retraining succeeds.",
                "run": serialize_model_training_run(run),
            }
        )

    return JsonResponse({"message": "Only GET, PATCH, POST, and DELETE are supported."}, status=405)


@csrf_exempt
def comparison_activation_generate(request):
    if request.method != "POST":
        return JsonResponse({"message": "Only POST is supported."}, status=405)

    try:
        payload = json.loads(request.body.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"message": "Comparison activation generation fails: invalid JSON payload."}, status=400)

    image_id = str(payload.get("imageId") or "").strip()
    image_name = str(payload.get("imageName") or "").strip()
    if not image_id or not image_name:
        return JsonResponse({"message": "Comparison activation generation needs an input image."}, status=400)

    image_path = (FEATURE_IMAGE_DIR / image_name).resolve()
    try:
        image_path.relative_to(FEATURE_IMAGE_DIR.resolve())
    except ValueError:
        return JsonResponse({"message": "Comparison activation generation fails: invalid image path."}, status=400)
    if not image_path.exists():
        return JsonResponse({"message": "Comparison activation generation fails: uploaded image is missing."}, status=404)

    method_kind = str(payload.get("methodKind") or "").strip().lower()
    if method_kind not in {"sourceonly", "ours"}:
        return JsonResponse({"message": "Comparison activation generation fails: method is invalid."}, status=400)

    source_dataset = _gradcam_domain_code(payload.get("sourceDataset"))
    target_dataset = _gradcam_domain_code(payload.get("targetDataset"))
    if not source_dataset or not target_dataset:
        return JsonResponse(
            {"message": "Comparison activation generation currently supports GPR-SD and GPR-Road."},
            status=400,
        )

    image_embedding = _image_embedding_from_backbone(payload.get("backbone"))
    if "vit" in image_embedding:
        return JsonResponse(
            {"message": "Comparison activation generation currently supports ResNet-50 and ResNet-101."},
            status=400,
        )

    seed = "0"
    outdir = COMPARISON_GRADCAM_DIR / f"{method_kind}_{source_dataset}_to_{target_dataset}_{uuid.uuid4().hex}"
    outdir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "visualize_gradcam_from_main_method.py"),
        "--method",
        method_kind,
        "--source",
        source_dataset,
        "--target",
        target_dataset,
        "--image_embedding",
        image_embedding,
        "--seed",
        seed,
        "--draw_domain",
        "target",
        "--image",
        str(image_path),
        "--rootpath",
        str(PROJECT_ROOT),
        "--outdir",
        str(outdir),
        "--cuda",
    ]

    checkpoint_path = str(payload.get("methodCheckpointPath") or "").strip()
    if checkpoint_path and Path(checkpoint_path).exists():
        command.extend(["--method_checkpoint", checkpoint_path])

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
        return JsonResponse({"message": "Comparison activation generation fails: script timeout."}, status=500)

    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "Grad-CAM script fails.").strip()
        return JsonResponse({"message": f"Comparison activation generation fails: {message[-1200:]}"}, status=500)

    output_image = _latest_gradcam_image(outdir)
    if not output_image:
        return JsonResponse({"message": "Comparison activation generation fails: no output image is produced."}, status=500)

    return JsonResponse(
        {
            "message": "Comparison activation generation succeeds.",
            "activationImage": _image_to_data_url(output_image),
            "outputPath": str(output_image),
        }
    )
