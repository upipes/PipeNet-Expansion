from django.db import models

from .domains import AreaDomain


class OriginalClassifier(models.Model):
    RESNET50 = "ResNet-50"
    RESNET101 = "ResNet-101"
    VITS16 = "ViT-S/16"
    FINE_TUNED = "fine_tuned"
    PRE_TRAINED = "pre_trained"

    MODEL_CHOICES = (
        (RESNET50, "ResNet-50"),
        (RESNET101, "ResNet-101"),
        (VITS16, "ViT-S/16"),
    )
    TRAINING_TYPE_CHOICES = (
        (FINE_TUNED, "Fine-tuned"),
        (PRE_TRAINED, "Pre-trained"),
    )

    model_name = models.CharField(max_length=64, choices=MODEL_CHOICES)
    domain = models.ForeignKey(
        AreaDomain,
        db_column="domain_id",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="original_classifiers",
    )
    domain_name = models.CharField(max_length=128)
    training_type = models.CharField(max_length=24, choices=TRAINING_TYPE_CHOICES)
    accuracy = models.DecimalField(max_digits=5, decimal_places=2)
    model_description = models.TextField()
    model_file_path = models.CharField(max_length=512, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "original_classifier"
        ordering = ["id"]
        indexes = [
            models.Index(fields=["model_name"]),
            models.Index(fields=["domain"]),
            models.Index(fields=["domain_name"]),
            models.Index(fields=["training_type"]),
            models.Index(fields=["accuracy"]),
        ]

    def __str__(self):
        return f"{self.model_name} / {self.domain_name} / {self.training_type}"

