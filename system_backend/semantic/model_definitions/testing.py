from django.db import models


class ModelTrainingRun(models.Model):
    source_domain = models.ForeignKey(
        "AreaDomain",
        on_delete=models.CASCADE,
        related_name="source_training_runs",
    )
    target_domain = models.ForeignKey(
        "AreaDomain",
        on_delete=models.CASCADE,
        related_name="target_training_runs",
    )
    method_name = models.CharField(max_length=64, default="Ours")
    model_name = models.CharField(max_length=64)
    source_dataset = models.CharField(max_length=128)
    target_dataset = models.CharField(max_length=128)
    backbone = models.CharField(max_length=64)
    semantic_generator = models.CharField(max_length=64, blank=True, null=True)
    embedding_model = models.CharField(max_length=64, blank=True, null=True)
    optimizer = models.CharField(max_length=64)
    knowledge_items = models.PositiveIntegerField(blank=True, null=True)
    refinement_iterations = models.PositiveIntegerField(blank=True, null=True)
    learning_rate = models.CharField(max_length=32)
    batch_size = models.PositiveIntegerField()
    epochs = models.PositiveIntegerField()
    accuracy = models.DecimalField(max_digits=5, decimal_places=2)
    class_accuracy = models.JSONField(default=dict)
    method_checkpoint_path = models.CharField(max_length=512, blank=True, null=True)
    model_description = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "model_training_run"
        ordering = ["-created_at", "-id"]
        indexes = [
            models.Index(fields=["method_name"]),
            models.Index(fields=["model_name"]),
            models.Index(fields=["source_dataset"]),
            models.Index(fields=["target_dataset"]),
            models.Index(fields=["backbone"]),
            models.Index(fields=["accuracy"]),
        ]

    def __str__(self):
        return f"{self.method_name} / {self.model_name} / {self.source_dataset} -> {self.target_dataset}"

