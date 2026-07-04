from django.db import models

from .domains import AreaDomain


class SemanticCategory(models.Model):
    name = models.CharField(max_length=128)
    domain = models.ForeignKey(AreaDomain, on_delete=models.CASCADE, related_name="semantic_categories")

    class Meta:
        db_table = "semantic_category"
        ordering = ["domain_id", "id"]
        constraints = [
            models.UniqueConstraint(fields=["domain", "name"], name="uk_semantic_category_domain_name"),
        ]

    def __str__(self):
        return self.name


class SemanticGenerationRun(models.Model):
    domain = models.ForeignKey(AreaDomain, on_delete=models.CASCADE, related_name="semantic_runs")
    llm_name = models.CharField(max_length=64)
    use_expert_knowledge = models.BooleanField(default=False)
    use_image_assist = models.BooleanField(default=False)
    generated_count = models.PositiveIntegerField(default=0)
    status = models.CharField(max_length=24, default="success")
    generated_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "semantic_generation_run"
        ordering = ["-generated_at", "-id"]

    def __str__(self):
        return f"{self.domain.code} / {self.llm_name} / {self.generated_at:%Y-%m-%d %H:%M:%S}"


class SemanticDescription(models.Model):
    run = models.ForeignKey(SemanticGenerationRun, on_delete=models.CASCADE, related_name="descriptions")
    domain = models.ForeignKey(AreaDomain, on_delete=models.CASCADE, related_name="semantic_descriptions")
    category = models.ForeignKey(SemanticCategory, on_delete=models.CASCADE, related_name="descriptions")

    primary_view = models.CharField(max_length=128)
    primary_brief_description = models.CharField(max_length=500)
    all_view_brief_descriptions = models.JSONField(default=dict)
    all_view_detailed_descriptions = models.JSONField(default=dict)
    llm_confidence = models.DecimalField(max_digits=5, decimal_places=2)
    generated_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "semantic_description"
        ordering = ["category_id", "id"]
        indexes = [
            models.Index(fields=["domain", "category"]),
            models.Index(fields=["run", "category"]),
        ]

    def __str__(self):
        return f"{self.domain.code} / {self.category.name} / {self.primary_view}"


class SemanticAnnotation(models.Model):
    CORRECT = "correct"
    INACCURATE = "inaccurate"
    INCORRECT = "incorrect"
    EFFECT_CHOICES = ((CORRECT, "Correct"), (INACCURATE, "Inaccurate"), (INCORRECT, "Incorrect"))

    description = models.ForeignKey(
        SemanticDescription,
        db_column="desc_id",
        on_delete=models.CASCADE,
        related_name="annotations",
    )
    view_name = models.CharField(max_length=128)
    view_text = models.TextField()
    annotated_text = models.CharField(max_length=1000)
    annotation_effect = models.CharField(max_length=24, choices=EFFECT_CHOICES)
    annotation_content = models.TextField(blank=True, null=True)
    update_revise = models.TextField(blank=True, null=True)
    annotated_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "semantic_annotation"
        ordering = ["annotated_at", "id"]
        indexes = [
            models.Index(fields=["description"]),
            models.Index(fields=["description", "view_name"]),
            models.Index(fields=["annotation_effect"]),
            models.Index(fields=["annotated_at"]),
        ]

    def __str__(self):
        return f"{self.description_id} / {self.view_name} / {self.annotation_effect}"

