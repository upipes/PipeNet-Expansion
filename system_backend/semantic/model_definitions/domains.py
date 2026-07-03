from django.db import models

class AreaDomain(models.Model):
    ROAD = "road"
    SOIL = "soil"
    DOMAIN_TYPE_CHOICES = ((ROAD, "Road"), (SOIL, "Soil"))

    code = models.CharField(max_length=64, unique=True)
    name = models.CharField(max_length=128)
    domain_type = models.CharField(max_length=16, choices=DOMAIN_TYPE_CHOICES)
    display_order = models.PositiveIntegerField(default=0)
    is_active = models.BooleanField(default=True)

    condition_text = models.CharField(max_length=255, blank=True)
    road_surface = models.CharField(max_length=255, blank=True)
    frequency_min = models.PositiveIntegerField(null=True, blank=True)
    frequency_max = models.PositiveIntegerField(null=True, blank=True)
    time_window_ns = models.PositiveIntegerField(null=True, blank=True)

    sand_percent = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    silt_percent = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    clay_percent = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    water_min = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    water_max = models.DecimalField(max_digits=5, decimal_places=2, null=True, blank=True)
    permittivity_min = models.DecimalField(max_digits=7, decimal_places=3, null=True, blank=True)
    permittivity_max = models.DecimalField(max_digits=7, decimal_places=3, null=True, blank=True)
    conductivity_min = models.DecimalField(max_digits=9, decimal_places=4, null=True, blank=True)
    conductivity_max = models.DecimalField(max_digits=9, decimal_places=4, null=True, blank=True)
    peplinski_dimension = models.DecimalField(max_digits=6, decimal_places=3, null=True, blank=True)

    area_description = models.TextField(blank=True)
    signal_behavior = models.TextField(blank=True)
    semantic_usage = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "area_domain"
        ordering = ["display_order", "id"]

    def __str__(self):
        return self.name

