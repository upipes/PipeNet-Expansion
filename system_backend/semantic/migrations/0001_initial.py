from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="AreaDomain",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("code", models.CharField(max_length=64, unique=True)),
                ("name", models.CharField(max_length=128)),
                ("domain_type", models.CharField(choices=[("road", "Road"), ("soil", "Soil")], max_length=16)),
                ("display_order", models.PositiveIntegerField(default=0)),
                ("is_active", models.BooleanField(default=True)),
                ("condition_text", models.CharField(blank=True, max_length=255)),
                ("road_surface", models.CharField(blank=True, max_length=255)),
                ("frequency_min", models.PositiveIntegerField(blank=True, null=True)),
                ("frequency_max", models.PositiveIntegerField(blank=True, null=True)),
                ("time_window_ns", models.PositiveIntegerField(blank=True, null=True)),
                ("sand_percent", models.DecimalField(blank=True, decimal_places=2, max_digits=5, null=True)),
                ("silt_percent", models.DecimalField(blank=True, decimal_places=2, max_digits=5, null=True)),
                ("clay_percent", models.DecimalField(blank=True, decimal_places=2, max_digits=5, null=True)),
                ("water_min", models.DecimalField(blank=True, decimal_places=2, max_digits=5, null=True)),
                ("water_max", models.DecimalField(blank=True, decimal_places=2, max_digits=5, null=True)),
                ("permittivity_min", models.DecimalField(blank=True, decimal_places=3, max_digits=7, null=True)),
                ("permittivity_max", models.DecimalField(blank=True, decimal_places=3, max_digits=7, null=True)),
                ("conductivity_min", models.DecimalField(blank=True, decimal_places=4, max_digits=9, null=True)),
                ("conductivity_max", models.DecimalField(blank=True, decimal_places=4, max_digits=9, null=True)),
                ("peplinski_dimension", models.DecimalField(blank=True, decimal_places=3, max_digits=6, null=True)),
                ("area_description", models.TextField(blank=True)),
                ("signal_behavior", models.TextField(blank=True)),
                ("semantic_usage", models.TextField(blank=True)),
                ("uploaded_json", models.JSONField(blank=True, null=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={"db_table": "area_domain", "ordering": ["display_order", "id"]},
        ),
    ]
