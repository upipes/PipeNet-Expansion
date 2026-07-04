from django.contrib import admin

from .models import AreaDomain


@admin.register(AreaDomain)
class AreaDomainAdmin(admin.ModelAdmin):
    list_display = ("code", "name", "domain_type", "display_order", "is_active", "updated_at")
    list_filter = ("domain_type", "is_active")
    search_fields = ("code", "name", "condition_text", "road_surface")
