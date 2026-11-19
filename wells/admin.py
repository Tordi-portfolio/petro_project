from django.contrib import admin

from .models import Analysis, Well


@admin.register(Well)
class WellAdmin(admin.ModelAdmin):
    list_display = ("name", "top", "base", "depth_unit", "uploaded_at")


@admin.register(Analysis)
class AnalysisAdmin(admin.ModelAdmin):
    list_display = ("well", "created_at", "suitable", "score")
    exclude = ("images",)
