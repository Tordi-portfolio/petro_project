from django.db import models
from django.urls import reverse


class Well(models.Model):
    name = models.CharField(max_length=200)
    las_file = models.FileField(upload_to="las/")
    uploaded_at = models.DateTimeField(auto_now_add=True)
    curve_names = models.JSONField(default=list, blank=True)
    top = models.FloatField(null=True, blank=True)
    base = models.FloatField(null=True, blank=True)
    depth_unit = models.CharField(max_length=4, default="m")

    class Meta:
        ordering = ["-uploaded_at"]

    def __str__(self):
        return self.name

    def get_absolute_url(self):
        return reverse("well_detail", args=[self.pk])

    @property
    def latest_analysis(self):
        return self.analyses.first()


class Analysis(models.Model):
    well = models.ForeignKey(Well, related_name="analyses", on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    params = models.JSONField(default=dict, blank=True)     # parameters as submitted by the user
    summary = models.JSONField(default=dict)                # full evaluation summary (see petrophysics.pipeline)
    images = models.JSONField(default=dict, blank=True)     # {"log_plot": base64 png, "crossplots": base64 png}
    suitable = models.BooleanField(default=False)
    score = models.IntegerField(default=0)

    class Meta:
        ordering = ["-created_at"]
        verbose_name_plural = "analyses"

    def __str__(self):
        return f"{self.well.name} @ {self.created_at:%Y-%m-%d %H:%M}"

    def get_absolute_url(self):
        return reverse("analysis_detail", args=[self.pk])
