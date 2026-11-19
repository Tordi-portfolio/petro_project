from django.urls import path

from . import views

urlpatterns = [
    path("", views.well_list, name="well_list"),
    path("screen/", views.screen_all, name="screen_all"),
    path("wells/<int:pk>/", views.well_detail, name="well_detail"),
    path("wells/<int:pk>/delete/", views.well_delete, name="well_delete"),
    path("analyses/<int:pk>/", views.analysis_detail, name="analysis_detail"),
    path("analyses/<int:pk>/download/<str:fmt>/", views.analysis_download, name="analysis_download"),
]
