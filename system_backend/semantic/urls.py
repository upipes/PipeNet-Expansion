from django.urls import path

from . import views


urlpatterns = [
    path("domains/", views.domain_list, name="domain-list"),
    path("domains/<int:domain_id>/", views.domain_detail, name="domain-detail"),
    path("domains/code/<str:code>/", views.domain_detail_by_code, name="domain-detail-by-code"),
    path("original-classifiers/", views.original_classifier_list, name="original-classifier-list"),
    path(
        "original-classifiers/<int:classifier_id>/",
        views.original_classifier_detail,
        name="original-classifier-detail",
    ),
    path(
        "original-classifiers/<int:classifier_id>/retrain/",
        views.original_classifier_retrain,
        name="original-classifier-retrain",
    ),
    path("feature-image-input/", views.feature_image_upload, name="feature-image-input"),
    path("feature-activation-map/", views.feature_activation_generate, name="feature-activation-map"),
    path("comparison-activation-map/", views.comparison_activation_generate, name="comparison-activation-map"),
    path("model-training-runs/", views.model_training_runs, name="model-training-runs"),
    path("model-training-runs/compare/", views.model_training_run_compare, name="model-training-run-compare"),
    path("model-training-runs/<int:run_id>/", views.model_training_run_detail, name="model-training-run-detail"),
    path("semantic-generation/", views.semantic_generation, name="semantic-generation"),
    path("semantic-generation/latest/", views.latest_semantic_generation, name="latest-semantic-generation"),
    path("semantic-annotations/", views.semantic_annotations, name="semantic-annotations"),
]
