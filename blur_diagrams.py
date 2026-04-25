from diagrams import Cluster, Diagram
from diagrams.aws.storage import S3
from diagrams.onprem.compute import Server
from diagrams.onprem.database import PostgreSQL
from diagrams.onprem.monitoring import Grafana, Prometheus
from diagrams.onprem.workflow import Airflow
from diagrams.programming.language import Python

with Diagram(
    "Схема ML-системы для размытия (заблюривания) лиц на изображениях",
    show=False,
    direction="LR",
    filename="ml_blur",
):
    
    with Cluster("Сбор и хранение информации"):
        upload_api = Server("API\nзагрузки")
        video_processor = Server("Обработчик\nвидео")
    
    with Cluster("Получение и хранение данных"):
        raw_data = S3("Сырые\nвидео")
        processed_data = S3("Обработанные\nкадры")
        dvc_remote = S3("DVC Remote\n(версионирование)")
    
    with Cluster("Эксперименты и версионирование"):
        mlflow = Server("MLflow")
        model_registry = Airflow("Реестр\nмоделей")
        metrics_store = PostgreSQL("Хранилище\nметрик")
    
    with Cluster("Обучение модели"):
        feature_extractor = Python("Извлечение\nпризнаков")
        trainer = Server("Тренировка\nмодели")
        dvc_pipeline = Server("DVC\nпайплайн")
    
    with Cluster("Инференс"):
        inference_api = Server("Inference API")
        with Cluster("Воркеры"):
            workers = [
                Server("..."),
                Server("Воркер N"),
                Server("Воркер 1"),
            ]
        output = S3("Видео с размытием\n(результат)")
    
    with Cluster("Мониторинг"):
        prometheus = Prometheus("Prometheus")
        grafana = Grafana("Grafana")
    
    upload_api >> raw_data
    raw_data >> dvc_pipeline
    dvc_pipeline >> dvc_remote
    dvc_pipeline >> processed_data
    
    processed_data >> feature_extractor >> trainer
    mlflow >> trainer
    trainer >> model_registry
    trainer >> metrics_store
    
    model_registry >> inference_api
    inference_api >> workers
    workers >> output
    workers >> prometheus
    prometheus >> grafana
