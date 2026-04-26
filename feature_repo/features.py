from datetime import timedelta

from feast import (
    Entity,
    FeatureService,
    FeatureView,
    Field,
    ValueType,
)
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
    PostgreSQLSource,
)
from feast.types import Float32

iris_entity = Entity(
    name="flower_id",
    description="ID цветка ириса",
    value_type=ValueType.INT64,
)

iris_source = PostgreSQLSource(
    name="iris_source",
    query="SELECT * FROM iris_data",
    timestamp_field="event_timestamp",
)

iris_features = FeatureView(
    name="iris_features",
    entities=[iris_entity],
    ttl=timedelta(days=1),
    schema=[
        Field(name="sepal length (cm)", dtype=Float32),
        Field(name="sepal width (cm)", dtype=Float32),
        Field(name="petal length (cm)", dtype=Float32),
        Field(name="petal width (cm)", dtype=Float32),
    ],
    source=iris_source,
)

iris_service = FeatureService(
    name="iris_service",
    features=[iris_features],
)
