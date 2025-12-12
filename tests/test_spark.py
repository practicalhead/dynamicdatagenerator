"""
Tests for the Spark module.

Tests Spark-based data generation and training.
Note: These tests require PySpark to be installed.
"""

import pytest
from unittest.mock import MagicMock, patch

from bank_synth.models import (
    ColumnMetadata,
    DataType,
    GenerationConfig,
    ModelPack,
    Relationship,
    RelationshipGraph,
    TableMetadata,
)


# Skip all tests if PySpark is not available
pyspark_available = True
try:
    from pyspark.sql import SparkSession
    from bank_synth.spark import SparkGenerator, SparkTrainer, SparkSessionManager
except ImportError:
    pyspark_available = False


@pytest.fixture
def sample_relationship_graph():
    """Create a sample relationship graph for testing."""
    graph = RelationshipGraph()

    customers = TableMetadata(
        name="CUSTOMERS",
        schema="CORE",
        columns=[
            ColumnMetadata(name="CUSTOMER_ID", data_type=DataType.BIGINT, is_primary_key=True, nullable=False),
            ColumnMetadata(name="FIRST_NAME", data_type=DataType.STRING),
            ColumnMetadata(name="LAST_NAME", data_type=DataType.STRING),
            ColumnMetadata(name="EMAIL", data_type=DataType.STRING),
            ColumnMetadata(name="STATUS", data_type=DataType.STRING, allowed_values=["ACTIVE", "INACTIVE"]),
        ],
        primary_key=["CUSTOMER_ID"],
    )
    graph.add_table(customers)

    accounts = TableMetadata(
        name="ACCOUNTS",
        schema="CORE",
        columns=[
            ColumnMetadata(name="ACCOUNT_ID", data_type=DataType.BIGINT, is_primary_key=True, nullable=False),
            ColumnMetadata(name="CUSTOMER_ID", data_type=DataType.BIGINT, is_foreign_key=True),
            ColumnMetadata(name="BALANCE", data_type=DataType.DECIMAL),
            ColumnMetadata(name="ACCOUNT_TYPE", data_type=DataType.STRING),
        ],
        primary_key=["ACCOUNT_ID"],
    )
    graph.add_table(accounts)

    # Add relationship
    rel = Relationship(
        name="fk_accounts_customer",
        parent_table="CUSTOMERS",
        parent_columns=["CUSTOMER_ID"],
        child_table="ACCOUNTS",
        child_columns=["CUSTOMER_ID"],
        cardinality="1:N",
        is_optional=False,
    )
    graph.add_relationship(rel)

    return graph


@pytest.fixture
def sample_model_pack(sample_relationship_graph):
    """Create a sample model pack for testing."""
    return ModelPack(
        schema_name="CORE",
        relationship_graph=sample_relationship_graph,
        tables_trained=["CUSTOMERS", "ACCOUNTS"],
        synthesizers={},
        table_stats={},
    )


@pytest.fixture
def sample_config():
    """Create a sample generation config."""
    return GenerationConfig(
        target_table="ACCOUNTS",
        target_rows=100,
        seed=42,
    )


@pytest.mark.skipif(not pyspark_available, reason="PySpark not installed")
class TestSparkSessionManager:
    """Tests for SparkSessionManager."""

    def test_init_local_mode(self):
        """Test initialization in local mode."""
        manager = SparkSessionManager(app_name="test_app", mode="local")
        assert manager.mode == "local"
        assert manager.app_name == "test_app"

    def test_get_session(self):
        """Test getting SparkSession."""
        manager = SparkSessionManager(
            app_name="test_session",
            mode="local",
            hive_enabled=False,
        )
        spark = manager.get_session()

        assert spark is not None
        assert isinstance(spark, SparkSession)

        # Cleanup
        manager.stop()

    def test_default_configs(self):
        """Test default configurations."""
        assert "local" in SparkSessionManager.DEFAULT_CONFIGS
        assert "yarn" in SparkSessionManager.DEFAULT_CONFIGS

        local_config = SparkSessionManager.DEFAULT_CONFIGS["local"]
        assert "spark.driver.memory" in local_config


@pytest.mark.skipif(not pyspark_available, reason="PySpark not installed")
class TestSparkGenerator:
    """Tests for SparkGenerator."""

    @pytest.fixture
    def spark_session(self):
        """Create a local SparkSession for testing."""
        spark = (
            SparkSession.builder
            .appName("test_generator")
            .master("local[2]")
            .config("spark.driver.memory", "1g")
            .getOrCreate()
        )
        yield spark
        spark.stop()

    def test_init(self, sample_model_pack, sample_config, spark_session):
        """Test SparkGenerator initialization."""
        generator = SparkGenerator(
            model_pack=sample_model_pack,
            config=sample_config,
            spark=spark_session,
        )

        assert generator.model_pack == sample_model_pack
        assert generator.config == sample_config

    def test_generate_small_dataset(self, sample_model_pack, sample_config, spark_session):
        """Test generating a small dataset."""
        generator = SparkGenerator(
            model_pack=sample_model_pack,
            config=sample_config,
            spark=spark_session,
        )

        result = generator.generate()

        assert "CUSTOMERS" in result
        assert "ACCOUNTS" in result

        # Check row counts (approximately)
        customers_count = result["CUSTOMERS"].count()
        accounts_count = result["ACCOUNTS"].count()

        assert customers_count > 0
        assert accounts_count == sample_config.target_rows

    def test_calculate_row_counts(self, sample_model_pack, sample_config, spark_session):
        """Test row count calculation."""
        generator = SparkGenerator(
            model_pack=sample_model_pack,
            config=sample_config,
            spark=spark_session,
        )

        row_counts = generator._calculate_row_counts(["CUSTOMERS", "ACCOUNTS"])

        assert "ACCOUNTS" in row_counts
        assert row_counts["ACCOUNTS"] == sample_config.target_rows

    def test_to_pandas(self, sample_model_pack, sample_config, spark_session):
        """Test converting to pandas DataFrames."""
        generator = SparkGenerator(
            model_pack=sample_model_pack,
            config=sample_config,
            spark=spark_session,
        )

        generator.generate()
        pandas_data = generator.to_pandas()

        assert "CUSTOMERS" in pandas_data
        assert "ACCOUNTS" in pandas_data

        import pandas as pd
        assert isinstance(pandas_data["CUSTOMERS"], pd.DataFrame)
        assert isinstance(pandas_data["ACCOUNTS"], pd.DataFrame)


@pytest.mark.skipif(not pyspark_available, reason="PySpark not installed")
class TestSparkTrainer:
    """Tests for SparkTrainer."""

    @pytest.fixture
    def spark_session(self):
        """Create a local SparkSession for testing."""
        spark = (
            SparkSession.builder
            .appName("test_trainer")
            .master("local[2]")
            .config("spark.driver.memory", "1g")
            .getOrCreate()
        )
        yield spark
        spark.stop()

    def test_init(self, sample_relationship_graph, spark_session):
        """Test SparkTrainer initialization."""
        trainer = SparkTrainer(
            relationship_graph=sample_relationship_graph,
            spark=spark_session,
        )

        assert trainer.relationship_graph == sample_relationship_graph

    def test_compute_statistics(self, sample_relationship_graph, spark_session):
        """Test computing statistics from Spark DataFrame."""
        trainer = SparkTrainer(
            relationship_graph=sample_relationship_graph,
            spark=spark_session,
        )

        # Create sample data
        sample_data = spark_session.createDataFrame([
            (1, "John", "Doe", "john@test.com", "ACTIVE"),
            (2, "Jane", "Smith", "jane@test.com", "INACTIVE"),
            (3, "Bob", "Johnson", "bob@test.com", "ACTIVE"),
        ], ["CUSTOMER_ID", "FIRST_NAME", "LAST_NAME", "EMAIL", "STATUS"])

        trainer._sample_data["CUSTOMERS"] = sample_data

        stats = trainer.compute_statistics()

        assert "CUSTOMERS" in stats
        assert stats["CUSTOMERS"].row_count == 3
        assert "CUSTOMER_ID" in stats["CUSTOMERS"].column_stats

    def test_build_fallback_models(self, sample_relationship_graph, spark_session):
        """Test building fallback synthesizer models."""
        trainer = SparkTrainer(
            relationship_graph=sample_relationship_graph,
            spark=spark_session,
        )

        # Create sample data
        sample_data = spark_session.createDataFrame([
            (1, "John", "Doe", "john@test.com", "ACTIVE"),
            (2, "Jane", "Smith", "jane@test.com", "INACTIVE"),
        ], ["CUSTOMER_ID", "FIRST_NAME", "LAST_NAME", "EMAIL", "STATUS"])

        trainer._sample_data["CUSTOMERS"] = sample_data
        trainer.compute_statistics()

        models = trainer.build_fallback_models()

        assert "CUSTOMERS" in models
        assert len(models["CUSTOMERS"]) > 0  # Serialized model bytes

    def test_train(self, sample_relationship_graph, spark_session):
        """Test full training pipeline."""
        trainer = SparkTrainer(
            relationship_graph=sample_relationship_graph,
            spark=spark_session,
        )

        # Create sample data
        sample_data = spark_session.createDataFrame([
            (1, "John", "Doe", "john@test.com", "ACTIVE"),
            (2, "Jane", "Smith", "jane@test.com", "INACTIVE"),
        ], ["CUSTOMER_ID", "FIRST_NAME", "LAST_NAME", "EMAIL", "STATUS"])

        trainer._sample_data["CUSTOMERS"] = sample_data

        model_pack = trainer.train(schema_name="CORE")

        assert model_pack is not None
        assert model_pack.schema_name == "CORE"
        assert "CUSTOMERS" in model_pack.tables_trained
        assert model_pack.training_config.get("engine") == "spark"


@pytest.mark.skipif(not pyspark_available, reason="PySpark not installed")
class TestSparkIntegration:
    """Integration tests for Spark components."""

    @pytest.fixture
    def spark_session(self):
        """Create a local SparkSession for testing."""
        spark = (
            SparkSession.builder
            .appName("test_integration")
            .master("local[2]")
            .config("spark.driver.memory", "1g")
            .getOrCreate()
        )
        yield spark
        spark.stop()

    def test_generate_with_fk_constraints(self, sample_model_pack, spark_session):
        """Test that FK constraints are applied correctly."""
        config = GenerationConfig(
            target_table="ACCOUNTS",
            target_rows=50,
            seed=42,
        )

        generator = SparkGenerator(
            model_pack=sample_model_pack,
            config=config,
            spark=spark_session,
        )

        result = generator.generate()

        customers_df = result["CUSTOMERS"].toPandas()
        accounts_df = result["ACCOUNTS"].toPandas()

        # Check that all CUSTOMER_IDs in accounts exist in customers
        customer_ids = set(customers_df["CUSTOMER_ID"])
        account_customer_ids = set(accounts_df["CUSTOMER_ID"].dropna())

        # All account customer IDs should be in customers
        assert account_customer_ids.issubset(customer_ids)
