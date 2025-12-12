"""
Tests for the output module.

Tests ORC writer, Parquet writer, and the unified OutputWriter.
"""

import pytest
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np

from bank_synth.models import (
    ColumnMetadata,
    DataType,
    GenerationConfig,
    ModelPack,
    RelationshipGraph,
    TableMetadata,
)
from bank_synth.output import OutputWriter, HiveOrcWriter, HiveParquetWriter


@pytest.fixture
def sample_model_pack():
    """Create a sample model pack for testing."""
    graph = RelationshipGraph()

    # Add customers table
    customers = TableMetadata(
        name="CUSTOMERS",
        schema="CORE",
        columns=[
            ColumnMetadata(name="CUSTOMER_ID", data_type=DataType.BIGINT, is_primary_key=True),
            ColumnMetadata(name="FIRST_NAME", data_type=DataType.STRING),
            ColumnMetadata(name="LAST_NAME", data_type=DataType.STRING),
            ColumnMetadata(name="EMAIL", data_type=DataType.STRING),
            ColumnMetadata(name="BALANCE", data_type=DataType.DECIMAL),
        ],
        primary_key=["CUSTOMER_ID"],
    )
    graph.add_table(customers)

    # Add accounts table
    accounts = TableMetadata(
        name="ACCOUNTS",
        schema="CORE",
        columns=[
            ColumnMetadata(name="ACCOUNT_ID", data_type=DataType.BIGINT, is_primary_key=True),
            ColumnMetadata(name="CUSTOMER_ID", data_type=DataType.BIGINT, is_foreign_key=True),
            ColumnMetadata(name="BALANCE", data_type=DataType.DECIMAL),
            ColumnMetadata(name="STATUS", data_type=DataType.STRING),
        ],
        primary_key=["ACCOUNT_ID"],
    )
    graph.add_table(accounts)

    return ModelPack(
        schema_name="CORE",
        relationship_graph=graph,
        tables_trained=["CUSTOMERS", "ACCOUNTS"],
    )


@pytest.fixture
def sample_data():
    """Create sample DataFrames for testing."""
    customers_df = pd.DataFrame({
        "CUSTOMER_ID": [1, 2, 3, 4, 5],
        "FIRST_NAME": ["John", "Jane", "Bob", "Alice", "Charlie"],
        "LAST_NAME": ["Doe", "Smith", "Johnson", "Williams", "Brown"],
        "EMAIL": ["john@test.com", "jane@test.com", "bob@test.com", "alice@test.com", "charlie@test.com"],
        "BALANCE": [1000.50, 2500.75, 500.00, 3500.25, 750.00],
    })

    accounts_df = pd.DataFrame({
        "ACCOUNT_ID": [101, 102, 103, 104, 105, 106],
        "CUSTOMER_ID": [1, 1, 2, 3, 4, 5],
        "BALANCE": [500.25, 500.25, 2500.75, 500.00, 3500.25, 750.00],
        "STATUS": ["ACTIVE", "ACTIVE", "ACTIVE", "CLOSED", "ACTIVE", "ACTIVE"],
    })

    return {
        "CUSTOMERS": customers_df,
        "ACCOUNTS": accounts_df,
    }


class TestHiveOrcWriter:
    """Tests for the Hive ORC writer."""

    def test_write_single_table(self, sample_model_pack, sample_data):
        """Test writing a single table to ORC."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            writer = HiveOrcWriter(output_dir, sample_model_pack)

            output_path = writer.write_table("CUSTOMERS", sample_data["CUSTOMERS"])

            assert output_path.exists()
            assert output_path.suffix == ".orc" or output_path.is_dir()

    def test_write_all_tables(self, sample_model_pack, sample_data):
        """Test writing all tables to ORC."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            writer = HiveOrcWriter(output_dir, sample_model_pack)

            output_paths = writer.write(sample_data)

            assert len(output_paths) == 2
            assert "CUSTOMERS" in output_paths
            assert "ACCOUNTS" in output_paths

            # Check DDL was generated
            ddl_path = output_dir / "ddl" / "hive_orc_tables.sql"
            assert ddl_path.exists()

    def test_write_with_compression(self, sample_model_pack, sample_data):
        """Test writing with different compression codecs."""
        for compression in ["snappy", "zlib", "none"]:
            with tempfile.TemporaryDirectory() as tmpdir:
                output_dir = Path(tmpdir)
                writer = HiveOrcWriter(output_dir, sample_model_pack, compression=compression)

                output_paths = writer.write(sample_data)
                assert len(output_paths) == 2

    def test_ddl_generation(self, sample_model_pack, sample_data):
        """Test that correct DDL is generated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            writer = HiveOrcWriter(output_dir, sample_model_pack)

            writer.write(sample_data)

            ddl_path = output_dir / "ddl" / "hive_orc_tables.sql"
            ddl_content = ddl_path.read_text()

            assert "CREATE EXTERNAL TABLE" in ddl_content
            assert "STORED AS ORC" in ddl_content
            assert "customers" in ddl_content.lower()
            assert "accounts" in ddl_content.lower()


class TestHiveParquetWriter:
    """Tests for the Hive Parquet writer."""

    def test_write_single_table(self, sample_model_pack, sample_data):
        """Test writing a single table to Parquet."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            writer = HiveParquetWriter(output_dir, sample_model_pack)

            output_path = writer.write_table("CUSTOMERS", sample_data["CUSTOMERS"])

            assert output_path.exists()
            assert output_path.suffix == ".parquet" or output_path.is_dir()

    def test_write_all_tables(self, sample_model_pack, sample_data):
        """Test writing all tables to Parquet."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            writer = HiveParquetWriter(output_dir, sample_model_pack)

            output_paths = writer.write(sample_data)

            assert len(output_paths) == 2
            assert "CUSTOMERS" in output_paths
            assert "ACCOUNTS" in output_paths

    def test_read_back_parquet(self, sample_model_pack, sample_data):
        """Test that written Parquet can be read back."""
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            writer = HiveParquetWriter(output_dir, sample_model_pack)

            output_path = writer.write_table("CUSTOMERS", sample_data["CUSTOMERS"])

            # Read back
            table = pq.read_table(output_path)
            df = table.to_pandas()

            assert len(df) == len(sample_data["CUSTOMERS"])
            assert "customer_id" in df.columns


class TestOutputWriter:
    """Tests for the unified output writer."""

    def test_write_multiple_formats(self, sample_model_pack, sample_data):
        """Test writing to multiple formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = GenerationConfig(
                target_table="ACCOUNTS",
                target_rows=100,
                output_formats=["hive_orc", "hive_parquet", "oracle"],
                output_dir=Path(tmpdir),
                run_id="test_run",
            )

            writer = OutputWriter(config, sample_model_pack)
            output_paths = writer.write(sample_data)

            assert "hive_orc" in output_paths
            assert "hive_parquet" in output_paths
            assert "oracle" in output_paths

    def test_write_orc_format(self, sample_model_pack, sample_data):
        """Test writing ORC format only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = GenerationConfig(
                target_table="ACCOUNTS",
                target_rows=100,
                output_formats=["hive_orc"],
                output_dir=Path(tmpdir),
                run_id="test_run",
            )

            writer = OutputWriter(config, sample_model_pack)
            output_paths = writer.write(sample_data)

            assert "hive_orc" in output_paths
            assert len(output_paths) == 1

    def test_manifest_generation(self, sample_model_pack, sample_data):
        """Test that manifest file is generated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = GenerationConfig(
                target_table="ACCOUNTS",
                target_rows=100,
                output_formats=["hive_orc"],
                output_dir=Path(tmpdir),
                run_id="test_run",
            )

            writer = OutputWriter(config, sample_model_pack)
            writer.write(sample_data)

            manifest_path = writer.get_output_dir() / "manifest.json"
            assert manifest_path.exists()

            import json
            with open(manifest_path) as f:
                manifest = json.load(f)

            assert manifest["run_id"] == "test_run"
            assert "CUSTOMERS" in manifest["tables"]
            assert "ACCOUNTS" in manifest["tables"]

    def test_oracle_csv_output(self, sample_model_pack, sample_data):
        """Test Oracle CSV output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = GenerationConfig(
                target_table="ACCOUNTS",
                target_rows=100,
                output_formats=["oracle"],
                output_dir=Path(tmpdir),
                run_id="test_run",
            )

            writer = OutputWriter(config, sample_model_pack)
            output_paths = writer.write(sample_data)

            assert "oracle" in output_paths

            # Check CSV files exist
            csv_path = output_paths["oracle"]["CUSTOMERS"]
            assert csv_path.exists()
            assert csv_path.suffix == ".csv"

            # Check control file exists
            ctl_path = writer.get_output_dir() / "ddl" / "customers.ctl"
            assert ctl_path.exists()
