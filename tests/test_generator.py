"""Tests for the generator module."""

import pytest
import pandas as pd
import numpy as np

from bank_synth.models import (
    ColumnMetadata,
    DataType,
    GenerationConfig,
    ModelPack,
    Relationship,
    RelationshipGraph,
    TableMetadata,
    TableStats,
    ColumnStats,
)
from bank_synth.generator import Generator


class TestGenerator:
    """Tests for Generator class."""

    @pytest.fixture
    def simple_model_pack(self):
        """Create a simple model pack for testing."""
        graph = RelationshipGraph()

        # Add tables
        customers = TableMetadata(
            name="CUSTOMERS",
            schema="TEST",
            columns=[
                ColumnMetadata(name="CUSTOMER_ID", data_type=DataType.BIGINT, is_primary_key=True),
                ColumnMetadata(name="NAME", data_type=DataType.STRING),
                ColumnMetadata(name="STATUS", data_type=DataType.STRING, allowed_values=["ACTIVE", "INACTIVE"]),
            ],
            primary_key=["CUSTOMER_ID"],
        )
        accounts = TableMetadata(
            name="ACCOUNTS",
            schema="TEST",
            columns=[
                ColumnMetadata(name="ACCOUNT_ID", data_type=DataType.BIGINT, is_primary_key=True),
                ColumnMetadata(name="CUSTOMER_ID", data_type=DataType.BIGINT, is_foreign_key=True),
                ColumnMetadata(name="BALANCE", data_type=DataType.DECIMAL, precision=18, scale=2),
            ],
            primary_key=["ACCOUNT_ID"],
        )

        graph.add_table(customers)
        graph.add_table(accounts)

        graph.add_relationship(Relationship(
            name="fk_accounts_customer",
            parent_table="CUSTOMERS",
            parent_columns=["CUSTOMER_ID"],
            child_table="ACCOUNTS",
            child_columns=["CUSTOMER_ID"],
        ))

        # Create model pack with fallback synthesizers
        model_pack = ModelPack(
            schema_name="TEST",
            relationship_graph=graph,
            tables_trained=["CUSTOMERS", "ACCOUNTS"],
        )

        # Add table stats
        model_pack.table_stats["CUSTOMERS"] = TableStats(
            name="CUSTOMERS",
            row_count=100,
            column_stats={
                "CUSTOMER_ID": ColumnStats(name="CUSTOMER_ID", data_type=DataType.BIGINT),
                "NAME": ColumnStats(name="NAME", data_type=DataType.STRING),
                "STATUS": ColumnStats(name="STATUS", data_type=DataType.STRING),
            },
        )
        model_pack.table_stats["ACCOUNTS"] = TableStats(
            name="ACCOUNTS",
            row_count=200,
            column_stats={
                "ACCOUNT_ID": ColumnStats(name="ACCOUNT_ID", data_type=DataType.BIGINT),
                "CUSTOMER_ID": ColumnStats(name="CUSTOMER_ID", data_type=DataType.BIGINT),
                "BALANCE": ColumnStats(name="BALANCE", data_type=DataType.DECIMAL),
            },
        )

        return model_pack

    def test_basic_generation(self, simple_model_pack):
        """Test basic data generation."""
        config = GenerationConfig(
            target_table="ACCOUNTS",
            target_rows=100,
            seed=42,
        )

        generator = Generator(simple_model_pack, config)
        data = generator.generate()

        assert "CUSTOMERS" in data
        assert "ACCOUNTS" in data
        assert len(data["ACCOUNTS"]) == 100

    def test_fk_consistency(self, simple_model_pack):
        """Test FK referential integrity."""
        config = GenerationConfig(
            target_table="ACCOUNTS",
            target_rows=100,
            seed=42,
        )

        generator = Generator(simple_model_pack, config)
        data = generator.generate()

        # All CUSTOMER_ID values in ACCOUNTS should exist in CUSTOMERS
        customer_ids = set(data["CUSTOMERS"]["CUSTOMER_ID"].dropna())
        account_customer_ids = set(data["ACCOUNTS"]["CUSTOMER_ID"].dropna())

        assert account_customer_ids.issubset(customer_ids)

    def test_deterministic_with_seed(self, simple_model_pack):
        """Test that same seed produces same results."""
        config1 = GenerationConfig(
            target_table="ACCOUNTS",
            target_rows=50,
            seed=12345,
        )
        config2 = GenerationConfig(
            target_table="ACCOUNTS",
            target_rows=50,
            seed=12345,
        )

        generator1 = Generator(simple_model_pack, config1)
        data1 = generator1.generate()

        generator2 = Generator(simple_model_pack, config2)
        data2 = generator2.generate()

        # Same seed should produce identical data
        pd.testing.assert_frame_equal(data1["CUSTOMERS"], data2["CUSTOMERS"])
        pd.testing.assert_frame_equal(data1["ACCOUNTS"], data2["ACCOUNTS"])

    def test_different_seeds_different_results(self, simple_model_pack):
        """Test that different seeds produce different results."""
        config1 = GenerationConfig(
            target_table="ACCOUNTS",
            target_rows=50,
            seed=111,
        )
        config2 = GenerationConfig(
            target_table="ACCOUNTS",
            target_rows=50,
            seed=222,
        )

        generator1 = Generator(simple_model_pack, config1)
        data1 = generator1.generate()

        generator2 = Generator(simple_model_pack, config2)
        data2 = generator2.generate()

        # Different seeds should produce different data
        # (at least the IDs should differ)
        assert not data1["CUSTOMERS"]["CUSTOMER_ID"].equals(data2["CUSTOMERS"]["CUSTOMER_ID"])

    def test_pk_uniqueness(self, simple_model_pack):
        """Test that primary keys are unique."""
        config = GenerationConfig(
            target_table="ACCOUNTS",
            target_rows=100,
            seed=42,
        )

        generator = Generator(simple_model_pack, config)
        data = generator.generate()

        # Check PK uniqueness
        assert data["CUSTOMERS"]["CUSTOMER_ID"].is_unique
        assert data["ACCOUNTS"]["ACCOUNT_ID"].is_unique
