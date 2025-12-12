"""Tests for core data models."""

import tempfile
from pathlib import Path

import pytest

from bank_synth.models import (
    ColumnMetadata,
    DataType,
    ModelPack,
    PrivacyLevel,
    Relationship,
    RelationshipGraph,
    TableMetadata,
)


class TestColumnMetadata:
    """Tests for ColumnMetadata."""

    def test_basic_column(self):
        col = ColumnMetadata(
            name="CUSTOMER_ID",
            data_type=DataType.BIGINT,
            nullable=False,
            is_primary_key=True,
        )
        assert col.name == "CUSTOMER_ID"
        assert col.data_type == DataType.BIGINT
        assert col.nullable is False
        assert col.is_primary_key is True

    def test_serialization(self):
        col = ColumnMetadata(
            name="AMOUNT",
            data_type=DataType.DECIMAL,
            precision=18,
            scale=2,
            privacy_level=PrivacyLevel.INTERNAL,
        )
        data = col.to_dict()
        restored = ColumnMetadata.from_dict(data)

        assert restored.name == col.name
        assert restored.data_type == col.data_type
        assert restored.precision == col.precision
        assert restored.scale == col.scale
        assert restored.privacy_level == col.privacy_level


class TestTableMetadata:
    """Tests for TableMetadata."""

    def test_basic_table(self):
        table = TableMetadata(
            name="CUSTOMERS",
            schema="CORE",
            columns=[
                ColumnMetadata(name="ID", data_type=DataType.BIGINT, is_primary_key=True),
                ColumnMetadata(name="NAME", data_type=DataType.STRING),
            ],
            primary_key=["ID"],
        )
        assert table.full_name == "CORE.CUSTOMERS"
        assert table.column_names == ["ID", "NAME"]
        assert table.get_column("ID").is_primary_key is True

    def test_case_insensitive_column_lookup(self):
        table = TableMetadata(
            name="TEST",
            columns=[
                ColumnMetadata(name="CUSTOMER_ID", data_type=DataType.BIGINT),
            ],
        )
        assert table.get_column("customer_id") is not None
        assert table.get_column("CUSTOMER_ID") is not None


class TestRelationshipGraph:
    """Tests for RelationshipGraph."""

    def test_add_tables_and_relationships(self):
        graph = RelationshipGraph()

        # Add tables
        customers = TableMetadata(name="CUSTOMERS", primary_key=["CUSTOMER_ID"])
        accounts = TableMetadata(name="ACCOUNTS", primary_key=["ACCOUNT_ID"])

        graph.add_table(customers)
        graph.add_table(accounts)

        # Add relationship
        rel = Relationship(
            name="fk_accounts_customer",
            parent_table="CUSTOMERS",
            parent_columns=["CUSTOMER_ID"],
            child_table="ACCOUNTS",
            child_columns=["CUSTOMER_ID"],
        )
        graph.add_relationship(rel)

        assert len(graph.tables) == 2
        assert len(graph.relationships) == 1
        assert graph.get_parent_tables("ACCOUNTS") == ["CUSTOMERS"]
        assert graph.get_child_tables("CUSTOMERS") == ["ACCOUNTS"]

    def test_dependency_closure(self):
        graph = RelationshipGraph()

        # Add tables: CUSTOMERS -> ACCOUNTS -> TRANSACTIONS
        graph.add_table(TableMetadata(name="CUSTOMERS", primary_key=["CUSTOMER_ID"]))
        graph.add_table(TableMetadata(name="ACCOUNTS", primary_key=["ACCOUNT_ID"]))
        graph.add_table(TableMetadata(name="TRANSACTIONS", primary_key=["TRANSACTION_ID"]))

        graph.add_relationship(Relationship(
            name="fk_accounts_customer",
            parent_table="CUSTOMERS",
            parent_columns=["CUSTOMER_ID"],
            child_table="ACCOUNTS",
            child_columns=["CUSTOMER_ID"],
        ))
        graph.add_relationship(Relationship(
            name="fk_transactions_account",
            parent_table="ACCOUNTS",
            parent_columns=["ACCOUNT_ID"],
            child_table="TRANSACTIONS",
            child_columns=["ACCOUNT_ID"],
        ))

        # Get closure for TRANSACTIONS
        closure, sorted_tables = graph.get_dependency_closure("TRANSACTIONS")

        assert "transactions" in closure
        assert "accounts" in closure
        assert "customers" in closure

        # Verify topological order (parents before children)
        assert sorted_tables.index("customers") < sorted_tables.index("accounts")
        assert sorted_tables.index("accounts") < sorted_tables.index("transactions")


class TestModelPack:
    """Tests for ModelPack."""

    def test_save_and_load(self):
        model_pack = ModelPack(
            version="1.0.0",
            schema_name="TEST",
            tables_trained=["CUSTOMERS", "ACCOUNTS"],
            sample_strategy="percent:1",
        )

        # Add a relationship
        model_pack.relationship_graph.add_table(
            TableMetadata(name="CUSTOMERS", primary_key=["CUSTOMER_ID"])
        )
        model_pack.relationship_graph.add_table(
            TableMetadata(name="ACCOUNTS", primary_key=["ACCOUNT_ID"])
        )
        model_pack.relationship_graph.add_relationship(Relationship(
            name="fk_test",
            parent_table="CUSTOMERS",
            parent_columns=["CUSTOMER_ID"],
            child_table="ACCOUNTS",
            child_columns=["CUSTOMER_ID"],
        ))

        # Save and reload
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_model"
            model_pack.save(path)

            loaded = ModelPack.load(path)

            assert loaded.version == model_pack.version
            assert loaded.schema_name == model_pack.schema_name
            assert loaded.tables_trained == model_pack.tables_trained
            assert len(loaded.relationship_graph.relationships) == 1
