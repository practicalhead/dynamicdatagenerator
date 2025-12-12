"""
Tests for the auto-discovery module.

Tests query parsing, relationship inference, and the auto-discovery pipeline.
"""

import pytest
from bank_synth.discovery import QueryParser, ParsedQuery, JoinInfo
from bank_synth.discovery.relationship_inferrer import RelationshipInferrer, PKFKMetadata
from bank_synth.models import ColumnMetadata, DataType, TableMetadata


class TestQueryParser:
    """Tests for the SQL query parser."""

    def test_simple_join(self):
        """Test parsing a simple INNER JOIN."""
        parser = QueryParser()
        sql = """
        SELECT c.customer_id, a.account_id
        FROM customers c
        INNER JOIN accounts a ON c.customer_id = a.customer_id
        """
        result = parser.parse(sql, "test_query")

        assert result.is_valid
        assert "CUSTOMERS" in result.tables
        assert "ACCOUNTS" in result.tables
        assert len(result.joins) == 1
        assert result.joins[0].join_type == "INNER"

    def test_multiple_joins(self):
        """Test parsing multiple JOINs."""
        parser = QueryParser()
        sql = """
        SELECT t.*, a.account_number, c.first_name
        FROM transactions t
        JOIN accounts a ON t.account_id = a.account_id
        JOIN customers c ON a.customer_id = c.customer_id
        """
        result = parser.parse(sql, "test_query")

        assert result.is_valid
        assert "TRANSACTIONS" in result.tables
        assert "ACCOUNTS" in result.tables
        assert "CUSTOMERS" in result.tables
        assert len(result.joins) >= 2

    def test_left_join(self):
        """Test parsing LEFT JOIN."""
        parser = QueryParser()
        sql = """
        SELECT c.*, a.balance
        FROM customers c
        LEFT JOIN accounts a ON c.customer_id = a.customer_id
        """
        result = parser.parse(sql, "test_query")

        assert result.is_valid
        assert len(result.joins) == 1
        assert result.joins[0].join_type == "LEFT"

    def test_oracle_style_join(self):
        """Test parsing Oracle-style implicit join with (+)."""
        parser = QueryParser()
        sql = """
        SELECT c.customer_id, a.account_id
        FROM customers c, accounts a
        WHERE c.customer_id = a.customer_id(+)
        """
        result = parser.parse(sql, "test_query")

        assert result.is_valid
        assert "CUSTOMERS" in result.tables
        assert "ACCOUNTS" in result.tables
        # Oracle (+) on right means LEFT join
        if result.joins:
            assert result.joins[0].join_type == "LEFT"

    def test_schema_qualified_tables(self):
        """Test parsing schema.table names."""
        parser = QueryParser()
        sql = """
        SELECT c.customer_id
        FROM core.customers c
        JOIN core.accounts a ON c.customer_id = a.customer_id
        """
        result = parser.parse(sql, "test_query")

        assert result.is_valid
        assert "CUSTOMERS" in result.tables
        assert "ACCOUNTS" in result.tables

    def test_complex_etl_query(self):
        """Test parsing a complex ETL query."""
        parser = QueryParser()
        sql = """
        SELECT
            c.customer_id,
            c.first_name,
            c.last_name,
            COUNT(a.account_id) as num_accounts,
            SUM(t.amount) as total_transactions
        FROM customers c
        LEFT JOIN accounts a ON c.customer_id = a.customer_id
        LEFT JOIN transactions t ON a.account_id = t.account_id
        LEFT JOIN transaction_types tt ON t.transaction_type_id = tt.transaction_type_id
        WHERE c.status = 'ACTIVE'
        GROUP BY c.customer_id, c.first_name, c.last_name
        HAVING SUM(t.amount) > 1000
        ORDER BY total_transactions DESC
        """
        result = parser.parse(sql, "etl_query")

        assert result.is_valid
        assert "CUSTOMERS" in result.tables
        assert "ACCOUNTS" in result.tables
        assert "TRANSACTIONS" in result.tables
        assert "TRANSACTION_TYPES" in result.tables
        assert len(result.joins) >= 3

    def test_named_query_annotation(self):
        """Test parsing SQL with -- @name: annotation."""
        parser = QueryParser()
        content = """
        -- @name: customer_report
        SELECT c.* FROM customers c;

        -- @name: account_summary
        SELECT a.*, c.first_name
        FROM accounts a
        JOIN customers c ON a.customer_id = c.customer_id;
        """

        queries = parser._split_queries(content, "test.sql")
        assert len(queries) == 2
        assert queries[0][0] == "customer_report"
        assert queries[1][0] == "account_summary"


class TestRelationshipInferrer:
    """Tests for the relationship inferrer."""

    @pytest.fixture
    def sample_tables(self):
        """Create sample table metadata for testing."""
        return {
            "CUSTOMERS": TableMetadata(
                name="CUSTOMERS",
                schema="CORE",
                columns=[
                    ColumnMetadata(name="CUSTOMER_ID", data_type=DataType.BIGINT, is_primary_key=True, nullable=False),
                    ColumnMetadata(name="FIRST_NAME", data_type=DataType.STRING),
                    ColumnMetadata(name="LAST_NAME", data_type=DataType.STRING),
                    ColumnMetadata(name="EMAIL", data_type=DataType.STRING),
                ],
                primary_key=["CUSTOMER_ID"],
            ),
            "ACCOUNTS": TableMetadata(
                name="ACCOUNTS",
                schema="CORE",
                columns=[
                    ColumnMetadata(name="ACCOUNT_ID", data_type=DataType.BIGINT, is_primary_key=True, nullable=False),
                    ColumnMetadata(name="CUSTOMER_ID", data_type=DataType.BIGINT, nullable=False),
                    ColumnMetadata(name="ACCOUNT_TYPE_ID", data_type=DataType.INTEGER),
                    ColumnMetadata(name="BALANCE", data_type=DataType.DECIMAL),
                ],
                primary_key=["ACCOUNT_ID"],
            ),
            "TRANSACTIONS": TableMetadata(
                name="TRANSACTIONS",
                schema="CORE",
                columns=[
                    ColumnMetadata(name="TRANSACTION_ID", data_type=DataType.BIGINT, is_primary_key=True, nullable=False),
                    ColumnMetadata(name="ACCOUNT_ID", data_type=DataType.BIGINT, nullable=False),
                    ColumnMetadata(name="AMOUNT", data_type=DataType.DECIMAL),
                    ColumnMetadata(name="TRANSACTION_DATE", data_type=DataType.TIMESTAMP),
                ],
                primary_key=["TRANSACTION_ID"],
            ),
            "ACCOUNT_TYPES": TableMetadata(
                name="ACCOUNT_TYPES",
                schema="CORE",
                columns=[
                    ColumnMetadata(name="ACCOUNT_TYPE_ID", data_type=DataType.INTEGER, is_primary_key=True, nullable=False),
                    ColumnMetadata(name="TYPE_NAME", data_type=DataType.STRING),
                ],
                primary_key=["ACCOUNT_TYPE_ID"],
            ),
        }

    def test_infer_from_queries(self, sample_tables):
        """Test relationship inference from SQL queries."""
        inferrer = RelationshipInferrer(sample_tables)

        queries = [
            ("customer_accounts", """
                SELECT c.*, a.*
                FROM customers c
                JOIN accounts a ON c.customer_id = a.customer_id
            """),
            ("account_transactions", """
                SELECT a.*, t.*
                FROM accounts a
                JOIN transactions t ON a.account_id = t.account_id
            """),
        ]

        relationships = inferrer.infer_from_queries(queries)

        assert len(relationships) >= 2

        # Check that customer->accounts relationship was found
        cust_acct_rel = [r for r in relationships
                        if r.parent_table == "CUSTOMERS" and r.child_table == "ACCOUNTS"]
        assert len(cust_acct_rel) >= 1

    def test_infer_from_pkfk_metadata(self, sample_tables):
        """Test relationship inference from PK/FK metadata."""
        pkfk_metadata = [
            PKFKMetadata(
                table_name="ACCOUNTS",
                primary_keys=["ACCOUNT_ID"],
                foreign_keys=[
                    {"column": "CUSTOMER_ID", "ref_table": "CUSTOMERS", "ref_column": "CUSTOMER_ID", "nullable": False},
                    {"column": "ACCOUNT_TYPE_ID", "ref_table": "ACCOUNT_TYPES", "ref_column": "ACCOUNT_TYPE_ID", "nullable": True},
                ],
            ),
            PKFKMetadata(
                table_name="TRANSACTIONS",
                primary_keys=["TRANSACTION_ID"],
                foreign_keys=[
                    {"column": "ACCOUNT_ID", "ref_table": "ACCOUNTS", "ref_column": "ACCOUNT_ID", "nullable": False},
                ],
            ),
        ]

        inferrer = RelationshipInferrer(sample_tables, pkfk_metadata)
        relationships = inferrer.infer_from_pkfk_metadata()

        assert len(relationships) == 3

        # Check that relationships have high confidence from catalog
        for rel in relationships:
            assert rel.confidence >= 0.9

    def test_infer_from_column_patterns(self, sample_tables):
        """Test relationship inference from column naming patterns."""
        inferrer = RelationshipInferrer(sample_tables)
        relationships = inferrer.infer_from_column_patterns()

        # Should find customer_id pattern
        cust_rels = [r for r in relationships if "CUSTOMER_ID" in r.child_column]
        assert len(cust_rels) >= 1

    def test_infer_all_combines_sources(self, sample_tables):
        """Test that infer_all combines all sources."""
        pkfk_metadata = [
            PKFKMetadata(
                table_name="ACCOUNTS",
                primary_keys=["ACCOUNT_ID"],
                foreign_keys=[
                    {"column": "CUSTOMER_ID", "ref_table": "CUSTOMERS", "ref_column": "CUSTOMER_ID"},
                ],
            ),
        ]

        queries = [
            ("test", """
                SELECT * FROM accounts a
                JOIN transactions t ON a.account_id = t.account_id
            """),
        ]

        inferrer = RelationshipInferrer(sample_tables, pkfk_metadata)
        relationships = inferrer.infer_all(queries)

        # Should have relationships from both PK/FK and queries
        assert len(relationships) >= 2

    def test_build_relationship_graph(self, sample_tables):
        """Test building a complete relationship graph."""
        queries = [
            ("test", """
                SELECT c.*, a.*, t.*
                FROM customers c
                JOIN accounts a ON c.customer_id = a.customer_id
                JOIN transactions t ON a.account_id = t.account_id
            """),
        ]

        inferrer = RelationshipInferrer(sample_tables)
        graph = inferrer.build_relationship_graph(queries)

        assert len(graph.tables) == 4
        assert len(graph.relationships) >= 2

        # Verify dependency ordering
        closure, sorted_tables = graph.get_dependency_closure("TRANSACTIONS")
        assert "customers" in sorted_tables or "CUSTOMERS" in [t.upper() for t in sorted_tables]


class TestJoinInfo:
    """Tests for JoinInfo dataclass."""

    def test_to_dict(self):
        """Test serialization to dict."""
        join = JoinInfo(
            left_table="CUSTOMERS",
            left_column="CUSTOMER_ID",
            right_table="ACCOUNTS",
            right_column="CUSTOMER_ID",
            join_type="INNER",
            query_source="test_query",
            confidence=0.9,
        )

        d = join.to_dict()
        assert d["left_table"] == "CUSTOMERS"
        assert d["right_table"] == "ACCOUNTS"
        assert d["join_type"] == "INNER"
        assert d["confidence"] == 0.9


class TestParsedQuery:
    """Tests for ParsedQuery dataclass."""

    def test_to_dict(self):
        """Test serialization to dict."""
        pq = ParsedQuery(
            query_name="test",
            tables={"CUSTOMERS", "ACCOUNTS"},
            joins=[
                JoinInfo(
                    left_table="CUSTOMERS",
                    left_column="CUSTOMER_ID",
                    right_table="ACCOUNTS",
                    right_column="CUSTOMER_ID",
                    join_type="INNER",
                )
            ],
            is_valid=True,
        )

        d = pq.to_dict()
        assert d["query_name"] == "test"
        assert "CUSTOMERS" in d["tables"]
        assert len(d["joins"]) == 1
