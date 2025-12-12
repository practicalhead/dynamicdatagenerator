"""
Quality assessment and reporting for generated data.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from bank_synth.models import ModelPack, RelationshipGraph, TableStats

logger = logging.getLogger(__name__)


class QualityReporter:
    """
    Generates quality reports comparing generated data to training statistics.

    Reports include:
    - Row count accuracy
    - Distribution similarity (for numeric columns)
    - Cardinality preservation
    - FK referential integrity validation
    - Null ratio comparison
    """

    def __init__(
        self,
        model_pack: ModelPack,
        generated_data: Dict[str, pd.DataFrame],
    ):
        """
        Initialize quality reporter.

        Args:
            model_pack: Trained model pack with statistics
            generated_data: Dictionary of generated DataFrames
        """
        self.model_pack = model_pack
        self.generated_data = generated_data
        self.graph = model_pack.relationship_graph

        self.report: Dict[str, Any] = {
            "generated_at": datetime.now().isoformat(),
            "summary": {},
            "tables": {},
            "referential_integrity": {},
        }

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive quality report.

        Returns:
            Report dictionary
        """
        logger.info("Generating quality report...")

        # Analyze each table
        for table_name, df in self.generated_data.items():
            table_report = self._analyze_table(table_name, df)
            self.report["tables"][table_name] = table_report

        # Check referential integrity
        self.report["referential_integrity"] = self._check_referential_integrity()

        # Generate summary
        self.report["summary"] = self._generate_summary()

        return self.report

    def _analyze_table(self, table_name: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze a single table's quality."""
        table_upper = table_name.upper()
        training_stats = self.model_pack.table_stats.get(table_upper)

        report = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": {},
        }

        if training_stats:
            report["training_row_count"] = training_stats.row_count
            report["row_count_ratio"] = len(df) / max(training_stats.row_count, 1)

        # Analyze each column
        for col_name in df.columns:
            col_report = self._analyze_column(
                col_name,
                df[col_name],
                training_stats.column_stats.get(col_name) if training_stats else None,
            )
            report["columns"][col_name] = col_report

        return report

    def _analyze_column(
        self,
        col_name: str,
        series: pd.Series,
        training_stats: Optional[Any],
    ) -> Dict[str, Any]:
        """Analyze a single column's quality."""
        report = {
            "dtype": str(series.dtype),
            "null_count": int(series.isna().sum()),
            "null_fraction": float(series.isna().mean()),
            "distinct_count": int(series.nunique()),
            "is_unique": series.nunique() == len(series),
        }

        # Numeric analysis
        if np.issubdtype(series.dtype, np.number):
            report["min"] = float(series.min()) if not series.isna().all() else None
            report["max"] = float(series.max()) if not series.isna().all() else None
            report["mean"] = float(series.mean()) if not series.isna().all() else None
            report["std"] = float(series.std()) if not series.isna().all() else None

        # Compare to training stats
        if training_stats:
            report["training_comparison"] = self._compare_to_training(series, training_stats)

        return report

    def _compare_to_training(self, series: pd.Series, training_stats: Any) -> Dict[str, Any]:
        """Compare generated column to training statistics."""
        comparison = {}

        # Null fraction comparison
        gen_null_frac = series.isna().mean()
        train_null_frac = training_stats.null_fraction
        comparison["null_fraction_diff"] = abs(gen_null_frac - train_null_frac)

        # Cardinality comparison
        if training_stats.distinct_count > 0:
            gen_distinct = series.nunique()
            comparison["cardinality_ratio"] = gen_distinct / training_stats.distinct_count

        # Numeric distribution comparison
        if training_stats.mean is not None and np.issubdtype(series.dtype, np.number):
            gen_mean = series.mean()
            gen_std = series.std()

            if training_stats.mean != 0:
                comparison["mean_relative_error"] = abs(gen_mean - training_stats.mean) / abs(training_stats.mean)
            else:
                comparison["mean_absolute_error"] = abs(gen_mean - training_stats.mean)

            if training_stats.std and training_stats.std != 0:
                comparison["std_relative_error"] = abs(gen_std - training_stats.std) / abs(training_stats.std)

        # Range comparison
        if training_stats.min_value is not None and training_stats.max_value is not None:
            gen_min = series.min()
            gen_max = series.max()

            try:
                train_min = float(training_stats.min_value)
                train_max = float(training_stats.max_value)
                comparison["min_in_range"] = gen_min >= train_min
                comparison["max_in_range"] = gen_max <= train_max
            except (ValueError, TypeError):
                pass

        return comparison

    def _check_referential_integrity(self) -> Dict[str, Any]:
        """Check FK referential integrity across all tables."""
        ri_report = {
            "total_relationships": len(self.graph.relationships),
            "valid_relationships": 0,
            "violations": [],
        }

        for rel in self.graph.relationships:
            parent_upper = rel.parent_table.upper()
            child_upper = rel.child_table.upper()

            if parent_upper not in self.generated_data or child_upper not in self.generated_data:
                continue

            parent_df = self.generated_data[parent_upper]
            child_df = self.generated_data[child_upper]

            # Check each FK column pair
            is_valid = True
            for parent_col, child_col in zip(rel.parent_columns, rel.child_columns):
                parent_col_upper = parent_col.upper()
                child_col_upper = child_col.upper()

                if parent_col_upper not in parent_df.columns or child_col_upper not in child_df.columns:
                    continue

                parent_values = set(parent_df[parent_col_upper].dropna().unique())
                child_values = set(child_df[child_col_upper].dropna().unique())

                # Check for orphans
                orphans = child_values - parent_values
                if orphans:
                    is_valid = False
                    ri_report["violations"].append({
                        "relationship": rel.name,
                        "parent_table": parent_upper,
                        "parent_column": parent_col_upper,
                        "child_table": child_upper,
                        "child_column": child_col_upper,
                        "orphan_count": len(orphans),
                        "sample_orphans": list(orphans)[:5],
                    })

            if is_valid:
                ri_report["valid_relationships"] += 1

        ri_report["integrity_score"] = (
            ri_report["valid_relationships"] / max(ri_report["total_relationships"], 1)
        )

        return ri_report

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate overall quality summary."""
        total_rows = sum(len(df) for df in self.generated_data.values())
        total_tables = len(self.generated_data)

        summary = {
            "total_tables": total_tables,
            "total_rows": total_rows,
            "referential_integrity_score": self.report["referential_integrity"].get("integrity_score", 1.0),
            "tables_with_issues": [],
        }

        # Identify tables with potential issues
        for table_name, table_report in self.report["tables"].items():
            issues = []

            # Check for high null rates
            for col_name, col_report in table_report.get("columns", {}).items():
                if col_report.get("null_fraction", 0) > 0.5:
                    issues.append(f"High null rate in {col_name}")

            if issues:
                summary["tables_with_issues"].append({
                    "table": table_name,
                    "issues": issues,
                })

        return summary

    def save(self, output_dir: Path) -> tuple:
        """
        Save quality report to files.

        Args:
            output_dir: Output directory

        Returns:
            Tuple of (json_path, markdown_path)
        """
        report_dir = Path(output_dir) / "report"
        report_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON
        json_path = report_dir / "quality.json"
        with open(json_path, "w") as f:
            json.dump(self.report, f, indent=2, default=str)

        # Generate and save Markdown
        md_content = self._generate_markdown()
        md_path = report_dir / "quality.md"
        with open(md_path, "w") as f:
            f.write(md_content)

        logger.info(f"Quality report saved to {report_dir}")
        return json_path, md_path

    def _generate_markdown(self) -> str:
        """Generate Markdown version of quality report."""
        lines = [
            "# Data Quality Report",
            "",
            f"Generated: {self.report['generated_at']}",
            "",
            "## Summary",
            "",
            f"- **Total Tables**: {self.report['summary']['total_tables']}",
            f"- **Total Rows**: {self.report['summary']['total_rows']:,}",
            f"- **Referential Integrity Score**: {self.report['summary']['referential_integrity_score']:.2%}",
            "",
        ]

        # Referential Integrity
        ri = self.report["referential_integrity"]
        lines.extend([
            "## Referential Integrity",
            "",
            f"- Valid Relationships: {ri['valid_relationships']}/{ri['total_relationships']}",
        ])

        if ri.get("violations"):
            lines.append("")
            lines.append("### Violations")
            lines.append("")
            for v in ri["violations"]:
                lines.append(f"- **{v['relationship']}**: {v['orphan_count']} orphan values in {v['child_table']}.{v['child_column']}")

        lines.append("")

        # Per-table details
        lines.append("## Table Details")
        lines.append("")

        for table_name, table_report in self.report["tables"].items():
            lines.append(f"### {table_name}")
            lines.append("")
            lines.append(f"- Rows: {table_report['row_count']:,}")
            lines.append(f"- Columns: {table_report['column_count']}")

            if "training_row_count" in table_report:
                lines.append(f"- Training Rows: {table_report['training_row_count']:,}")
                lines.append(f"- Row Count Ratio: {table_report['row_count_ratio']:.2f}")

            lines.append("")

            # Column table
            lines.append("| Column | Type | Nulls | Distinct |")
            lines.append("|--------|------|-------|----------|")

            for col_name, col_report in table_report.get("columns", {}).items():
                null_pct = f"{col_report['null_fraction']:.1%}"
                lines.append(
                    f"| {col_name} | {col_report['dtype']} | {null_pct} | {col_report['distinct_count']:,} |"
                )

            lines.append("")

        return "\n".join(lines)
