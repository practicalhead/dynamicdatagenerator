"""
Command-line interface for bank_synth.

Provides train, generate, and discover commands for synthetic data generation.
Zero-intervention workflow using query-based relationship discovery.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional

import click
import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from bank_synth.models import GenerationConfig, ModelPack, RelationshipGraph

console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with Rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@click.group()
@click.version_option(version="0.1.0", prog_name="bank_synth")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
def cli(verbose: bool) -> None:
    """
    Bank Synth - Synthetic Data Generator for Banking Relational Models

    Train models on sample data and generate synthetic test data for Oracle and Hive.
    """
    setup_logging(verbose)


@cli.command()
@click.option(
    "--source",
    type=click.Choice(["oracle", "hive", "samples", "both", "auto"]),
    default="auto",
    help="Data source for training (auto uses queries + samples)",
)
@click.option(
    "--oracle_conn",
    type=str,
    default=None,
    help="Oracle connection string (user/pwd@host:port/service)",
)
@click.option(
    "--hive_spark",
    is_flag=True,
    help="Use Spark for Hive metastore access",
)
@click.option(
    "--schema",
    type=str,
    required=True,
    help="Schema/database name",
)
@click.option(
    "--tables_file",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="YAML file with table list and configuration",
)
@click.option(
    "--tables",
    type=str,
    default=None,
    help="Comma-separated list of table names (auto-discovered from queries if not provided)",
)
@click.option(
    "--sample_dir",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Directory containing sample Parquet/CSV files",
)
@click.option(
    "--sample_strategy",
    type=str,
    default="percent:1",
    help="Sampling strategy (percent:N, rows:N, or full)",
)
@click.option(
    "--queries",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to SQL queries file or directory (ETL/reporting queries for relationship discovery)",
)
@click.option(
    "--relationships",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="YAML file with relationship definitions (optional, auto-discovered from queries)",
)
@click.option(
    "--privacy_policy",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="YAML file with privacy policy",
)
@click.option(
    "--model_out",
    type=click.Path(path_type=Path),
    required=True,
    help="Output path for model pack",
)
@click.option(
    "--min_confidence",
    type=float,
    default=0.3,
    help="Minimum confidence threshold for auto-discovered relationships",
)
def train(
    source: str,
    oracle_conn: Optional[str],
    hive_spark: bool,
    schema: str,
    tables_file: Optional[Path],
    tables: Optional[str],
    sample_dir: Optional[Path],
    sample_strategy: str,
    queries: Optional[Path],
    relationships: Optional[Path],
    privacy_policy: Optional[Path],
    model_out: Path,
    min_confidence: float,
) -> None:
    """
    Train synthetic data models from sample data.

    Zero-intervention mode: Provide queries to auto-discover relationships.
    No manual relationship configuration needed!

    Examples:

        # Zero-intervention: Auto-discover from queries (RECOMMENDED)
        bank_synth train --source auto --schema CORE \\
            --queries ./etl_queries/ --sample_dir ./samples \\
            --model_out models/CORE_modelpack

        # Train from local sample files with manual relationships
        bank_synth train --source samples --schema CORE \\
            --sample_dir ./samples --tables_file configs/tables.yaml \\
            --relationships configs/relationships.yaml \\
            --model_out models/CORE_modelpack

        # Train from Oracle database with auto-discovery
        bank_synth train --source oracle --schema CORE \\
            --oracle_conn "user/pwd@localhost:1521/ORCL" \\
            --queries ./reporting_queries/ \\
            --model_out models/CORE_modelpack
    """
    from bank_synth.trainer import Trainer

    console.print("[bold blue]Bank Synth Training[/bold blue]")
    console.print(f"Source: {source}")
    console.print(f"Schema: {schema}")

    # Parse table list
    table_list: List[str] = []
    tables_config = None

    if tables_file:
        with open(tables_file, "r") as f:
            tables_config = yaml.safe_load(f)
            if isinstance(tables_config, dict) and "tables" in tables_config:
                table_list = tables_config["tables"]
            elif isinstance(tables_config, list):
                table_list = tables_config
    elif tables:
        table_list = [t.strip() for t in tables.split(",")]

    # Initialize extractors
    oracle_extractor = None
    hive_extractor = None

    if source in ("oracle", "both", "auto") and oracle_conn:
        from bank_synth.metadata import OracleMetadataExtractor
        oracle_extractor = OracleMetadataExtractor(oracle_conn)

    if source in ("hive", "both", "auto") and hive_spark:
        from bank_synth.metadata import HiveMetadataExtractor
        hive_extractor = HiveMetadataExtractor()

    # Use auto-discovery if queries are provided or source is "auto"
    use_auto_discovery = queries is not None or source == "auto"

    if use_auto_discovery:
        from bank_synth.discovery import AutoDiscoveryPipeline

        console.print("[cyan]Using auto-discovery mode (zero-intervention)[/cyan]")
        if queries:
            console.print(f"Queries: {queries}")

        pipeline = AutoDiscoveryPipeline(
            oracle_extractor=oracle_extractor,
            hive_extractor=hive_extractor,
            privacy_policy_file=privacy_policy,
        )

        # Resolve metadata using auto-discovery
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Auto-discovering relationships...", total=None)

            graph = pipeline.discover(
                schema=schema,
                queries=queries,
                tables=table_list if table_list else None,
                sample_dir=sample_dir,
                source="samples" if source == "auto" and sample_dir else source,
                min_confidence=min_confidence,
            )

            progress.update(task, completed=True)

        # Update table_list from discovered tables
        if not table_list:
            table_list = list(graph.tables.keys())

        privacy_policy_dict = pipeline.get_privacy_policy()

    else:
        # Traditional mode with manual relationships
        from bank_synth.metadata import MetadataResolver

        if not table_list:
            console.print("[red]Error: No tables specified. Use --tables, --tables_file, or provide --queries for auto-discovery[/red]")
            sys.exit(1)

        resolver = MetadataResolver(
            oracle_extractor=oracle_extractor,
            hive_extractor=hive_extractor,
            relationships_file=relationships,
            privacy_policy_file=privacy_policy,
        )

        # Resolve metadata
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Resolving metadata...", total=None)

            if source == "samples" and sample_dir:
                graph = resolver.resolve_from_samples(
                    schema=schema,
                    sample_dir=sample_dir,
                    tables_config=tables_config,
                )
            else:
                graph = resolver.resolve(
                    schema=schema,
                    tables=table_list,
                    source=source,
                )

            progress.update(task, completed=True)

        privacy_policy_dict = resolver.get_privacy_policy()

    console.print(f"Tables: {len(table_list)}")
    console.print(f"Resolved {len(graph.tables)} tables, {len(graph.relationships)} relationships")

    # Initialize trainer
    trainer = Trainer(
        relationship_graph=graph,
        privacy_policy=privacy_policy_dict,
    )

    # Load samples
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading samples...", total=None)

        if source == "samples" and sample_dir:
            trainer.load_samples(sample_dir, tables=table_list)
        elif source in ("oracle", "both") and oracle_extractor:
            # Sample from Oracle
            for table in table_list:
                try:
                    df = oracle_extractor.sample_table(schema, table, sample_strategy)
                    trainer.load_sample_dataframe(table, df)
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not sample {table}: {e}[/yellow]")
        elif source in ("hive", "both") and hive_extractor:
            # Sample from Hive
            for table in table_list:
                try:
                    df = hive_extractor.sample_table(schema, table, sample_strategy)
                    trainer.load_sample_dataframe(table, df)
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not sample {table}: {e}[/yellow]")

        progress.update(task, completed=True)

    # Train
    console.print("\n[bold]Training models...[/bold]")
    model_pack = trainer.train(
        schema_name=schema,
        sample_strategy=sample_strategy,
    )

    # Save model pack
    model_pack.save(model_out)
    console.print(f"\n[green]Model pack saved to: {model_out}[/green]")

    # Print summary
    table = Table(title="Training Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Tables Trained", str(len(model_pack.tables_trained)))
    table.add_row("Relationships", str(len(model_pack.relationship_graph.relationships)))
    table.add_row("Model Pack Location", str(model_out))

    console.print(table)


@cli.command()
@click.option(
    "--model",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to trained model pack",
)
@click.option(
    "--target_table",
    type=str,
    required=True,
    help="Target table to generate",
)
@click.option(
    "--target_rows",
    type=int,
    required=True,
    help="Number of rows to generate for target table",
)
@click.option(
    "--include_children",
    is_flag=True,
    default=False,
    help="Include child tables in generation",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducibility",
)
@click.option(
    "--output_formats",
    type=str,
    default="hive_orc,oracle",
    help="Comma-separated output formats (hive_orc, hive_parquet, oracle)",
)
@click.option(
    "--output_dir",
    type=click.Path(path_type=Path),
    default=Path("output"),
    help="Output directory",
)
@click.option(
    "--run_id",
    type=str,
    default=None,
    help="Custom run ID (default: timestamp)",
)
@click.option(
    "--table_rows",
    type=str,
    default=None,
    help="Override row counts for specific tables (format: TABLE1:N,TABLE2:M)",
)
@click.option(
    "--parent_scale",
    type=float,
    default=0.1,
    help="Scale factor for parent table row counts",
)
def generate(
    model: Path,
    target_table: str,
    target_rows: int,
    include_children: bool,
    seed: Optional[int],
    output_formats: str,
    output_dir: Path,
    run_id: Optional[str],
    table_rows: Optional[str],
    parent_scale: float,
) -> None:
    """
    Generate synthetic data from a trained model.

    Examples:

        # Generate 10,000 rows for TRANSACTIONS table
        bank_synth generate --model models/CORE_modelpack \\
            --target_table TRANSACTIONS --target_rows 10000 \\
            --output_dir output

        # Generate with specific seed for reproducibility
        bank_synth generate --model models/CORE_modelpack \\
            --target_table ACCOUNTS --target_rows 5000 \\
            --seed 42 --output_formats hive,oracle

        # Generate with custom row counts for parent tables
        bank_synth generate --model models/CORE_modelpack \\
            --target_table TRANSACTIONS --target_rows 10000 \\
            --table_rows "CUSTOMERS:1000,ACCOUNTS:2000"
    """
    from bank_synth.generator import Generator
    from bank_synth.output import OutputWriter
    from bank_synth.utils import QualityReporter

    console.print("[bold blue]Bank Synth Generation[/bold blue]")
    console.print(f"Model: {model}")
    console.print(f"Target: {target_table}")
    console.print(f"Rows: {target_rows:,}")

    # Parse output formats
    formats = [f.strip().lower() for f in output_formats.split(",")]

    # Parse table row overrides
    table_row_counts = {}
    if table_rows:
        for item in table_rows.split(","):
            if ":" in item:
                tbl, cnt = item.split(":")
                table_row_counts[tbl.strip().upper()] = int(cnt.strip())

    # Create generation config
    config = GenerationConfig(
        target_table=target_table,
        target_rows=target_rows,
        include_children=include_children,
        seed=seed,
        output_formats=formats,
        output_dir=output_dir,
        run_id=run_id,
        table_row_counts=table_row_counts,
        parent_scale_factor=parent_scale,
    )

    # Load model pack
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading model pack...", total=None)
        model_pack = ModelPack.load(model)
        progress.update(task, completed=True)

    console.print(f"Model pack version: {model_pack.version}")
    console.print(f"Tables in model: {len(model_pack.tables_trained)}")

    # Initialize generator
    generator = Generator(model_pack, config)

    # Generate data
    console.print("\n[bold]Generating data...[/bold]")
    generated_data = generator.generate()

    # Print generation summary
    table = Table(title="Generated Tables")
    table.add_column("Table", style="cyan")
    table.add_column("Rows", style="green", justify="right")
    table.add_column("Columns", style="yellow", justify="right")

    for tbl_name, df in generated_data.items():
        table.add_row(tbl_name, f"{len(df):,}", str(len(df.columns)))

    console.print(table)

    # Write outputs
    console.print("\n[bold]Writing outputs...[/bold]")
    writer = OutputWriter(config, model_pack)
    output_paths = writer.write(generated_data)

    # Generate quality report
    console.print("\n[bold]Generating quality report...[/bold]")
    reporter = QualityReporter(model_pack, generated_data)
    report = reporter.generate_report()
    json_path, md_path = reporter.save(writer.get_output_dir())

    # Print final summary
    console.print("\n[green]Generation complete![/green]")
    console.print(f"Output directory: {writer.get_output_dir()}")

    summary_table = Table(title="Output Summary")
    summary_table.add_column("Format", style="cyan")
    summary_table.add_column("Files", style="green", justify="right")

    for fmt, paths in output_paths.items():
        summary_table.add_row(fmt.upper(), str(len(paths)))

    summary_table.add_row("Reports", "2")
    console.print(summary_table)

    # Print referential integrity score
    ri_score = report["referential_integrity"]["integrity_score"]
    if ri_score == 1.0:
        console.print(f"\n[green]Referential Integrity: {ri_score:.0%}[/green]")
    else:
        console.print(f"\n[yellow]Referential Integrity: {ri_score:.0%}[/yellow]")
        console.print("See quality report for details on violations.")


@cli.command()
@click.option(
    "--schema",
    type=str,
    required=True,
    help="Schema/database name",
)
@click.option(
    "--queries",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to SQL queries file or directory (ETL/reporting queries)",
)
@click.option(
    "--sample_dir",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Directory containing sample Parquet/CSV files",
)
@click.option(
    "--oracle_conn",
    type=str,
    default=None,
    help="Oracle connection string for PK/FK metadata",
)
@click.option(
    "--hive_spark",
    is_flag=True,
    help="Use Spark for Hive metastore access",
)
@click.option(
    "--min_confidence",
    type=float,
    default=0.3,
    help="Minimum confidence threshold for relationships",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Output path for discovered relationships YAML",
)
def discover(
    schema: str,
    queries: Path,
    sample_dir: Optional[Path],
    oracle_conn: Optional[str],
    hive_spark: bool,
    min_confidence: float,
    output: Optional[Path],
) -> None:
    """
    Discover table relationships from SQL queries (zero-intervention).

    This command analyzes your ETL/reporting queries and database metadata
    to automatically discover table relationships without manual configuration.

    Examples:

        # Discover relationships from queries
        bank_synth discover --schema CORE --queries ./etl_queries/

        # Discover with sample data for metadata
        bank_synth discover --schema CORE \\
            --queries ./etl_queries/ --sample_dir ./samples

        # Discover with Oracle metadata
        bank_synth discover --schema CORE \\
            --queries ./etl_queries/ \\
            --oracle_conn "user/pwd@host:1521/SID"

        # Save discovered relationships to file
        bank_synth discover --schema CORE \\
            --queries ./etl_queries/ --output discovered_relationships.yaml
    """
    from bank_synth.discovery import AutoDiscoveryPipeline

    console.print("[bold blue]Bank Synth - Relationship Discovery[/bold blue]")
    console.print(f"Schema: {schema}")
    console.print(f"Queries: {queries}")

    # Initialize extractors
    oracle_extractor = None
    hive_extractor = None

    if oracle_conn:
        from bank_synth.metadata import OracleMetadataExtractor
        oracle_extractor = OracleMetadataExtractor(oracle_conn)
        console.print(f"Oracle: Connected")

    if hive_spark:
        from bank_synth.metadata import HiveMetadataExtractor
        hive_extractor = HiveMetadataExtractor()
        console.print(f"Hive: Connected via Spark")

    pipeline = AutoDiscoveryPipeline(
        oracle_extractor=oracle_extractor,
        hive_extractor=hive_extractor,
    )

    # Discover relationships
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing queries and discovering relationships...", total=None)

        graph = pipeline.discover(
            schema=schema,
            queries=queries,
            sample_dir=sample_dir,
            min_confidence=min_confidence,
        )

        progress.update(task, completed=True)

    # Display results
    console.print(f"\n[green]Discovery complete![/green]")

    # Tables table
    tables_table = Table(title="Discovered Tables")
    tables_table.add_column("Table", style="cyan")
    tables_table.add_column("Columns", style="green", justify="right")
    tables_table.add_column("PK", style="yellow")

    for table_name in sorted(graph.tables.keys()):
        table_meta = graph.get_table(table_name)
        pk_cols = [c.name for c in table_meta.columns if c.is_primary_key]
        tables_table.add_row(
            table_name,
            str(len(table_meta.columns)),
            ", ".join(pk_cols) if pk_cols else "-",
        )

    console.print(tables_table)

    # Relationships table
    if graph.relationships:
        rel_table = Table(title="Discovered Relationships")
        rel_table.add_column("Parent", style="cyan")
        rel_table.add_column("Parent Column", style="green")
        rel_table.add_column("Child", style="yellow")
        rel_table.add_column("Child Column", style="magenta")
        rel_table.add_column("Cardinality", style="blue")

        for rel in graph.relationships:
            rel_table.add_row(
                rel.parent_table,
                ", ".join(rel.parent_columns),
                rel.child_table,
                ", ".join(rel.child_columns),
                rel.cardinality,
            )

        console.print(rel_table)
    else:
        console.print("\n[yellow]No relationships discovered.[/yellow]")
        console.print("Try providing more queries or ensure queries have JOINs.")

    # Save to file if requested
    if output:
        output = Path(output)
        relationships_data = {
            "schema": schema,
            "discovered_from": str(queries),
            "min_confidence": min_confidence,
            "tables": list(graph.tables.keys()),
            "relationships": [rel.to_dict() for rel in graph.relationships],
        }

        with open(output, "w") as f:
            yaml.dump(relationships_data, f, default_flow_style=False)

        console.print(f"\n[green]Saved relationships to: {output}[/green]")


@cli.command()
@click.option(
    "--model",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to trained model pack",
)
def info(model: Path) -> None:
    """
    Display information about a trained model pack.

    Example:

        bank_synth info --model models/CORE_modelpack
    """
    console.print("[bold blue]Model Pack Information[/bold blue]")

    model_pack = ModelPack.load(model)

    # Basic info
    info_table = Table(title="Model Pack Details")
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="green")

    info_table.add_row("Version", model_pack.version)
    info_table.add_row("Created", model_pack.created_at)
    info_table.add_row("Schema", model_pack.schema_name or "N/A")
    info_table.add_row("Sample Strategy", model_pack.sample_strategy or "N/A")
    info_table.add_row("Tables Trained", str(len(model_pack.tables_trained)))
    info_table.add_row("Relationships", str(len(model_pack.relationship_graph.relationships)))

    console.print(info_table)

    # Tables
    tables_table = Table(title="Trained Tables")
    tables_table.add_column("Table", style="cyan")
    tables_table.add_column("Rows (Sample)", style="green", justify="right")
    tables_table.add_column("Columns", style="yellow", justify="right")

    for table_name in sorted(model_pack.tables_trained):
        stats = model_pack.table_stats.get(table_name)
        row_count = stats.row_count if stats else "N/A"
        col_count = len(stats.column_stats) if stats else "N/A"
        tables_table.add_row(table_name, str(row_count), str(col_count))

    console.print(tables_table)

    # Relationships
    if model_pack.relationship_graph.relationships:
        rel_table = Table(title="Relationships")
        rel_table.add_column("Name", style="cyan")
        rel_table.add_column("Parent", style="green")
        rel_table.add_column("Child", style="yellow")
        rel_table.add_column("Cardinality", style="magenta")

        for rel in model_pack.relationship_graph.relationships:
            rel_table.add_row(
                rel.name,
                f"{rel.parent_table}.{','.join(rel.parent_columns)}",
                f"{rel.child_table}.{','.join(rel.child_columns)}",
                rel.cardinality,
            )

        console.print(rel_table)


if __name__ == "__main__":
    cli()
