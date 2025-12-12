# Synthetic Data Generator

**DDG (Dynamic Data Generator)** is a production-grade framework for generating massive, relationally consistent synthetic test data for banking systems using PySpark distributed computing.

## Key Features

- **Zero-Intervention Workflow**: Auto-discovers table relationships from ETL/reporting queries - no manual configuration required
- **PySpark Parallel Generation**: Generate millions/billions of rows using distributed computing across Spark clusters
- **Hive ORC Output**: Write directly to Hive tables in ORC format for optimal query performance
- **100% Referential Integrity**: FK constraints automatically enforced via topological ordering
- **Privacy-Aware**: Column-level privacy controls (PII masking, format-preserving generation)
- **Enterprise Ready**: Supports Oracle, Hive, YARN, and Kubernetes deployments

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BANK SYNTH PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │   STAGE 1    │    │   STAGE 2    │    │   STAGE 3    │    │  STAGE 4  │ │
│  │  DISCOVERY   │───▶│   TRAINING   │───▶│  GENERATION  │───▶│  OUTPUT   │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘ │
│        │                    │                   │                   │       │
│   ┌────▼────┐         ┌────▼────┐         ┌────▼────┐         ┌────▼────┐  │
│   │ Queries │         │ Samples │         │  Spark  │         │   ORC   │  │
│   │ PK/FK   │         │  Stats  │         │Parallel │         │ Parquet │  │
│   │Patterns │         │ Models  │         │  Gen    │         │   CSV   │  │
│   └─────────┘         └─────────┘         └─────────┘         └─────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install -e .

# For PySpark support (required for large-scale generation)
pip install pyspark>=3.5
```

## Quick Start

### Zero-Intervention Mode (Recommended)

The simplest way to use Bank Synth - just provide your ETL/reporting queries and sample data:

```bash
# Step 1: Auto-discover relationships from queries
bank_synth discover --schema CORE --queries ./etl_queries/

# Step 2: Train with auto-discovery
bank_synth train --source auto --schema CORE \
    --queries ./etl_queries/ --sample_dir ./samples \
    --model_out models/CORE_modelpack

# Step 3: Generate massive data with Spark
bank_synth generate --model models/CORE_modelpack \
    --target_table TRANSACTIONS --target_rows 10000000 \
    --spark --hive_database test_data
```

---

## Stage 1: Relationship Discovery

Bank Synth automatically discovers table relationships without manual configuration using three sources:

### 1.1 Query-Based Discovery (Primary)

Parses your ETL and reporting queries to extract JOIN relationships:

```sql
-- Example: etl_queries/customer_report.sql
-- @name: customer_account_summary
SELECT c.customer_id, c.first_name, COUNT(a.account_id)
FROM customers c
LEFT JOIN accounts a ON c.customer_id = a.customer_id
GROUP BY c.customer_id, c.first_name;
```

The parser automatically extracts:
- Tables involved: `CUSTOMERS`, `ACCOUNTS`
- Relationship: `CUSTOMERS.CUSTOMER_ID` → `ACCOUNTS.CUSTOMER_ID`
- Join type: `LEFT` (relationship is optional)

### 1.2 PK/FK Metadata (Authoritative)

When database access is available, extracts constraints from:
- Oracle: `ALL_CONSTRAINTS`, `ALL_CONS_COLUMNS`
- Hive: Metastore table properties

### 1.3 Column Pattern Inference (Fallback)

Infers relationships from naming conventions:
- `customer_id` → `customers.customer_id`
- `account_type_id` → `account_types.account_type_id`

### Discovery CLI

```bash
# Discover relationships from queries
bank_synth discover --schema CORE --queries ./etl_queries/

# With database metadata
bank_synth discover --schema CORE --queries ./etl_queries/ \
    --oracle_conn "user/pwd@host:1521/SID"

# Save discovered relationships
bank_synth discover --schema CORE --queries ./etl_queries/ \
    --output discovered_relationships.yaml
```

---

## Stage 2: Model Training

Training analyzes sample data to learn:
- Column distributions (mean, std, min, max)
- Categorical value frequencies
- Null patterns
- Cross-column correlations

### Training Modes

#### Auto Mode (Zero-Intervention)
```bash
bank_synth train --source auto --schema CORE \
    --queries ./etl_queries/ --sample_dir ./samples \
    --model_out models/CORE_modelpack
```

#### From Sample Files
```bash
bank_synth train --source samples --schema CORE \
    --sample_dir ./samples --tables_file configs/tables.yaml \
    --model_out models/CORE_modelpack
```

#### From Oracle Database
```bash
bank_synth train --source oracle --schema CORE \
    --oracle_conn "user/pwd@host:1521/service" \
    --queries ./etl_queries/ \
    --sample_strategy "percent:1" \
    --model_out models/CORE_modelpack
```

#### From Hive (Spark)
```bash
bank_synth train --source hive --schema CORE \
    --hive_spark --queries ./etl_queries/ \
    --sample_strategy "rows:100000" \
    --model_out models/CORE_modelpack
```

### Model Pack Output

Training produces a portable `ModelPack` containing:
```
models/CORE_modelpack/
├── metadata.json          # Version, schema, training config
├── relationship_graph.json # Tables and relationships
├── table_stats.json       # Column statistics
├── synthesizers.pkl       # Trained models (serialized)
└── encoders.json          # Categorical encoders
```

---

## Stage 3: Data Generation

### Generation Modes

#### Standard Mode (Pandas)
For datasets up to ~1M rows:

```bash
bank_synth generate --model models/CORE_modelpack \
    --target_table TRANSACTIONS --target_rows 100000 \
    --seed 42 --output_dir output
```

#### Spark Mode (Parallel)
For massive datasets (millions to billions of rows):

```bash
bank_synth generate --model models/CORE_modelpack \
    --target_table TRANSACTIONS --target_rows 100000000 \
    --spark --hive_database test_data
```

### How Generation Works

1. **Dependency Resolution**: Identifies all parent tables via FK relationships
2. **Topological Sort**: Determines generation order (parents before children)
3. **Parallel Generation**: Uses Spark to generate rows across executors
4. **FK Constraint Application**: Samples FK values from generated parent keys
5. **Privacy Transformation**: Applies masking/generation for sensitive columns

### Generation Flow

```
Target: TRANSACTIONS (10M rows)
          │
          ▼
┌─────────────────────────────────────┐
│     Dependency Resolution           │
│  TRANSACTIONS depends on:           │
│    - ACCOUNTS (FK: account_id)      │
│    - TRANSACTION_TYPES (FK: type_id)│
│  ACCOUNTS depends on:               │
│    - CUSTOMERS (FK: customer_id)    │
│    - ACCOUNT_TYPES (FK: type_id)    │
└─────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│     Topological Sort                │
│  Generation Order:                  │
│    1. CUSTOMERS (1M rows)           │
│    2. ACCOUNT_TYPES (10 rows)       │
│    3. TRANSACTION_TYPES (20 rows)   │
│    4. ACCOUNTS (2M rows)            │
│    5. TRANSACTIONS (10M rows)       │
└─────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│     Spark Parallel Generation       │
│  - 100 partitions                   │
│  - 100K rows per partition          │
│  - Broadcast parent keys for FK     │
└─────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────┐
│     Write to Hive ORC               │
│  test_data.transactions             │
│  test_data.accounts                 │
│  test_data.customers                │
│  ...                                │
└─────────────────────────────────────┘
```

### Spark Configuration

Bank Synth auto-configures Spark for optimal performance:

| Target Rows | Partitions | Rows/Partition |
|-------------|------------|----------------|
| 100K | 1 | 100K |
| 1M | 2 | 500K |
| 10M | 13 | 750K |
| 100M | 134 | 750K |
| 1B | 1,334 | 750K |

Manual configuration:
```python
from bank_synth.spark import SparkSessionManager

manager = SparkSessionManager(
    app_name="BankSynth",
    mode="yarn",  # or "kubernetes", "local"
    custom_configs={
        "spark.executor.instances": "50",
        "spark.executor.memory": "16g",
        "spark.sql.shuffle.partitions": "2000",
    }
)
```

---

## Stage 4: Output

### Output Formats

#### Hive ORC (Recommended)
```bash
bank_synth generate ... --output_formats hive_orc
```

Writes optimized ORC files with:
- Snappy compression (default)
- Predicate pushdown support
- Type-specific encodings

#### Hive Parquet
```bash
bank_synth generate ... --output_formats hive_parquet
```

#### Oracle CSV
```bash
bank_synth generate ... --output_formats oracle
```

Generates:
- CSV data files
- SQL*Loader control files

### Direct Hive Write (Spark Mode)

```bash
bank_synth generate --model models/CORE_modelpack \
    --target_table TRANSACTIONS --target_rows 10000000 \
    --spark --hive_database test_data
```

Creates Hive tables directly:
```sql
test_data.customers
test_data.accounts
test_data.transactions
...
```

### Output Structure

```
output/<run_id>/
├── hive_orc/                    # ORC format files
│   ├── customers/
│   │   └── customers.orc
│   ├── accounts/
│   │   └── accounts.orc
│   └── transactions/
│       └── transactions.orc
├── hive_parquet/               # Parquet format files
│   └── ...
├── oracle/                     # CSV files
│   ├── customers.csv
│   ├── accounts.csv
│   └── transactions.csv
├── ddl/                        # DDL scripts
│   ├── hive_orc_tables.sql
│   ├── hive_parquet_tables.sql
│   └── customers.ctl           # SQL*Loader control
├── report/                     # Quality reports
│   ├── quality.json
│   └── quality.md
└── manifest.json               # Generation metadata
```

---

## Configuration Files

### queries/ Directory

Place your ETL and reporting SQL files here for auto-discovery:

```sql
-- queries/etl_queries.sql
-- @name: customer_account_summary
SELECT c.*, a.*
FROM customers c
JOIN accounts a ON c.customer_id = a.customer_id;

-- @name: transaction_details
SELECT t.*, a.account_number, c.first_name
FROM transactions t
JOIN accounts a ON t.account_id = a.account_id
JOIN customers c ON a.customer_id = c.customer_id;
```

### privacy.yaml (Optional)

Controls privacy handling for sensitive columns:

```yaml
tables:
  CUSTOMERS:
    ssn:
      level: pii
      faker_provider: ssn
    email:
      level: pii
      faker_provider: email
    date_of_birth:
      level: sensitive
      format_pattern: "9999-99-99"

defaults:
  level: internal
```

Privacy levels:
| Level | Behavior |
|-------|----------|
| `public` | No restrictions |
| `internal` | Learn distribution, mask in reports |
| `sensitive` | Format-preserving generation only |
| `pii` | Generate using Faker providers |
| `restricted` | Drop column entirely |

---

## CLI Reference

### discover

Discover relationships from SQL queries:

```bash
bank_synth discover [OPTIONS]

Options:
  --schema TEXT          Schema name [required]
  --queries PATH         SQL queries file/directory [required]
  --sample_dir PATH      Sample data directory
  --oracle_conn TEXT     Oracle connection for PK/FK
  --hive_spark          Use Spark for Hive metadata
  --min_confidence FLOAT Minimum confidence threshold (default: 0.3)
  --output PATH         Save discovered relationships to file
```

### train

Train models from sample data:

```bash
bank_synth train [OPTIONS]

Options:
  --source [oracle|hive|samples|both|auto]  Data source (default: auto)
  --schema TEXT                              Schema name [required]
  --queries PATH                             SQL queries for discovery
  --tables TEXT                              Comma-separated table names
  --tables_file PATH                         YAML with table config
  --sample_dir PATH                          Sample data directory
  --sample_strategy TEXT                     Sampling (percent:N, rows:N)
  --oracle_conn TEXT                         Oracle connection string
  --hive_spark                               Use Spark for Hive
  --relationships PATH                       Manual relationships YAML
  --privacy_policy PATH                      Privacy policy YAML
  --min_confidence FLOAT                     Auto-discovery confidence
  --model_out PATH                           Output model path [required]
```

### generate

Generate synthetic data:

```bash
bank_synth generate [OPTIONS]

Options:
  --model PATH              Model pack path [required]
  --target_table TEXT       Target table [required]
  --target_rows INTEGER     Number of rows [required]
  --spark                   Use Spark for parallel generation
  --hive_database TEXT      Write directly to Hive database
  --include_children        Include child tables
  --seed INTEGER            Random seed
  --output_formats TEXT     Formats (hive_orc,hive_parquet,oracle)
  --output_dir PATH         Output directory
  --run_id TEXT             Custom run ID
  --table_rows TEXT         Override counts (TABLE:N,TABLE:N)
  --parent_scale FLOAT      Parent table scale factor
```

### info

Display model pack information:

```bash
bank_synth info --model PATH
```

---

## Python API

### Zero-Intervention Discovery

```python
from bank_synth import auto_discover

# Discover relationships from queries
graph = auto_discover(
    schema="CORE",
    queries="./etl_queries/",
    sample_dir="./samples/",
)

print(f"Tables: {list(graph.tables.keys())}")
print(f"Relationships: {len(graph.relationships)}")
```

### Spark Generation

```python
from bank_synth import ModelPack, GenerationConfig
from bank_synth.spark import SparkGenerator

# Load model
model_pack = ModelPack.load("models/CORE_modelpack")

# Configure generation
config = GenerationConfig(
    target_table="TRANSACTIONS",
    target_rows=10_000_000,
    seed=42,
)

# Generate with Spark
generator = SparkGenerator(model_pack, config)
spark_data = generator.generate()

# Write to Hive
generator.write_to_hive(
    database="test_data",
    format="orc",
)
```

### Custom Spark Configuration

```python
from bank_synth.spark import SparkSessionManager

manager = SparkSessionManager(
    app_name="MyDataGenerator",
    mode="yarn",
    custom_configs={
        "spark.executor.instances": "100",
        "spark.executor.memory": "16g",
        "spark.executor.cores": "4",
    }
)

spark = manager.get_session()
```

---

## Requirements

**Core:**
- Python 3.9+
- pandas >= 2.0
- pyarrow >= 14.0
- faker >= 22.0
- networkx >= 3.0
- click >= 8.0
- pyyaml >= 6.0
- rich >= 13.0

**Optional:**
- pyspark >= 3.5 (for parallel generation)
- oracledb >= 2.0 (for Oracle support)
- sdv >= 1.10 (for advanced synthesizers)

---

## Performance Benchmarks

| Rows | Mode | Time | Cluster |
|------|------|------|---------|
| 100K | Pandas | ~30s | Local |
| 1M | Pandas | ~5min | Local |
| 10M | Spark | ~2min | 10 executors |
| 100M | Spark | ~15min | 50 executors |
| 1B | Spark | ~2hr | 100 executors |

---

## License

MIT License
