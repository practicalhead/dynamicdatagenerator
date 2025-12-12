# Bank Synth - Synthetic Data Generator

**DDG (Dynamic Data Generator)** is a model-trained framework that generates realistic, relationally consistent synthetic test data for banking systems without copying production data.

## Features

- **Two-Phase Workflow**: Train on governed samples, generate in lower environments
- **Relational Consistency**: 100% FK consistency within generated data closure
- **Multi-Platform**: Supports Oracle and Hive output formats
- **Privacy-Aware**: Column-level privacy controls (PII/sensitive handling)
- **Deterministic**: Reproducible generation with seed support
- **SDV Integration**: Uses Synthetic Data Vault for distribution learning

## Installation

```bash
pip install -e .
```

## Quick Start

### 1. Generate Sample Data (Optional)

```bash
cd samples
python generate_samples.py
```

### 2. Train a Model

Train from local sample files:

```bash
bank_synth train \
  --source samples \
  --schema CORE \
  --sample_dir ./samples \
  --tables_file configs/tables.yaml \
  --relationships configs/relationships.yaml \
  --privacy_policy configs/privacy.yaml \
  --model_out models/CORE_modelpack
```

Or train from Oracle database:

```bash
bank_synth train \
  --source oracle \
  --oracle_conn "user/pwd@host:1521/service" \
  --schema CORE \
  --tables CUSTOMERS,ACCOUNTS,TRANSACTIONS \
  --sample_strategy "percent:1" \
  --relationships configs/relationships.yaml \
  --model_out models/CORE_modelpack
```

### 3. Generate Synthetic Data

```bash
bank_synth generate \
  --model models/CORE_modelpack \
  --target_table TRANSACTIONS \
  --target_rows 10000 \
  --seed 42 \
  --output_dir output
```

### 4. View Model Information

```bash
bank_synth info --model models/CORE_modelpack
```

## Architecture

### Components

1. **Metadata Resolver** - Introspects Oracle/Hive catalogs or sample files
2. **Trainer** - Learns distributions using SDV-style modeling
3. **Generator** - Produces data with topological ordering and FK consistency
4. **Output Handlers** - Writes Hive (Parquet) and Oracle (CSV/SQL) formats

### Dependency Resolution

When you request generation for a target table, bank_synth automatically:

1. Identifies all parent tables via FK relationships
2. Determines generation order via topological sort
3. Generates parent tables first
4. Ensures FK values come from generated parent keys

## Configuration Files

### tables.yaml

Defines which tables to train and column-level configuration:

```yaml
tables:
  - CUSTOMERS
  - ACCOUNTS
  - TRANSACTIONS

table_config:
  CUSTOMERS:
    primary_key:
      - CUSTOMER_ID
    columns:
      email:
        faker_provider: email
      status:
        allowed_values: ["ACTIVE", "INACTIVE"]
```

### relationships.yaml

Defines PK/FK relationships (overrides database-discovered relationships):

```yaml
relationships:
  - name: fk_accounts_customer
    parent_table: CUSTOMERS
    parent_columns: [CUSTOMER_ID]
    child_table: ACCOUNTS
    child_columns: [CUSTOMER_ID]
    cardinality: "1:N"
```

### privacy.yaml

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
```

Privacy levels:
- `public` - No restrictions
- `internal` - Learn distribution, mask in reports
- `sensitive` - Format-preserving generation
- `pii` - Generate using Faker providers
- `restricted` - Drop column entirely

## Output Structure

```
output/<run_id>/
├── hive/
│   ├── customers/
│   │   └── customers.parquet
│   ├── accounts/
│   │   └── accounts.parquet
│   └── transactions/
│       └── transactions.parquet
├── oracle/
│   ├── customers.csv
│   ├── accounts.csv
│   └── transactions.csv
├── ddl/
│   ├── hive_load.sql
│   └── oracle_load.sql
├── report/
│   ├── quality.json
│   └── quality.md
└── manifest.json
```

## CLI Reference

### train

```bash
bank_synth train [OPTIONS]

Options:
  --source [oracle|hive|samples|both]  Data source for training
  --oracle_conn TEXT                   Oracle connection string
  --hive_spark                         Use Spark for Hive access
  --schema TEXT                        Schema/database name [required]
  --tables_file PATH                   YAML file with table configuration
  --tables TEXT                        Comma-separated table names
  --sample_dir PATH                    Directory with sample files
  --sample_strategy TEXT               Sampling strategy (percent:N, rows:N)
  --relationships PATH                 YAML file with relationships
  --privacy_policy PATH                YAML file with privacy policy
  --model_out PATH                     Output path for model pack [required]
```

### generate

```bash
bank_synth generate [OPTIONS]

Options:
  --model PATH              Path to trained model pack [required]
  --target_table TEXT       Target table to generate [required]
  --target_rows INTEGER     Number of rows for target table [required]
  --include_children        Include child tables in generation
  --seed INTEGER            Random seed for reproducibility
  --output_formats TEXT     Comma-separated formats (hive,oracle)
  --output_dir PATH         Output directory
  --run_id TEXT             Custom run identifier
  --table_rows TEXT         Override row counts (TABLE:N,TABLE:N)
  --parent_scale FLOAT      Scale factor for parent tables
```

### info

```bash
bank_synth info --model PATH
```

## Requirements

- Python 3.9+
- pandas >= 2.0
- pyarrow >= 14.0
- sdv >= 1.10
- faker >= 22.0
- networkx >= 3.0
- click >= 8.0
- pyyaml >= 6.0
- rich >= 13.0

Optional:
- oracledb >= 2.0 (for Oracle support)
- pyspark >= 3.5 (for Hive support)

## License

MIT License
