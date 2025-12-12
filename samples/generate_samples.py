#!/usr/bin/env python3
"""
Generate sample data files for testing bank_synth training.

This script creates small sample datasets in Parquet format that can be
used to test the training pipeline without requiring database access.
"""

import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

# Initialize Faker with seed for reproducibility
fake = Faker()
Faker.seed(42)
random.seed(42)
np.random.seed(42)

# Output directory
OUTPUT_DIR = Path(__file__).parent


def generate_account_types() -> pd.DataFrame:
    """Generate reference data for account types."""
    return pd.DataFrame({
        "ACCOUNT_TYPE_ID": [1, 2, 3, 4, 5],
        "TYPE_NAME": ["Checking", "Savings", "Money Market", "Certificate of Deposit", "IRA"],
        "TYPE_CODE": ["CHK", "SAV", "MMA", "CD", "IRA"],
    })


def generate_transaction_types() -> pd.DataFrame:
    """Generate reference data for transaction types."""
    return pd.DataFrame({
        "TRANSACTION_TYPE_ID": [1, 2, 3, 4, 5, 6],
        "TYPE_NAME": ["Deposit", "Withdrawal", "Transfer", "Payment", "Fee", "Interest"],
        "TYPE_CODE": ["DEP", "WDR", "TRF", "PMT", "FEE", "INT"],
    })


def generate_branches(n: int = 20) -> pd.DataFrame:
    """Generate branch reference data."""
    branches = []
    for i in range(1, n + 1):
        branches.append({
            "BRANCH_ID": i,
            "BRANCH_NAME": f"{fake.city()} Branch",
            "BRANCH_CODE": f"BR{i:03d}",
            "ADDRESS": fake.street_address(),
            "CITY": fake.city(),
            "STATE": fake.state_abbr(),
            "ZIP_CODE": fake.zipcode(),
        })
    return pd.DataFrame(branches)


def generate_customers(n: int = 500) -> pd.DataFrame:
    """Generate customer sample data."""
    customers = []
    statuses = ["ACTIVE", "INACTIVE", "SUSPENDED", "CLOSED"]
    status_weights = [0.85, 0.08, 0.02, 0.05]

    for i in range(1, n + 1):
        created = fake.date_between(start_date="-10y", end_date="-1y")
        customers.append({
            "CUSTOMER_ID": i,
            "CUSTOMER_NAME": fake.name(),
            "SSN": fake.ssn(),
            "DATE_OF_BIRTH": fake.date_of_birth(minimum_age=18, maximum_age=90),
            "EMAIL": fake.email(),
            "PHONE": fake.phone_number(),
            "CREATED_DATE": datetime.combine(created, datetime.min.time()),
            "STATUS": random.choices(statuses, weights=status_weights)[0],
        })
    return pd.DataFrame(customers)


def generate_accounts(customers_df: pd.DataFrame, n: int = 1000) -> pd.DataFrame:
    """Generate account sample data."""
    accounts = []
    statuses = ["OPEN", "CLOSED", "FROZEN"]
    status_weights = [0.90, 0.08, 0.02]

    customer_ids = customers_df["CUSTOMER_ID"].tolist()

    for i in range(1, n + 1):
        opened = fake.date_between(start_date="-8y", end_date="-1m")
        accounts.append({
            "ACCOUNT_ID": i,
            "CUSTOMER_ID": random.choice(customer_ids),
            "ACCOUNT_TYPE_ID": random.randint(1, 5),
            "ACCOUNT_NUMBER": f"{random.randint(1000,9999)}-{random.randint(1000,9999)}-{random.randint(1000,9999)}-{random.randint(1000,9999)}",
            "BALANCE": round(random.uniform(0, 100000), 2),
            "OPENED_DATE": opened,
            "STATUS": random.choices(statuses, weights=status_weights)[0],
        })
    return pd.DataFrame(accounts)


def generate_transactions(accounts_df: pd.DataFrame, n: int = 5000) -> pd.DataFrame:
    """Generate transaction sample data."""
    transactions = []
    account_ids = accounts_df["ACCOUNT_ID"].tolist()

    for i in range(1, n + 1):
        tx_date = fake.date_time_between(start_date="-2y", end_date="now")
        tx_type = random.randint(1, 6)

        # Amount varies by transaction type
        if tx_type in [1, 2]:  # Deposit/Withdrawal
            amount = round(random.uniform(10, 5000), 2)
        elif tx_type == 3:  # Transfer
            amount = round(random.uniform(100, 10000), 2)
        elif tx_type == 4:  # Payment
            amount = round(random.uniform(25, 2000), 2)
        elif tx_type == 5:  # Fee
            amount = round(random.uniform(5, 50), 2)
        else:  # Interest
            amount = round(random.uniform(0.01, 100), 2)

        transactions.append({
            "TRANSACTION_ID": i,
            "ACCOUNT_ID": random.choice(account_ids),
            "TRANSACTION_TYPE_ID": tx_type,
            "AMOUNT": amount,
            "TRANSACTION_DATE": tx_date,
            "DESCRIPTION": fake.sentence(nb_words=6),
            "REFERENCE_NUMBER": f"REF-{random.randint(100000000, 999999999)}",
        })
    return pd.DataFrame(transactions)


def generate_customer_addresses(customers_df: pd.DataFrame) -> pd.DataFrame:
    """Generate customer address sample data."""
    addresses = []
    address_types = ["HOME", "WORK", "MAILING"]
    address_id = 1

    for customer_id in customers_df["CUSTOMER_ID"]:
        # Each customer gets 1-3 addresses
        n_addresses = random.choices([1, 2, 3], weights=[0.5, 0.35, 0.15])[0]

        for j in range(n_addresses):
            addresses.append({
                "ADDRESS_ID": address_id,
                "CUSTOMER_ID": customer_id,
                "ADDRESS_TYPE": address_types[j] if j < len(address_types) else "HOME",
                "STREET_ADDRESS": fake.street_address(),
                "CITY": fake.city(),
                "STATE": fake.state_abbr(),
                "ZIP_CODE": fake.zipcode(),
                "IS_PRIMARY": j == 0,
            })
            address_id += 1

    return pd.DataFrame(addresses)


def main():
    """Generate all sample data files."""
    print("Generating sample data for bank_synth testing...")

    # Generate reference tables
    print("  - Account Types")
    account_types_df = generate_account_types()
    account_types_df.to_parquet(OUTPUT_DIR / "ACCOUNT_TYPES.parquet")

    print("  - Transaction Types")
    transaction_types_df = generate_transaction_types()
    transaction_types_df.to_parquet(OUTPUT_DIR / "TRANSACTION_TYPES.parquet")

    print("  - Branches")
    branches_df = generate_branches()
    branches_df.to_parquet(OUTPUT_DIR / "BRANCHES.parquet")

    # Generate core tables
    print("  - Customers")
    customers_df = generate_customers(500)
    customers_df.to_parquet(OUTPUT_DIR / "CUSTOMERS.parquet")

    print("  - Accounts")
    accounts_df = generate_accounts(customers_df, 1000)
    accounts_df.to_parquet(OUTPUT_DIR / "ACCOUNTS.parquet")

    print("  - Transactions")
    transactions_df = generate_transactions(accounts_df, 5000)
    transactions_df.to_parquet(OUTPUT_DIR / "TRANSACTIONS.parquet")

    print("  - Customer Addresses")
    addresses_df = generate_customer_addresses(customers_df)
    addresses_df.to_parquet(OUTPUT_DIR / "CUSTOMER_ADDRESSES.parquet")

    print(f"\nGenerated sample files in: {OUTPUT_DIR}")
    print("\nSummary:")
    print(f"  - ACCOUNT_TYPES: {len(account_types_df)} rows")
    print(f"  - TRANSACTION_TYPES: {len(transaction_types_df)} rows")
    print(f"  - BRANCHES: {len(branches_df)} rows")
    print(f"  - CUSTOMERS: {len(customers_df)} rows")
    print(f"  - ACCOUNTS: {len(accounts_df)} rows")
    print(f"  - TRANSACTIONS: {len(transactions_df)} rows")
    print(f"  - CUSTOMER_ADDRESSES: {len(addresses_df)} rows")


if __name__ == "__main__":
    main()
