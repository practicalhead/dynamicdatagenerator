-- @name: customer_account_summary
-- ETL query for customer account summary report
SELECT
    c.customer_id,
    c.first_name,
    c.last_name,
    c.email,
    COUNT(a.account_id) as num_accounts,
    SUM(a.balance) as total_balance
FROM customers c
LEFT JOIN accounts a ON c.customer_id = a.customer_id
GROUP BY c.customer_id, c.first_name, c.last_name, c.email;

-- @name: transaction_details
-- ETL query for transaction reporting
SELECT
    t.transaction_id,
    t.transaction_date,
    t.amount,
    t.description,
    a.account_number,
    a.account_type_id,
    c.customer_id,
    c.first_name || ' ' || c.last_name as customer_name,
    tt.type_name as transaction_type
FROM transactions t
INNER JOIN accounts a ON t.account_id = a.account_id
INNER JOIN customers c ON a.customer_id = c.customer_id
LEFT JOIN transaction_types tt ON t.transaction_type_id = tt.transaction_type_id
WHERE t.transaction_date >= TRUNC(SYSDATE) - 30;

-- @name: account_type_report
-- Report on accounts by type
SELECT
    at.account_type_id,
    at.type_name,
    at.description,
    COUNT(a.account_id) as account_count,
    SUM(a.balance) as total_balance,
    AVG(a.balance) as avg_balance
FROM account_types at
LEFT JOIN accounts a ON at.account_type_id = a.account_type_id
GROUP BY at.account_type_id, at.type_name, at.description;

-- @name: customer_address_report
-- Customer addresses for mailing
SELECT
    c.customer_id,
    c.first_name,
    c.last_name,
    ca.address_type,
    ca.street_address,
    ca.city,
    ca.state,
    ca.postal_code,
    ca.country
FROM customers c
INNER JOIN customer_addresses ca ON c.customer_id = ca.customer_id
WHERE ca.is_primary = 1;
