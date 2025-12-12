-- @name: daily_transaction_summary
-- Daily transaction summary for dashboard
SELECT
    TRUNC(t.transaction_date) as trans_date,
    tt.type_name,
    COUNT(*) as transaction_count,
    SUM(t.amount) as total_amount,
    AVG(t.amount) as avg_amount
FROM transactions t
JOIN transaction_types tt ON t.transaction_type_id = tt.transaction_type_id
GROUP BY TRUNC(t.transaction_date), tt.type_name
ORDER BY trans_date DESC;

-- @name: high_value_customers
-- High value customer report
SELECT
    c.customer_id,
    c.first_name,
    c.last_name,
    c.email,
    c.phone,
    c.status,
    SUM(a.balance) as total_balance,
    COUNT(DISTINCT a.account_id) as num_accounts,
    MAX(t.transaction_date) as last_transaction
FROM customers c
JOIN accounts a ON c.customer_id = a.customer_id
LEFT JOIN transactions t ON a.account_id = t.account_id
GROUP BY c.customer_id, c.first_name, c.last_name, c.email, c.phone, c.status
HAVING SUM(a.balance) > 100000
ORDER BY total_balance DESC;

-- @name: account_activity
-- Account activity summary
SELECT
    a.account_id,
    a.account_number,
    at.type_name as account_type,
    c.first_name || ' ' || c.last_name as owner_name,
    a.balance,
    a.status,
    COUNT(t.transaction_id) as num_transactions,
    SUM(CASE WHEN t.amount > 0 THEN t.amount ELSE 0 END) as total_credits,
    SUM(CASE WHEN t.amount < 0 THEN ABS(t.amount) ELSE 0 END) as total_debits
FROM accounts a
JOIN account_types at ON a.account_type_id = at.account_type_id
JOIN customers c ON a.customer_id = c.customer_id
LEFT JOIN transactions t ON a.account_id = t.account_id
    AND t.transaction_date >= ADD_MONTHS(SYSDATE, -12)
GROUP BY a.account_id, a.account_number, at.type_name,
         c.first_name, c.last_name, a.balance, a.status;
