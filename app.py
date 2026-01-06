import streamlit as st
import psycopg2
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import re
import os

#NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.downloader.download('punkt')

# --- Database Schema ---
# to map natural language words to table/column names
SCHEMA_MAP = {
    "customers": "customers",
    "customer": "customers",
    "products": "products",
    "product": "products",
    "orders": "orders",
    "order": "orders",
    
    # Columns
    "sales": "oi.quantity * p.price",
    "revenue": "oi.quantity * p.price",
    "total sales": "oi.quantity * p.price",
    "total revenue": "oi.quantity * p.price",
    
    "product name": "p.name",
    "product category": "p.category",
    "category": "p.category",
    
    "customer name": "c.name",
    
    "order date": "o.order_date",
    "date": "o.order_date",
}

# --- NLTK Translation Engine ---
def translate_to_sql(question):
    """
    Translates a plain English question into a SQL query using NLTK and rules.
    This is a simple, rule-based engine as required by the assignment.
    """
    
    question = question.lower()
    tokens = word_tokenize(question)
    
    # Keywords
    agg_map = {
        "total": "SUM",
        "sum": "SUM",
        "average": "AVG",
        "avg": "AVG",
        "count": "COUNT"
    }
    
    # State variables
    select_clause = []
    agg_func = None
    agg_metric = None
    group_by_clause = []
    where_clause = []
    limit_clause = None
    
    # To find Aggregate Function
    for token in tokens:
        if token in agg_map:
            agg_func = agg_map[token]
            break # Assume one aggregate for simplicity

    # To find Metric (what to aggregate)
    # This checks for multi-word phrases first
    if "total sales" in question or "revenue" in question:
        agg_metric = (SCHEMA_MAP["sales"], "total_sales")
    elif "sales" in tokens:
         agg_metric = (SCHEMA_MAP["sales"], "total_sales")
         
    # Handle COUNT
    if agg_func == "COUNT":
        if "products" in tokens:
            agg_metric = ("p.product_id", "product_count")
        elif "customers" in tokens:
            agg_metric = ("c.customer_id", "customer_count")
        elif "orders" in tokens:
            agg_metric = ("o.order_id", "order_count")

    # To find Dimension (the 'by' part)
    if "by" in tokens:
        try:
            by_index = tokens.index("by")
            # Check for "product category" (2 words)
            if "product" in tokens and "category" in tokens:
                dim_col = SCHEMA_MAP["product category"]
                dim_alias = "category"
            # Check for single word dimension
            else:
                dim_token = tokens[by_index + 1]
                dim_col = SCHEMA_MAP[dim_token] # e.g., 'product' -> 'p.name'
                dim_alias = dim_token
            
            group_by_clause.append(dim_col)
            select_clause.append(f"{dim_col} AS {dim_alias}")
        except Exception:
            pass # No valid dimension found after 'by'

    # To find Top N
    if "top" in tokens:
        try:
            top_index = tokens.index("top")
            limit_num = int(tokens[top_index + 1])
            limit_clause = f"ORDER BY 2 DESC LIMIT {limit_num}" # Order by 2nd col (metric)
        except Exception:
            pass # Not a valid "top N" query

    # To assemble SELECT clause
    if agg_func and agg_metric:
        select_clause.append(f"{agg_func}({agg_metric[0]}) AS {agg_metric[1]}")
    elif agg_metric:
        select_clause.append(f"{agg_metric[0]} AS {agg_metric[1]}")

    if not select_clause:
        return "SELECT 'Could not understand query' AS error;"

    sql_select = f"SELECT {', '.join(select_clause)}"

    # To assemble FROM/JOINs (The hardest part)
    # We infer joins based on the columns used (p., c., o., oi.)
    query_parts = sql_select + " ".join(group_by_clause)
    
    tables = set()
    if "p." in query_parts or "oi." in query_parts:
        tables.add("products p")
        tables.add("order_items oi")
        tables.add("orders o")
    if "c." in query_parts:
        tables.add("customers c")
        tables.add("orders o")
        
    # Build JOIN logic
    from_clause = ""
    if "products p" in tables and "order_items oi" in tables and "orders o" in tables:
        from_clause = "FROM products p JOIN order_items oi ON p.product_id = oi.product_id JOIN orders o ON oi.order_id = o.order_id"
        if "customers c" in tables:
             from_clause += " JOIN customers c ON o.customer_id = c.customer_id"
    
    elif "customers c" in tables and "orders o" in tables:
         from_clause = "FROM customers c JOIN orders o ON c.customer_id = o.customer_id"
         if "order_items oi" in tables: # e.g. count orders by customer
            from_clause = "FROM customers c JOIN orders o ON c.customer_id = o.customer_id"
    
    elif "products p" in tables and agg_func == "COUNT":
        from_clause = "FROM products p"
    elif "customers c" in tables and agg_func == "COUNT":
        from_clause = "FROM customers c"
    elif "orders o" in tables and agg_func == "COUNT":
        from_clause = "FROM orders o"

    if not from_clause:
        return "SELECT 'Could not determine tables to query' AS error;"

    # Final Assembly
    sql_groupby = ""
    if group_by_clause:
        sql_groupby = f"GROUP BY {', '.join(group_by_clause)}"
    
    sql_limit = limit_clause if limit_clause else ""
    
    final_sql = f"{sql_select} {from_clause} {sql_groupby} {sql_limit};"
    
    return final_sql


# --- Streamlit App UI ---

st.set_page_config(layout="wide")
st.title("🇬🇧 English-to-SQL Translator SQL")
st.markdown("Using **NLTK** for natural language parsing (pre-LLM).")

DB_CONFIG = {
    "host": os.getenv("@db.rfigdgucilngksdklmal.supabase.co:5432/postgres"),
    "port": os.getenv("5432"),
    "dbname": os.getenv("postgres"),
    "user": os.getenv("N0Kzs"),
    "password": os.getenv("N0Kzs1O3%_11")
}

if st.sidebar.button("Connect"):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        st.session_state.db_conn = conn
        st.sidebar.success("✅ Connected to cloud database!")
    except Exception as e:
        st.session_state.db_conn = None
        st.sidebar.error(f"Connection failed: {e}")

# --- Main App Interface ---
if st.session_state.db_conn:
    st.header("Business Question")
    
    # Sample questions
    st.info("""
    **Sample Questions my NLTK parser understands:**
    * `Total sales by product category`
    * `Average revenue by customer`
    * `Count of products by category`
    * `Top 5 products by total sales`
    * `Count of customers`
    * `Count of orders`
    """)

    question = st.text_input("Enter your business question:", value="Total sales by product category")

    if st.button("🚀 Generate SQL & Run Query"):
        with st.spinner("Translating English to SQL..."):
            # Translation Engine
            sql_query = translate_to_sql(question)
            
            st.subheader("Generated SQL Query")
            st.code(sql_query, language="sql")
            
            try:
                # SQL Execution
                df = pd.read_sql_query(sql_query, st.session_state.db_conn)
                
                st.subheader("Query Results")
                st.dataframe(df)
                
                # Optional Visualization
                st.subheader("📊 Visualization")
                # Try to find a good column to use as the x-axis (the dimension)
                if not df.empty and len(df.columns) > 1:
                    try:
                        # Use the first column as the index (x-axis)
                        chart_df = df.set_index(df.columns[0])
                        st.bar_chart(chart_df)
                    except Exception as e:
                        st.warning(f"Could not auto-generate chart: {e}")
                else:
                    st.info("No data or insufficient data to plot.")

            except Exception as e:
                st.error(f"Error executing query: {e}")
else:
    st.warning("Please connect to your database using the sidebar.")