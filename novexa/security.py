import re
MAX_TEXT_LENGTH = 20000
def clean_text(value, limit=MAX_TEXT_LENGTH):
    if not isinstance(value, str):
        raise ValueError("Expected text input.")
    value = value.strip()
    if len(value) > limit:
        raise ValueError(f"Input is too long (limit: {limit:,} characters).")
    return value
def validate_dataframe(df, max_rows=50000, max_cols=150):
    if df is None or df.empty:
        raise ValueError("The dataset is empty.")
    if len(df) > max_rows:
        raise ValueError(f"Dataset exceeds the {max_rows:,}-row safety limit.")
    if len(df.columns) > max_cols:
        raise ValueError(f"Dataset exceeds the {max_cols}-column safety limit.")
    if len(set(map(str, df.columns))) != len(df.columns):
        raise ValueError("Column names must be unique.")
    return df
def redact_error(exc):
    msg = str(exc).replace("\n", " ")
    msg = re.sub(r'AIza[0-9A-Za-z_-]{20,}', '[REDACTED]', msg)
    return msg[:300]
