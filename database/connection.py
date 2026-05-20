from sqlalchemy import create_engine, text
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from src.constants import DB_USER, DB_PASS, DB_HOST, DB_PORT, DB_NAME

def get_engine():
    url = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(url)
    return engine

def test_connection():
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
        print("DB Connection successful")

if __name__ == "__main__":
    test_connection()
    