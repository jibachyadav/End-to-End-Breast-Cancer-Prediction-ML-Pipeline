from sqlalchemy import create_engine, text

DB_USER = "bc_user"
DB_PASS = "bc_password123"
DB_HOST = "localhost"
DB_PORT = 3306
DB_NAME = "breast_cancer_db"

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
