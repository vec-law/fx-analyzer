import psycopg2
from db.config import DB_CONFIG

def save_model_weights(train_uuid, arch_name, weights, mse_loss=None, mae_loss=None):
    query = """
        INSERT INTO model (train_uuid, architecture_id, weights, mse_loss, mae_loss)
        VALUES (
            %s, 
            (SELECT id FROM architecture WHERE name = %s), 
            %s, 
            %s, 
            %s
        )
        ON CONFLICT (train_uuid, architecture_id) 
        DO UPDATE SET 
            weights = EXCLUDED.weights,
            mse_loss = EXCLUDED.mse_loss,
            mae_loss = EXCLUDED.mae_loss;
    """
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute(query, (train_uuid, arch_name, weights, mse_loss, mae_loss))
                conn.commit()
        return True
    except Exception as e:
        raise Exception(f"Błąd bazy danych przy zapisywaniu modelu: {str(e)}")

def load_model_weights(train_uuid, arch_name):
    query = """
        SELECT weights 
        FROM model 
        WHERE train_uuid = %s 
        AND architecture_id = (SELECT id FROM architecture WHERE name = %s);
    """
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute(query, (train_uuid, arch_name))
                result = cur.fetchone()
                if result:
                    return bytes(result[0])
                return None
    except Exception as e:
        raise Exception(f"Błąd bazy danych przy odczycie modelu: {str(e)}")
