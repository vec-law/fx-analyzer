import psycopg2
import uuid
from db.config import DB_CONFIG

def get_user_predictions(user_id, status=None):
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        prediction.pred_uuid, 
                        prediction.train_uuid, 
                        status.name, 
                        prediction.all_samples, 
                        prediction.predicted_samples, 
                        prediction.created_at,
                        instrument.name,
                        timeframe.name
                    FROM prediction
                    JOIN status ON prediction.status_id = status.id
                    JOIN training ON prediction.train_uuid = training.train_uuid
                    JOIN instrument ON training.instrument_id = instrument.id
                    JOIN timeframe ON training.timeframe_id = timeframe.id
                    WHERE training.user_id = %s
                    AND (%s IS NULL OR status.name = %s)
                    ORDER BY prediction.created_at DESC
                """, (user_id, status, status))
                rows = cur.fetchall()
                return [
                    {
                        'pred_uuid': r[0],
                        'train_uuid': r[1],
                        'status': r[2],
                        'all_samples': r[3],
                        'predicted_samples': r[4],
                        'created_at': r[5],
                        'instrument_name': r[6],
                        'timeframe_name': r[7]
                    } for r in rows
                ]
    except Exception as e:
        raise Exception(f"Błąd: {str(e)}")

def add_prediction(train_uuid, all_samples, predicted_samples):
    pred_uuid = str(uuid.uuid4())
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO prediction (
                        pred_uuid, train_uuid, status_id, 
                        all_samples, predicted_samples
                    ) 
                    VALUES (%s, %s, (SELECT id FROM status WHERE name = 'pending'), %s, %s)
                """, (pred_uuid, train_uuid, all_samples, predicted_samples))
                conn.commit()
                return pred_uuid
    except ValueError as e:
        raise e
    except Exception as e:
        raise Exception(f"Błąd bazy danych przy dodawaniu predykcji: {str(e)}")

def get_predictions():
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        prediction.pred_uuid, 
                        prediction.train_uuid, 
                        status.name, 
                        prediction.all_samples, 
                        prediction.predicted_samples, 
                        prediction.created_at,
                        instrument.name,
                        timeframe.name
                    FROM prediction
                    JOIN status ON prediction.status_id = status.id
                    JOIN training ON prediction.train_uuid = training.train_uuid
                    JOIN instrument ON training.instrument_id = instrument.id
                    JOIN timeframe ON training.timeframe_id = timeframe.id
                    ORDER BY prediction.created_at DESC
                """)
                rows = cur.fetchall()
                return [
                    {
                        'pred_uuid': r[0],
                        'train_uuid': r[1],
                        'status': r[2],
                        'all_samples': r[3],
                        'predicted_samples': r[4],
                        'created_at': r[5],
                        'instrument_name': r[6],
                        'timeframe_name': r[7]
                    } for r in rows
                ]
    except Exception as e:
        raise Exception(f"Błąd podczas pobierania listy predykcji: {str(e)}")

def del_prediction(pred_uuid):
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM prediction WHERE pred_uuid = %s", (pred_uuid,))
                if cur.rowcount == 0:
                    raise Exception(f"Predykcja {pred_uuid} nie istnieje.")
                conn.commit()
        return True
    except Exception as e:
        raise Exception(f"Błąd podczas usuwania predykcji: {str(e)}")

def update_prediction_status(pred_uuid, status_name):
    try:
        if not pred_uuid or not status_name:
            raise ValueError("Błąd: Brak UUID predykcji lub nazwy statusu")
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE prediction
                    SET status_id = (SELECT id FROM status WHERE name = %s)
                    WHERE pred_uuid = %s
                """, (status_name, pred_uuid))
                if cur.rowcount == 0:
                    raise Exception(f"Nie znaleziono predykcji o UUID: {pred_uuid}")
                conn.commit()
                return True
    except Exception as e:
        raise Exception(f"Nie udało się zaktualizować statusu predykcji: {str(e)}")

def get_prediction_config(pred_uuid):
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        prediction.all_samples, 
                        prediction.predicted_samples,
                        prediction.train_uuid,
                        status.name
                    FROM prediction
                    JOIN status ON prediction.status_id = status.id
                    WHERE prediction.pred_uuid = %s
                """, (pred_uuid,))
                row = cur.fetchone()
                if row:
                    return {
                        "all_samples": row[0],
                        "predicted_samples": row[1],
                        "train_uuid": row[2],
                        "status": row[3]
                    }
                return None
    except Exception as e:
        raise Exception(f"Błąd bazy danych przy pobieraniu konfiguracji predykcji: {str(e)}")

def save_prediction_result(pred_uuid, arch_name, data_bytes):
    query = """
        INSERT INTO prediction_result (pred_uuid, architecture_id, data)
        VALUES (
            %s, 
            (SELECT id FROM architecture WHERE name = %s), 
            %s
        )
        ON CONFLICT (pred_uuid, architecture_id) 
        DO UPDATE SET data = EXCLUDED.data;
    """
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute(query, (pred_uuid, arch_name, psycopg2.Binary(data_bytes)))
                if cur.rowcount == 0:
                    raise Exception(f"Nie znaleziono architektury o nazwie: {arch_name}")
                conn.commit()
                return True
    except Exception as e:
        raise Exception(f"Błąd bazy danych przy zapisywaniu wyniku: {str(e)}")

def load_prediction_result(pred_uuid, arch_name):
    query = """
        SELECT prediction_result.data 
        FROM prediction_result
        JOIN architecture ON prediction_result.architecture_id = architecture.id
        WHERE prediction_result.pred_uuid = %s 
        AND architecture.name = %s
    """
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute(query, (pred_uuid, arch_name))
                result = cur.fetchone()
                if result:
                    return bytes(result[0])
                return None
    except Exception as e:
        raise Exception(f"Błąd bazy danych przy odczycie wyniku: {str(e)}")
