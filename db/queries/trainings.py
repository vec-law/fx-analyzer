import psycopg2
import uuid
import pandas as pd
from db.config import DB_CONFIG

def count_running_trainings():
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT train_uuid FROM training
                    JOIN status ON status_id = status.id
                    WHERE status.name = 'running'
                """)
                rows = cur.fetchall()
                return len(rows)
    except Exception as e:
        raise Exception(f"Błąd bazy danych: {str(e)}")

def get_pending_trainings():
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT train_uuid FROM training
                    JOIN status ON status_id = status.id
                    WHERE status.name = 'pending'
                """)
                rows = cur.fetchall()
                return [row[0] for row in rows]
    except Exception as e:
        raise Exception(f"Błąd bazy danych: {str(e)}")

def get_user_trainings(user_id):
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        training.train_uuid, 
                        instrument.name, 
                        timeframe.name, 
                        data_source.name, 
                        status.name, 
                        training.created_at
                    FROM training
                    JOIN instrument ON training.instrument_id = instrument.id
                    JOIN timeframe ON training.timeframe_id = timeframe.id
                    JOIN data_source ON training.data_source_id = data_source.id
                    JOIN status ON training.status_id = status.id
                    WHERE training.user_id = %s
                    ORDER BY training.created_at DESC
                """, (user_id, ))
                rows = cur.fetchall()
                return [
                    {
                        'train_uuid': r[0],
                        'instrument': r[1],
                        'timeframe_name': r[2],
                        'data_source': r[3],
                        'status': r[4],
                        'created_at': r[5]
                    } for r in rows
                ]
    except Exception as e:
        raise Exception(f"Błąd podczas pobierania listy treningów: {str(e)}")

def add_training(user_id, config):
    train_uuid = str(uuid.uuid4())
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO training (train_uuid, user_id, instrument_id, timeframe_id, status_id, data_source_id) 
                    VALUES (%s, %s,
                        (SELECT id FROM instrument WHERE name = %s), 
                        (SELECT id FROM timeframe WHERE name = %s), 
                        (SELECT id FROM status WHERE name = 'created'),
                        (SELECT id FROM data_source WHERE name = %s))
                """, (
                    train_uuid, user_id,
                    config['instrument_name'],
                    config['timeframe_name'],
                    config['data_source_name']
                ))
                cur.execute("""
                    INSERT INTO parameter_set (train_uuid, all_samples, test_samples, seed, epochs, train_noise, learning_rate) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    train_uuid,
                    config['all_samples'],
                    config['test_samples'],
                    config['seed'],
                    config['epochs'],
                    config['train_noise'],
                    config['learning_rate']
                ))
                for feature in config['features']:
                    cur.execute("""
                        INSERT INTO feature_def (train_uuid, feature_type_id, feature_periods, shift) 
                        VALUES (%s, (SELECT id FROM feature_type WHERE name = %s), %s, %s)
                    """, (
                        train_uuid,
                        feature['feature_type'],
                        feature['feature_periods'],
                        feature['shift']
                    ))
                for target in config['targets']:
                    column_name = target['column']
                    target_shift = target['shift']
                    base_id = None
                    calc_id = None
                    cur.execute("SELECT id FROM base_column WHERE name = %s", (column_name,))
                    row_base = cur.fetchone()
                    if row_base:
                        base_id = row_base[0]
                    else:
                        cur.execute("SELECT id FROM calculated_column WHERE name = %s", (column_name,))
                        row_calc = cur.fetchone()
                        if row_calc:
                            calc_id = row_calc[0]
                    if base_id is None and calc_id is None:
                        raise ValueError(f"Kolumna {column_name} nie istnieje w słownikach.")
                    cur.execute("""
                        INSERT INTO target_def (train_uuid, base_column_id, calculated_column_id, shift) 
                        VALUES (%s, %s, %s, %s)
                    """, (train_uuid, base_id, calc_id, target_shift))
                for architecture in config['architectures']:
                    cur.execute("""
                        INSERT INTO training_architecture (train_uuid, architecture_id) 
                        VALUES (%s, (SELECT id FROM architecture WHERE name = %s))
                    """, (train_uuid, architecture))
                conn.commit()
                return train_uuid
    except ValueError as e:
        raise e
    except Exception as e:
        raise Exception(f"Błąd bazy danych przy dodawaniu zadania: {str(e)}")

def del_training(train_uuid):
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM training WHERE train_uuid = %s", (train_uuid,))
                if cur.rowcount == 0:
                    raise Exception(f"Trening {train_uuid} nie istnieje w bazie danych.")
                conn.commit()
                return True
    except Exception as e:
        raise Exception(f"Błąd podczas usuwania zadania: {str(e)}")

def get_training_config(train_uuid):
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                config = {
                    'instrument_name': None,
                    'timeframe_name': None,
                    "data_source_name": None,
                    "all_samples": None,
                    "test_samples": None,
                    "seed": None,
                    "epochs": None,
                    "train_noise": None,
                    "learning_rate": None,
                    'features': [],
                    'targets': [],
                    'architectures': []
                }
                cur.execute("""
                    SELECT
                        instrument.name,
                        timeframe.name,
                        data_source.name,
                        all_samples,
                        test_samples,
                        seed,
                        epochs,
                        train_noise,
                        learning_rate   
                    FROM training
                    JOIN instrument ON training.instrument_id = instrument.id
                    JOIN timeframe ON training.timeframe_id = timeframe.id
                    JOIN parameter_set ON training.train_uuid = parameter_set.train_uuid
                    JOIN data_source ON training.data_source_id = data_source.id
                    WHERE training.train_uuid = %s
                """, (train_uuid, ))
                result = cur.fetchone()
                if result is None:
                    raise ValueError(f"Brak treningu: {train_uuid}")
                config["instrument_name"] = result[0]
                config["timeframe_name"] = result[1]
                config["data_source_name"] = result[2]
                config["all_samples"] = result[3]
                config["test_samples"] = result[4]
                config["seed"] = result[5]
                config["epochs"] = result[6]
                config["train_noise"] = result[7]
                config["learning_rate"] = result[8]
                cur.execute("""
                    SELECT feature_type.name, feature_def.feature_periods, feature_def.shift
                    FROM feature_def
                    JOIN feature_type ON feature_def.feature_type_id = feature_type.id
                    WHERE feature_def.train_uuid = %s
                    ORDER BY feature_def.id
                """, (train_uuid, ))
                for f_type, f_periods, f_shift in cur.fetchall():
                    config['features'].append(f"{f_type}:{'-'.join(map(str, f_periods))}:{f_shift}")
                cur.execute("""
                    SELECT base_column.name, target_def.shift
                    FROM target_def
                    JOIN base_column ON target_def.base_column_id = base_column.id
                    WHERE target_def.train_uuid = %s
                """, (train_uuid, ))
                for name, shift in cur.fetchall():
                    config['targets'].append(f"{name}:{shift}")
                cur.execute("""
                    SELECT calculated_column.name, target_def.shift
                    FROM target_def
                    JOIN calculated_column ON target_def.calculated_column_id = calculated_column.id
                    WHERE target_def.train_uuid = %s
                """, (train_uuid, ))
                for name, shift in cur.fetchall():
                    config['targets'].append(f"{name}:{shift}")
                cur.execute("""
                    SELECT architecture.name
                    FROM architecture
                    JOIN training_architecture ON training_architecture.architecture_id = architecture.id
                    WHERE training_architecture.train_uuid = %s
                    ORDER BY architecture.id
                """, (train_uuid, ))
                config['architectures'] = [row[0] for row in cur.fetchall()]
                return config
    except ValueError as e:
        raise e
    except Exception as e:
        raise Exception(f"Błąd podczas pobierania konfiguracji: {str(e)}")

def get_training_status(train_uuid):
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT status.name 
                    FROM training
                    JOIN status ON training.status_id = status.id
                    WHERE training.train_uuid = %s
                """, (train_uuid,))
                result = cur.fetchone()
                return result[0] if result else None
    except Exception as e:
        raise Exception(f"Błąd pobierania statusu: {str(e)}")

def update_training_status(train_uuid, status_name):
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE training
                    SET status_id = (SELECT id FROM status WHERE name = %s)
                    WHERE train_uuid = %s
                """, (status_name, train_uuid))
                if cur.rowcount == 0:
                    raise Exception(f"Nie znaleziono treningu o UUID: {train_uuid}")
                conn.commit()
                return True
    except Exception as e:
        raise Exception(f"Nie udało się zaktualizować statusu: {str(e)}")

def save_training_stats(train_uuid, ser_mean, ser_std):
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                data = []
                for col in ser_mean.index:
                    data.append((train_uuid, col, 'mean', float(ser_mean[col])))
                    data.append((train_uuid, col, 'std', float(ser_std[col])))
                if data:
                    cur.executemany("""
                        INSERT INTO statistic (train_uuid, column_name, stat_name, stat_value)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (train_uuid, column_name, stat_name) 
                        DO UPDATE SET stat_value = EXCLUDED.stat_value
                    """, data)
                conn.commit()
    except Exception as e:
        raise Exception(f"Błąd bazy danych przy zapisywaniu statystyk: {str(e)}")

def load_training_stats(train_uuid):
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT column_name, stat_name, stat_value 
                    FROM statistic 
                    WHERE train_uuid = %s
                """, (train_uuid,))
                rows = cur.fetchall()
                if not rows:
                    return None, None
                means = {}
                stds = {}
                for col_name, stat_name, value in rows:
                    if stat_name == 'mean':
                        means[col_name] = value
                    elif stat_name == 'std':
                        stds[col_name] = value
                ser_mean = pd.Series(means)
                ser_std = pd.Series(stds)
                return ser_mean, ser_std
    except Exception as e:
        raise Exception(f"Błąd bazy danych przy odczycie statystyk: {str(e)}")

def add_training_log(train_uuid, message):
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO log (train_uuid, message) 
                    VALUES (%s, %s)
                """, (train_uuid, message))

                conn.commit()

                return True
    except ValueError as e:
        raise e
    except Exception as e:
        raise Exception(f"Błąd bazy danych: {str(e)}")
