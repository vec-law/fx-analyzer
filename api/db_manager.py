import psycopg2
import uuid
import pandas as pd
import bcrypt
import os
from dotenv import load_dotenv
from psycopg2.extras import register_uuid

register_uuid()
load_dotenv()

class DBManager:
    def __init__(self, db_config):
        self.config = db_config

    def get_trainings(self, user_id):
        try:
            with psycopg2.connect(**self.config) as conn:
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
                
        except ValueError as e:
            raise e
        except Exception as e:
            raise Exception(f"Błąd podczas pobierania listy treningów: {str(e)}")

    def add_training(self, user_id, config):
        train_uuid = str(uuid.uuid4())

        try:
            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO training (train_uuid, user_id, instrument_id, timeframe_id, status_id, data_source_id) 
                        VALUES (%s, %s,
                            (SELECT id FROM instrument WHERE name = %s), 
                            (SELECT id FROM timeframe WHERE name = %s), 
                            (SELECT id FROM status WHERE name = 'pending'),
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
                            VALUES (%s, 
                                (SELECT id FROM feature_type WHERE name = %s), 
                                %s, %s)
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
        
    def del_training(self, train_uuid):
        try:
            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM training WHERE train_uuid = %s", (train_uuid,))

                    if cur.rowcount == 0:
                        raise Exception(f"Trening {train_uuid} nie istnieje w bazie danych.")

                    conn.commit()
                    return True

        except ValueError as e:
            raise e
        except Exception as e:
            raise Exception(f"Błąd podczas usuwania zadania: {str(e)}")
        
    def get_training_config(self, train_uuid):
        try:
            with psycopg2.connect(**self.config) as conn:
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

    def get_training_status(self, train_uuid):
        try:
            with psycopg2.connect(**self.config) as conn:
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

    def update_training_status(self, train_uuid, status_name):
        try:
            with psycopg2.connect(**self.config) as conn:
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



    def save_training_stats(self, train_uuid, ser_mean, ser_std):
        try:
            with psycopg2.connect(**self.config) as conn:
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

    def load_training_stats(self, train_uuid):
        try:
            with psycopg2.connect(**self.config) as conn:
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

    def save_model_weights(self, train_uuid, arch_name, weights, mse_loss=None, mae_loss=None):
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
            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (train_uuid, arch_name, weights, mse_loss, mae_loss))
                    conn.commit()
            return True
        except Exception as e:
            raise Exception(f"Błąd bazy danych przy zapisywaniu modelu: {str(e)}")

    def load_model_weights(self, train_uuid, arch_name):
        query = """
            SELECT weights 
            FROM model 
            WHERE train_uuid = %s 
            AND architecture_id = (SELECT id FROM architecture WHERE name = %s);
        """
        try:
            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (train_uuid, arch_name))
                    result = cur.fetchone()

                    if result:
                        weights = bytes(result[0])
                        return weights
                    return None
        except Exception as e:
            raise Exception(f"Błąd bazy danych przy odczycie modelu: {str(e)}")

    def add_prediction(self, train_uuid, all_samples, predicted_samples):
        pred_uuid = str(uuid.uuid4())
        try:
            with psycopg2.connect(**self.config) as conn:
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

    def get_predictions(self):
        try:
            with psycopg2.connect(**self.config) as conn:
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

    def del_prediction(self, pred_uuid):
        try:
            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM prediction WHERE pred_uuid = %s", (pred_uuid,))
                    if cur.rowcount == 0:
                        raise Exception(f"Predykcja {pred_uuid} nie istnieje.")
                    conn.commit()
            return True
        except Exception as e:
            raise Exception(f"Błąd podczas usuwania predykcji: {str(e)}")

    def update_prediction_status(self, pred_uuid, status_name):
        try:
            if not pred_uuid or not status_name:
                raise ValueError("Błąd: Brak UUID predykcji lub nazwy statusu")

            with psycopg2.connect(**self.config) as conn:
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

    def get_prediction_config(self, pred_uuid):
        try:
            with psycopg2.connect(**self.config) as conn:
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
                    config = None

                    if row:
                        config = {
                            "all_samples": row[0],
                            "predicted_samples": row[1],
                            "train_uuid": row[2],
                            "status": row[3]
                        }

                    return config
        except Exception as e:
            raise Exception(f"Błąd bazy danych przy pobieraniu konfiguracji predykcji: {str(e)}")

    def save_prediction_result(self, pred_uuid, arch_name, data_bytes):
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
            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (pred_uuid, arch_name, psycopg2.Binary(data_bytes)))

                    if cur.rowcount == 0:
                        raise Exception(f"Nie znaleziono architektury o nazwie: {arch_name}")

                    conn.commit()
                    return True
        except Exception as e:
            raise Exception(f"Błąd bazy danych przy zapisywaniu wyniku: {str(e)}")

    def load_prediction_result(self, pred_uuid, arch_name):
        query = """
            SELECT prediction_result.data 
            FROM prediction_result
            JOIN architecture ON prediction_result.architecture_id = architecture.id
            WHERE prediction_result.pred_uuid = %s 
            AND architecture.name = %s
        """
        try:
            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (pred_uuid, arch_name))
                    result = cur.fetchone()

                    if result:
                        return bytes(result[0])
                    return None
        except Exception as e:
            raise Exception(f"Błąd bazy danych przy odczycie wyniku: {str(e)}")
        
    def add_user(self, user_name, password, is_admin):
        try:           
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

            user_type = "admin" if is_admin else "user"
        
            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO app_user (name, password_hash, role_id)
                        VALUES (%s, %s, (SELECT id FROM role WHERE name = %s))
                    """, (user_name, password_hash, user_type))
                    conn.commit()
                    return True
                
        except ValueError as e:
            raise e
        except Exception as e:
            raise Exception(f"Błąd bazy danych: {str(e)}")
        
    def del_user(self, user_id):
        try:
            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM app_user WHERE id = %s", (user_id, ))

                    if cur.rowcount == 0:
                        raise Exception("Użytkownik nie istnieje w bazie danych")

                    conn.commit()
                    return True
                
        except Exception as e:
            raise Exception(f"Błąd bazy danych: {str(e)}")
        
    def block_user(self, user_id):
        try:                
            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE app_user
                        SET is_blocked = TRUE
                        WHERE id = %s
                    """, (user_id, ))

                    if cur.rowcount == 0:
                        raise Exception("Użytkownik nie istnieje w bazie danych")
                    
                    conn.commit()
        except ValueError as e:
            raise e
        except Exception as e:
            raise Exception(f"Błąd bazy danych: {str(e)}")
        
    def unblock_user(self, user_id):
        try:                
            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE app_user
                        SET is_blocked = FALSE
                        WHERE id = %s
                    """, (user_id, ))

                    if cur.rowcount == 0:
                        raise Exception("Użytkownik nie istnieje w bazie danych")
                    
                    conn.commit()
        except ValueError as e:
            raise e
        except Exception as e:
            raise Exception(f"Błąd bazy danych: {str(e)}")
    
    def login_user(self, user_id, user_name, password):
        try:           
            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT password_hash FROM app_user
                        WHERE app_user.name = %s
                    """, (user_name, ))
                    result = cur.fetchone()

                    password_hash = result[0]
                    password_hash = bytes(password_hash)

                    if not bcrypt.checkpw(password.encode('utf-8'), password_hash):
                        raise ValueError(f"Podano nieprawidłowe dane logowania")

                    session_token = uuid.uuid4()

                    cur.execute("""
                        UPDATE app_user
                        SET session_token = %s
                        WHERE app_user.id = %s
                    """, (session_token, user_id))

                    conn.commit()
                    
                    return session_token
        except ValueError as e:
            raise e
        except Exception as e:
            raise Exception(f"Błąd bazy danych: {str(e)}")

    def logout_user(self, user_id):
        try:
            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE app_user
                        SET session_token = NULL
                        WHERE app_user.id = %s
                    """, (user_id, ))
                    conn.commit()

                    return True
        
        except ValueError as e:
            raise e
        except Exception as e:
            raise Exception(f"Błąd bazy danych: {str(e)}")
        
    def change_password(self, user_id, new_password):
        try:
            new_password_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
        
            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE app_user
                        SET password_hash = %s
                        WHERE id = %s
                    """, (new_password_hash, user_id))
                    conn.commit()
                    return True
                
        except ValueError as e:
            raise e
        except Exception as e:
            raise Exception(f"Błąd bazy danych: {str(e)}")
        
    def get_session_token(self, user_id):
        try:
            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT session_token FROM app_user
                        WHERE id = %s
                    """, (user_id, ))
                    result = cur.fetchone()

                    if result is None: return None
                    else: return result[0]
        except ValueError as e:
            raise e
        except Exception as e:
            raise Exception(f"Błąd bazy danych: {str(e)}")

    def ensure_admin(self):
        try:
            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT app_user.id FROM app_user
                        JOIN role ON role_id = role.id
                        WHERE role.name = 'admin'
                    """)
                    
                    if cur.fetchone() is None:
                        self.add_user(
                            os.getenv("ADMIN_LOGIN"),
                            os.getenv("ADMIN_PASSWORD"),
                            is_admin=True
                        ) 

        except ValueError as e:
            raise e
        except Exception as e:
            raise Exception(f"Błąd bazy danych: {str(e)}")
        
    def is_blocked(self, user_id):
        try:
            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT is_blocked FROM app_user
                        WHERE id = %s
                    """, (user_id, ))
                    result = cur.fetchone()

                    if result is None: return None
                    else: return result[0]
        except ValueError as e:
            raise e
        except Exception as e:
            raise Exception(f"Błąd bazy danych: {str(e)}")
               
    def get_users(self):
        try:
            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT
                            app_user.id,
                            app_user.name,
                            role.name,
                            is_blocked
                        FROM app_user
                        JOIN role ON role_id = role.id
                    """)
                    return cur.fetchall()
                
        except ValueError as e:
            raise e
        except Exception as e:
            raise Exception(f"Błąd bazy danych: {str(e)}")
        
    def get_user_id(self, user_name):
        try:
            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT id FROM app_user
                        WHERE name = %s
                    """, (user_name, ))
                    user_id = cur.fetchone()

                    if user_id is None: return None
                    else: return user_id[0]
        except ValueError as e:
            raise e
        except Exception as e:
            raise Exception(f"Błąd bazy danych: {str(e)}")
        
    def get_user_name(self, user_id):
        try:
            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT name FROM app_user
                        WHERE id = %s
                    """, (user_id, ))
                    user_name = cur.fetchone()

                    if user_name is None: return None
                    else: return user_name[0]
        except ValueError as e:
            raise e
        except Exception as e:
            raise Exception(f"Błąd bazy danych: {str(e)}")
        
    def user_exists(self, user_id):
        try:
            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT id FROM app_user
                        WHERE id = %s
                    """, (user_id, ))
                    result = cur.fetchone()

                    if result is None: return False
                    else: return True
        except ValueError as e:
            raise e
        except Exception as e:
            raise Exception(f"Błąd bazy danych: {str(e)}")
        
    def validate_access(self, user_id, session_token):
        try:
            if not self.user_exists(user_id):
                raise ValueError(f"Użytkownik nie istnieje")
            
            if self.is_blocked(user_id):
                raise ValueError(f"Użytkownik zablokowany")
            
            db_session_token = self.get_session_token(user_id)

            if db_session_token is None:
                raise ValueError(f"Użytkownik wylogowany")
            
            elif db_session_token != session_token:
                raise ValueError("Nieprawidłowe dane logowania sesji")
            
            return True

        except ValueError as e:
            raise e
        except Exception as e:
            raise Exception(f"Błąd bazy danych: {str(e)}")
        
    def get_role(self, user_id):
        try:
            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT role.name FROM role
                        JOIN app_user ON role_id = role.id
                        WHERE app_user.id = %s
                    """, (user_id, ))
                    result = cur.fetchone()

                    if result is None: return None
                    else: return result[0]
        except ValueError as e:
            raise e
        except Exception as e:
            raise Exception(f"Błąd bazy danych: {str(e)}")