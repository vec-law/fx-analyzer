import psycopg2
import uuid
import pandas as pd

class DatabaseManager:
    def __init__(self, db_config):
        self.config = db_config

    def add_training_job(self, config):
        job_uuid = str(uuid.uuid4())

        try:
            if not config.get('features') or not config.get('targets') or not config.get('architectures'):
                raise ValueError("Błąd walidacji: Brak Targetów, Featurów lub Architektury.")

            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO training_job (job_uuid, instrument_id, timeframe_id, status_id, data_source_id) 
                        VALUES (%s, 
                            (SELECT id FROM instrument WHERE name = %s), 
                            (SELECT id FROM timeframe WHERE name = %s), 
                            (SELECT id FROM status WHERE name = 'pending'),
                            (SELECT id FROM data_source WHERE name = %s))
                    """, (
                        job_uuid, 
                        config['instrument']['name'], 
                        config['timeframe']['name'], 
                        config['data_source']
                    ))

                    cur.execute("""
                        INSERT INTO parameter_set (training_job_uuid, samples_limit, test_samples, seed, epochs, train_noise, learning_rate) 
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        job_uuid, 
                        config['parameter_set']['samples_limit'], 
                        config['parameter_set']['test_samples'], 
                        config['parameter_set']['seed'], 
                        config['parameter_set']['epochs'], 
                        config['parameter_set']['train_noise'], 
                        config['parameter_set']['learning_rate']
                    ))

                    for feature in config['features']:
                        cur.execute("""
                            INSERT INTO feature_def (training_job_uuid, feature_type_id, feature_periods, shift) 
                            VALUES (%s, 
                                (SELECT id FROM feature_type WHERE name = %s), 
                                %s, %s)
                        """, (
                            job_uuid, 
                            feature['feature_type'], 
                            feature['feature_periods'],
                            feature['shift']
                        ))

                    for target in config['targets']:
                        column_name = target['column']
                        shift = target['shift']

                        cur.execute("SELECT base_column.id FROM base_column WHERE name = %s", (column_name,))
                        base_column = cur.fetchone()
                        
                        base_column_id = base_column[0] if base_column else None
                        calculated_column_id = None

                        if not base_column_id:
                            cur.execute("SELECT calculated_column.id FROM calculated_column WHERE name = %s", (column_name,))
                            calculated_column = cur.fetchone()
                            calculated_column_id = calculated_column[0] if calculated_column else None
                            
                        if not base_column_id and not calculated_column_id:
                            raise ValueError(f"Kolumna {column_name} nie istnieje w bazie danych.")

                        cur.execute("""
                            INSERT INTO target_def (training_job_uuid, base_column_id, calculated_column_id, shift) 
                            VALUES (%s, %s, %s, %s)
                        """, (job_uuid, base_column_id, calculated_column_id, shift))

                    for architecture in config['architectures']:
                        cur.execute("""
                            INSERT INTO training_job_architecture (training_job_uuid, architecture_id) 
                            VALUES (%s, (SELECT id FROM architecture WHERE name = %s))
                        """, (job_uuid, architecture))

                    conn.commit()
                    return job_uuid            
        except ValueError as e:
            raise e
        except Exception as e:
            raise Exception(f"Błąd bazy danych przy dodawaniu zadania: {str(e)}")
        
    def get_training_status(self, job_uuid):
        try:
            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT s.name 
                        FROM training_job tj
                        JOIN status s ON tj.status_id = s.id
                        WHERE tj.job_uuid = %s
                    """, (job_uuid,))
                    result = cur.fetchone()
                    return result[0] if result else None
        except Exception as e:
            raise Exception(f"Błąd pobierania statusu: {str(e)}")
        
    def update_training_status(self, job_uuid, status_name):
        try:
            if not job_uuid or not status_name:
                raise ValueError("Błąd: Brak UUID zadania lub nazwy statusu.")

            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE training_job 
                        SET status_id = (SELECT id FROM status WHERE name = %s)
                        WHERE job_uuid = %s
                    """, (status_name, job_uuid))
                    
                    if cur.rowcount == 0:
                        raise Exception(f"Nie znaleziono zadania o UUID: {job_uuid}")

                    conn.commit()
                    return True
        except Exception as e:
            raise Exception(f"Nie udało się zaktualizować statusu: {str(e)}")


    def del_training_job(self, job_uuid):
        try:
            if not job_uuid:
                raise ValueError("Błąd: Nie podano UUID do usunięcia.")

            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM training_job WHERE job_uuid = %s", (job_uuid,))

                    if cur.rowcount == 0:
                        raise Exception(f"Zadanie {job_uuid} nie istnieje w bazie danych.")
                    
                    conn.commit()
                    return True
        except Exception as e:
            raise Exception(f"Błąd podczas usuwania zadania: {str(e)}")

    def get_training_jobs(self):
        try:
            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT 
                            training_job.job_uuid, 
                            instrument.name, 
                            timeframe.name, 
                            data_source.name, 
                            status.name, 
                            training_job.created_at
                        FROM training_job
                        JOIN instrument ON training_job.instrument_id = instrument.id
                        JOIN timeframe ON training_job.timeframe_id = timeframe.id
                        JOIN data_source ON training_job.data_source_id = data_source.id
                        JOIN status ON training_job.status_id = status.id
                        ORDER BY training_job.created_at DESC
                    """)
                    rows = cur.fetchall()
                    
                    return [
                        {
                            'job_uuid': r[0], 
                            'instrument': r[1], 
                            'timeframe_name': r[2], 
                            'data_source': r[3], 
                            'status': r[4], 
                            'created_at': r[5]
                        } for r in rows
                    ]
        except Exception as e:
            raise Exception(f"Błąd podczas pobierania listy zadań: {str(e)}")
        
    def save_training_stats(self, job_uuid, ser_mean, ser_std):
        try:
            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cur:
                    data = []
                    for col in ser_mean.index:
                        data.append((job_uuid, col, 'mean', float(ser_mean[col])))
                        data.append((job_uuid, col, 'std', float(ser_std[col])))

                    if data:
                        cur.executemany("""
                            INSERT INTO statistic (training_job_uuid, column_name, stat_name, stat_value)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (training_job_uuid, column_name, stat_name) 
                            DO UPDATE SET stat_value = EXCLUDED.stat_value
                        """, data)
                    
                    conn.commit()
        except Exception as e:
            raise Exception(f"Błąd bazy danych przy zapisywaniu statystyk: {str(e)}")

    def load_training_stats(self, job_uuid):
        try:
            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT column_name, stat_name, stat_value 
                        FROM statistic 
                        WHERE training_job_uuid = %s
                    """, (job_uuid,))
                    
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
        
    def save_model_weights(self, job_uuid, arch_name, weights, mse_loss=None, mae_loss=None):
        query = """
            INSERT INTO model (training_job_uuid, architecture_id, weights, mse_loss, mae_loss)
            VALUES (
                %s, 
                (SELECT id FROM architecture WHERE name = %s), 
                %s, 
                %s, 
                %s
            )
            ON CONFLICT (training_job_uuid, architecture_id) 
            DO UPDATE SET 
                weights = EXCLUDED.weights,
                mse_loss = EXCLUDED.mse_loss,
                mae_loss = EXCLUDED.mae_loss;
        """
        try:
            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (job_uuid, arch_name, weights, mse_loss, mae_loss))
                    conn.commit()
            return True
        except Exception as e:
            raise Exception(f"Błąd bazy danych przy zapisywaniu modelu: {str(e)}")
        
    def load_model_weights(self, job_uuid, arch_name):
        query = """
            SELECT weights 
            FROM model 
            WHERE training_job_uuid = %s 
            AND architecture_id = (SELECT id FROM architecture WHERE name = %s);
        """
        try:
            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (job_uuid, arch_name))
                    result = cur.fetchone()
                    
                    if result:
                        weights = bytes(result[0])
                        return weights
                    return None
        except Exception as e:
            raise Exception(f"Błąd bazy danych przy odczycie modelu: {str(e)}")
        
    def add_simulation_job(self, training_uuid, samples_simulation, predicted_samples, strategies):
        sim_uuid = str(uuid.uuid4())
        try:
            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cur:
                    strategy_ids = []
                    for strategy_name in strategies:
                        cur.execute("SELECT id FROM strategy WHERE name = %s", (strategy_name,))
                        result = cur.fetchone()
                        if not result:
                            raise ValueError(f"Błąd: Strategia '{strategy_name}' nie istnieje w bazie danych.")
                        strategy_ids.append(result[0])

                    cur.execute("""
                        INSERT INTO simulation (
                            sim_uuid, training_job_uuid, status_id, 
                            samples_simulation, predicted_samples
                        ) 
                        VALUES (%s, %s, (SELECT id FROM status WHERE name = 'pending'), %s, %s)
                    """, (sim_uuid, training_uuid, samples_simulation, predicted_samples))

                    for strategy_id in strategy_ids:
                        cur.execute("""
                            INSERT INTO simulation_strategy (simulation_uuid, strategy_id)
                            VALUES (%s, %s)
                        """, (sim_uuid, strategy_id))

                    conn.commit()
                    return sim_uuid
        except ValueError as e:
            raise e
        except Exception as e:
            raise Exception(f"Błąd bazy danych przy dodawaniu symulacji: {str(e)}")
        
    def get_simulations(self):
        try:
            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT 
                            simulation.sim_uuid, 
                            simulation.training_job_uuid, 
                            status.name, 
                            simulation.samples_simulation, 
                            simulation.predicted_samples, 
                            simulation.created_at,
                            STRING_AGG(strategy.name, ', ')
                        FROM simulation
                        JOIN status ON simulation.status_id = status.id
                        JOIN simulation_strategy ON simulation.sim_uuid = simulation_strategy.simulation_uuid
                        JOIN strategy ON simulation_strategy.strategy_id = strategy.id
                        GROUP BY simulation.sim_uuid, status.name
                        ORDER BY simulation.created_at DESC
                    """)
                    rows = cur.fetchall()
                    
                    return [
                        {
                            'sim_uuid': r[0],
                            'training_job_uuid': r[1],
                            'status': r[2],
                            'samples_simulation': r[3],
                            'predicted_samples': r[4],
                            'created_at': r[5],
                            'strategies': r[6]
                        } for r in rows
                    ]
        except Exception as e:
            raise Exception(f"Błąd podczas pobierania listy symulacji: {str(e)}")
        
    def del_simulation_job(self, sim_uuid):
        try:
            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM simulation WHERE sim_uuid = %s", (sim_uuid,))
                    if cur.rowcount == 0:
                        raise Exception(f"Symulacja {sim_uuid} nie istnieje.")
                    conn.commit()
            return True
        except Exception as e:
            raise Exception(f"Błąd podczas usuwania symulacji: {str(e)}")
        
    def update_simulation_status(self, sim_uuid, status_name):
        """
        Updates the status of a specific simulation job in the 'simulation' table.
        """
        try:
            if not sim_uuid or not status_name:
                raise ValueError("Błąd: Brak UUID symulacji lub nazwy statusu.")

            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE simulation 
                        SET status_id = (SELECT id FROM status WHERE name = %s)
                        WHERE sim_uuid = %s
                    """, (status_name, sim_uuid))
                    
                    if cur.rowcount == 0:
                        raise Exception(f"Nie znaleziono symulacji o UUID: {sim_uuid}")

                    conn.commit()
                    return True
        except Exception as e:
            raise Exception(f"Nie udało się zaktualizować statusu symulacji: {str(e)}")
        
    def get_simulation_config(self, sim_uuid):
        """
        Pobiera konfigurację konkretnej symulacji na podstawie jej UUID.
        """
        try:
            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT 
                            simulation.samples_simulation, 
                            simulation.predicted_samples,
                            ARRAY_AGG(strategy.name) as strategies,
                            simulation.training_job_uuid
                        FROM simulation
                        JOIN simulation_strategy ON simulation.sim_uuid = simulation_strategy.simulation_uuid
                        JOIN strategy ON simulation_strategy.strategy_id = strategy.id
                        WHERE simulation.sim_uuid = %s
                        GROUP BY simulation.sim_uuid, simulation.training_job_uuid
                    """, (sim_uuid,))
                    
                    row = cur.fetchone()
                    config = None
                    
                    if row:
                        config = {
                            "samples_simulation": row[0],
                            "predicted_samples": row[1],
                            "strategies": row[2],
                            "train_uuid": row[3]
                        }
                        
                    return config
        except Exception as e:
            raise Exception(f"Błąd bazy danych przy pobieraniu konfiguracji symulacji: {str(e)}")
        
    def save_simulation_result(self, sim_uuid, strategy_name, arch_name, data_bytes):
        query = """
            INSERT INTO result (sim_uuid, strategy_id, architecture_id, data)
            VALUES (
                %s, 
                (SELECT id FROM strategy WHERE name = %s), 
                (SELECT id FROM architecture WHERE name = %s), 
                %s
            )
            ON CONFLICT (sim_uuid, strategy_id, architecture_id) 
            DO UPDATE SET data = EXCLUDED.data
        """
        try:
            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (sim_uuid, strategy_name, arch_name, psycopg2.Binary(data_bytes)))
                    conn.commit()
                    return True
        except Exception as e:
            raise Exception(f"Błąd bazy danych przy zapisywaniu wyniku: {str(e)}")

    def load_simulation_result(self, sim_uuid, strategy_name, arch_name):
        query = """
            SELECT data 
            FROM result 
            WHERE sim_uuid = %s 
            AND strategy_id = (SELECT id FROM strategy WHERE name = %s)
            AND architecture_id = (SELECT id FROM architecture WHERE name = %s)
        """
        try:
            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (sim_uuid, strategy_name, arch_name))
                    result = cur.fetchone()
                    
                    if result:
                        # result[0] zawiera memoryview/bytes z kolumny BYTEA
                        return bytes(result[0])
                    return None
        except Exception as e:
            raise Exception(f"Błąd bazy danych przy odczycie wyniku: {str(e)}")
        

    def get_training_config(self, job_uuid):
        try:
            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cur:
                    config = {
                        'instrument': {},
                        'timeframe': {},
                        'parameter_set': {},
                        'base_columns': [],
                        'calculated_columns': [],
                        'features': [],
                        'targets': [],
                        'architectures': [],
                        'data_source': None,
                        'feature_names': [],
                        'target_names': []
                    }
                    
                    cur.execute("""
                        SELECT 
                            instrument.name, 
                            instrument.ticker
                        FROM instrument
                        JOIN training_job ON training_job.instrument_id = instrument.id
                        WHERE training_job.job_uuid = %s
                    """, (job_uuid,))
                    instrument = cur.fetchone()
                    if instrument:
                        config['instrument'] = {
                            'name': instrument[0],
                            'ticker': instrument[1]
                        }

                    cur.execute("""
                        SELECT 
                            timeframe.name, 
                            timeframe.range,
                            timeframe.check_period,
                            timeframe.min_count
                        FROM timeframe
                        JOIN training_job ON training_job.timeframe_id = timeframe.id
                        WHERE training_job.job_uuid = %s
                    """, (job_uuid,))
                    timeframe = cur.fetchone()
                    if timeframe:
                        config['timeframe'] = {
                            'name': timeframe[0],
                            'range': timeframe[1],
                            'check_period': timeframe[2],
                            'min_count': timeframe[3]
                        }

                    cur.execute("""
                        SELECT 
                            samples_limit, 
                            test_samples, 
                            seed, 
                            epochs, 
                            train_noise, 
                            learning_rate
                        FROM parameter_set
                        WHERE training_job_uuid = %s
                    """, (job_uuid,))
                    parameter_set = cur.fetchone()
                    if parameter_set:
                        config['parameter_set'] = {
                            'samples_limit': parameter_set[0],
                            'test_samples': parameter_set[1],
                            'seed': parameter_set[2],
                            'epochs': parameter_set[3],
                            'train_noise': parameter_set[4],
                            'learning_rate': parameter_set[5]
                        }

                    cur.execute("SELECT base_column.name FROM base_column ORDER BY id")
                    base_columns = cur.fetchall()
                    config['base_columns'] = [base_column[0] for base_column in base_columns]

                    cur.execute("SELECT calculated_column.name FROM calculated_column ORDER BY id")
                    calculated_columns = cur.fetchall()
                    config['calculated_columns'] = [calculated_column[0] for calculated_column in calculated_columns]

                    cur.execute("""
                        SELECT 
                            feature_type.name, 
                            feature_def.feature_periods, 
                            feature_def.shift
                        FROM feature_def
                        JOIN feature_type ON feature_def.feature_type_id = feature_type.id
                        WHERE feature_def.training_job_uuid = %s
                        ORDER BY feature_def.id
                    """, (job_uuid,))
                    features = cur.fetchall()
                    for f_type, f_periods, f_shift in features:
                        config['features'].append({
                            'feature_type': f_type, 
                            'feature_periods': f_periods,
                            'shift': f_shift
                        })
                        periods_str = "-".join(map(str, f_periods))
                        config['feature_names'].append(f"{f_type}:{periods_str}:{f_shift}")

                    cur.execute("""
                        SELECT 
                            base_column_id, 
                            calculated_column_id, 
                            shift 
                        FROM target_def 
                        WHERE training_job_uuid = %s 
                        ORDER BY id
                    """, (job_uuid,))
                    targets = cur.fetchall()

                    for base_column_id, calculated_column_id, target_shift in targets:
                        target_name = None
                        
                        if base_column_id is not None:
                            cur.execute("SELECT base_column.name FROM base_column WHERE id = %s", (base_column_id,))
                            base_column = cur.fetchone()
                            if base_column:
                                target_name = base_column[0]
                                
                        elif calculated_column_id is not None:
                            cur.execute("SELECT calculated_column.name FROM calculated_column WHERE id = %s", (calculated_column_id,))
                            calculated_column = cur.fetchone()
                            if calculated_column:
                                target_name = calculated_column[0]
                                
                        if target_name:
                            config['targets'].append({
                                'column': target_name, 
                                'shift': target_shift
                            })
                            config['target_names'].append(f"{target_name}:{target_shift}")

                    cur.execute("""
                        SELECT architecture.name
                        FROM architecture
                        JOIN training_job_architecture ON training_job_architecture.architecture_id = architecture.id
                        WHERE training_job_architecture.training_job_uuid = %s
                        ORDER BY architecture_id
                    """, (job_uuid,))
                    architectures = cur.fetchall()
                    config['architectures'] = [architecture[0] for architecture in architectures]

                    cur.execute("""
                        SELECT data_source.name
                        FROM data_source
                        JOIN training_job ON training_job.data_source_id = data_source.id
                        WHERE training_job.job_uuid = %s
                    """, (job_uuid,))
                    data_source = cur.fetchone()
                    if data_source:
                        config['data_source'] = data_source[0]

                    if not config['data_source']:
                        raise ValueError(f"Nie znaleziono danych dla zadania: {job_uuid}")

                    return config
        except Exception as e:
            raise Exception(f"Błąd podczas pobierania konfiguracji: {str(e)}")