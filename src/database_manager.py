import psycopg2
import uuid

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

                    # ZAKOMENTOWANY STARY BLOK:
                    # cur.execute("""
                    #     INSERT INTO parameter_set (training_job_uuid, samples_limit, train_ratio, seed, epochs, train_noise, learning_rate) 
                    #     VALUES (%s, %s, %s, %s, %s, %s, %s)
                    # """, (
                    #     job_uuid, 
                    #     config['parameter_set']['samples_limit'], 
                    #     config['parameter_set']['train_ratio'], 
                    #     config['parameter_set']['seed'], 
                    #     config['parameter_set']['epochs'], 
                    #     config['parameter_set']['train_noise'], 
                    #     config['parameter_set']['learning_rate']
                    # ))

                    # NOWY BLOK KODU:
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
                        cur.execute("""
                            INSERT INTO target_def (training_job_uuid, base_column_id, shift) 
                            VALUES (%s, (SELECT id FROM base_column WHERE name = %s), %s)
                        """, (job_uuid, target['base_column'], target['shift']))

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
        
    def get_training_config(self, job_uuid):
        try:
            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cur:
                    config = {
                        'instrument': {},
                        'timeframe': {},
                        'parameter_set': {},
                        'base_columns': [],
                        'features': [],
                        'targets': [],
                        'architectures': [],
                        'data_source': None
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

                    # ZAKOMENTOWANY STARY BLOK:
                    # cur.execute("""
                    #     SELECT 
                    #         samples_limit, 
                    #         train_ratio, 
                    #         seed, 
                    #         epochs, 
                    #         train_noise, 
                    #         learning_rate
                    #     FROM parameter_set
                    #     WHERE training_job_uuid = %s
                    # """, (job_uuid,))
                    # parameter_set = cur.fetchone()
                    # if parameter_set:
                    #     config['parameter_set'] = {
                    #         'samples_limit': parameter_set[0],
                    #         'train_ratio': parameter_set[1],
                    #         'seed': parameter_set[2],
                    #         'epochs': parameter_set[3],
                    #         'train_noise': parameter_set[4],
                    #         'learning_rate': parameter_set[5]
                    #     }

                    # NOWY BLOK KODU:
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

                    cur.execute("SELECT base_column.name FROM base_column")
                    base_columns = cur.fetchall()
                    config['base_columns'] = [base_column[0] for base_column in base_columns]

                    cur.execute("""
                        SELECT 
                            feature_type.name, 
                            feature_def.feature_periods, 
                            feature_def.shift
                        FROM feature_def
                        JOIN feature_type ON feature_def.feature_type_id = feature_type.id
                        WHERE feature_def.training_job_uuid = %s
                    """, (job_uuid,))
                    features = cur.fetchall()
                    config['features'] = [
                        {
                            'feature_type': feature[0], 
                            'feature_periods': feature[1],
                            'shift': feature[2]
                        } for feature in features
                    ]

                    cur.execute("""
                        SELECT base_column.name, target_def.shift 
                        FROM target_def
                        JOIN base_column ON target_def.base_column_id = base_column.id
                        WHERE target_def.training_job_uuid = %s
                    """, (job_uuid,))
                    targets = cur.fetchall()
                    config['targets'] = [
                        {
                            'base_column': target[0], 
                            'shift': target[1]
                        } for target in targets
                    ]

                    cur.execute("""
                        SELECT architecture.name
                        FROM architecture
                        JOIN training_job_architecture ON training_job_architecture.architecture_id = architecture.id
                        WHERE training_job_architecture.training_job_uuid = %s
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

    def update_training_config(self, job_uuid, config):
        try:
            if not config.get('features') or not config.get('targets') or not config.get('architectures'):
                raise ValueError("Błąd walidacji: Konfiguracja musi zawierać targety, feature'y i architektury.")

            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE training_job 
                        SET instrument_id = (SELECT id FROM instrument WHERE name = %s),
                            timeframe_id = (SELECT id FROM timeframe WHERE name = %s),
                            data_source_id = (SELECT id FROM data_source WHERE name = %s)
                        WHERE job_uuid = %s
                    """, (
                        config['instrument']['name'], 
                        config['timeframe']['name'], 
                        config['data_source'], 
                        job_uuid
                    ))

                    # ZAKOMENTOWANY STARY BLOK:
                    # cur.execute("""
                    #     UPDATE parameter_set 
                    #     SET samples_limit = %s, train_ratio = %s, seed = %s, 
                    #         epochs = %s, train_noise = %s, learning_rate = %s
                    #     WHERE training_job_uuid = %s
                    # """, (
                    #     config['parameter_set']['samples_limit'], 
                    #     config['parameter_set']['train_ratio'], 
                    #     config['parameter_set']['seed'], 
                    #     config['parameter_set']['epochs'], 
                    #     config['parameter_set']['train_noise'], 
                    #     config['parameter_set']['learning_rate'], 
                    #     job_uuid
                    # ))

                    # NOWY BLOK KODU:
                    cur.execute("""
                        UPDATE parameter_set 
                        SET samples_limit = %s, test_samples = %s, seed = %s, 
                            epochs = %s, train_noise = %s, learning_rate = %s
                        WHERE training_job_uuid = %s
                    """, (
                        config['parameter_set']['samples_limit'], 
                        config['parameter_set']['test_samples'], 
                        config['parameter_set']['seed'], 
                        config['parameter_set']['epochs'], 
                        config['parameter_set']['train_noise'], 
                        config['parameter_set']['learning_rate'], 
                        job_uuid
                    ))

                    cur.execute("DELETE FROM feature_def WHERE training_job_uuid = %s", (job_uuid,))
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

                    cur.execute("DELETE FROM target_def WHERE training_job_uuid = %s", (job_uuid,))
                    for target in config['targets']:
                        cur.execute("""
                            INSERT INTO target_def (training_job_uuid, base_column_id, shift) 
                            VALUES (%s, (SELECT id FROM base_column WHERE name = %s), %s)
                        """, (job_uuid, target['base_column'], target['shift']))

                    cur.execute("DELETE FROM training_job_architecture WHERE training_job_uuid = %s", (job_uuid,))
                    for architecture in config['architectures']:
                        cur.execute("""
                            INSERT INTO training_job_architecture (training_job_uuid, architecture_id) 
                            VALUES (%s, (SELECT id FROM architecture WHERE name = %s))
                        """, (job_uuid, architecture))

                    conn.commit()
                    return True
        except ValueError as e:
            raise e
        except Exception as e:
            raise Exception(f"Błąd podczas aktualizacji konfiguracji: {str(e)}")
        
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
