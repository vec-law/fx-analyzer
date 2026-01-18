import psycopg2
import uuid

class DatabaseManager:
    def __init__(self, db_config):
        self.config = db_config

    def drop_pending_trainings(self):
        try:
            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        DELETE FROM training_job
                        WHERE status_id = (SELECT id FROM status WHERE name = 'pending')
                        RETURNING job_uuid;
                    """)
                    deleted_rows = cur.fetchall()
                    deleted_uuids = [str(row[0]) for row in deleted_rows]
                    
                    conn.commit()
                    
                    if deleted_uuids:
                        print(f"[drop_pending_trainings] Usunięto {len(deleted_uuids)} zadanie(-a): {deleted_uuids}")
                    else:
                        print("[drop_pending_trainings] Nie znaleziono żadnych zadań do usunięcia.")
                        
                    return deleted_uuids
        except Exception as e:
            print(f"[drop_pending_trainings] Błąd: {e}")
            return None
        
    def get_pending_trainings(self):
        try:
            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT training_job.job_uuid 
                        FROM training_job
                        JOIN status ON training_job.status_id = status.id
                        WHERE status.name = 'pending'
                        ORDER BY training_job.created_at ASC
                    """)
                    return [row[0] for row in cur.fetchall()]
        except Exception as e:
            print(f"[get_pending_trainings] Błąd: {e}")
            return []

    def insert_training_job(self, params):
        job_uuid = str(uuid.uuid4())
        try:
            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO training_job (job_uuid, instrument_id, timeframe_id, status_id, data_source_id) 
                        VALUES (%s, 
                            (SELECT id FROM instrument WHERE name = %s), 
                            (SELECT id FROM timeframe WHERE name = %s), 
                            (SELECT id FROM status WHERE name = 'pending'),
                            (SELECT id FROM data_source WHERE name = %s))
                    """, (job_uuid, params['instrument'], params['timeframe'], params['data_source']))

                    cur.execute("""
                        INSERT INTO parameter_set (training_job_uuid, samples_limit, train_ratio, seed, epochs, train_noise, learning_rate) 
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (job_uuid, params['samples_limit'], params['train_ratio'], params['seed'], params['epochs'], params['train_noise'], params['learning_rate']))

                    for target in params['targets']:
                        cur.execute("""
                            INSERT INTO target_def (training_job_uuid, target_type_id, shift) 
                            VALUES (%s, (SELECT id FROM target_type WHERE name = %s), %s)
                        """, (job_uuid, target['type'], target['shift']))

                    for feature in params['features']:
                        cur.execute("""
                            INSERT INTO feature_def (training_job_uuid, feature_type_id, start_from, stop_at, step, shift) 
                            VALUES (%s, (SELECT id FROM feature_type WHERE name = %s), %s, %s, %s, %s)
                        """, (job_uuid, feature['type'], feature['start_from'], feature['stop_at'], feature['step'], feature['shift']))

                    for architecture in params['architectures']:
                        cur.execute("""
                            INSERT INTO training_job_architecture (training_job_uuid, architecture_id) 
                            VALUES (%s, (SELECT id FROM architecture WHERE name = %s))
                        """, (job_uuid, architecture))

                    conn.commit()

                    print(f"[insert_training_job] Dodano zadanie: {job_uuid}")
                    
                    return job_uuid
        except Exception as e:
            print(f"[insert_training_job] Błąd: {e}")
            return None
        
    def get_training_job_config(self, job_uuid):
        try:
            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT 
                            instrument.name, 
                            timeframe.name,
                            data_source.name, -- Dodano pobieranie źródła danych
                            parameter_set.samples_limit, 
                            parameter_set.train_ratio, 
                            parameter_set.seed, 
                            parameter_set.epochs, 
                            parameter_set.train_noise, 
                            parameter_set.learning_rate
                        FROM training_job
                        JOIN instrument ON training_job.instrument_id = instrument.id
                        JOIN timeframe ON training_job.timeframe_id = timeframe.id
                        JOIN data_source ON training_job.data_source_id = data_source.id -- Dodano JOIN
                        JOIN parameter_set ON training_job.job_uuid = parameter_set.training_job_uuid
                        WHERE training_job.job_uuid = %s
                    """, (job_uuid,))
                    
                    row = cur.fetchone()
                    if not row:
                        return None

                    config = {
                        'instrument': row[0],
                        'timeframe': row[1],
                        'data_source': row[2],
                        'samples_limit': row[3],
                        'train_ratio': row[4],
                        'seed': row[5],
                        'epochs': row[6],
                        'train_noise': row[7],
                        'learning_rate': row[8],
                        'targets': [],
                        'features': [],
                        'architectures': []
                    }

                    cur.execute("""
                        SELECT target_type.name, target_def.shift 
                        FROM target_def
                        JOIN target_type ON target_def.target_type_id = target_type.id
                        WHERE target_def.training_job_uuid = %s
                    """, (job_uuid,))
                    for r in cur.fetchall():
                        config['targets'].append({'type': r[0], 'shift': r[1]})

                    cur.execute("""
                        SELECT feature_type.name, feature_def.start_from, 
                               feature_def.stop_at, feature_def.step, feature_def.shift 
                        FROM feature_def
                        JOIN feature_type ON feature_def.feature_type_id = feature_type.id
                        WHERE feature_def.training_job_uuid = %s
                    """, (job_uuid,))
                    for r in cur.fetchall():
                        config['features'].append({
                            'type': r[0], 
                            'start_from': r[1], 
                            'stop_at': r[2], 
                            'step': r[3], 
                            'shift': r[4]
                        })

                    cur.execute("""
                        SELECT architecture.name 
                        FROM training_job_architecture
                        JOIN architecture ON training_job_architecture.architecture_id = architecture.id
                        WHERE training_job_architecture.training_job_uuid = %s
                    """, (job_uuid,))
                    config['architectures'] = [r[0] for r in cur.fetchall()]

                    print(f"[get_training_job_config] Pobrano konfigurację zadania: {job_uuid}")

                    return config
        except Exception as e:
            print(f"[get_training_job_config] Błąd: {e}")
            return None
        
    def update_training_job_status(self, job_uuid, status_name):
        try:
            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE training_job 
                        SET status_id = (SELECT id FROM status WHERE name = %s)
                        WHERE job_uuid = %s
                    """, (status_name, job_uuid))
                    conn.commit()
                    return True
        except Exception as e:
            print(f"[update_training_job_status] Błąd: {e}")
            return False
        
    def get_trainings_overview(self):
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
                    results = []
                    for r in rows:
                        results.append({
                            'job_uuid': r[0],
                            'instrument': r[1],
                            'timeframe': r[2],
                            'data_source': r[3],
                            'status': r[4],
                            'created_at': r[5]
                        })
                    return results
        except Exception as e:
            print(f"[get_trainings_overview] Błąd: {e}")
            return []
        
    def delete_training_job(self, job_uuid):
        try:
            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM training_job WHERE job_uuid = %s", (job_uuid,))
                    
                    if cur.rowcount > 0:
                        conn.commit()
                        print(f"[delete_training_job] Usunięto zadanie: {job_uuid}")
                        return True
                    else:
                        print(f"[delete_training_job] Nie znaleziono zadania o UUID: {job_uuid}")
                        return False
        except Exception as e:
            print(f"[delete_training_job] Błąd: {e}")
            return False

    def update_training_job_config(self, job_uuid, params):
        try:
            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cur:
                    # 1. Aktualizacja głównych parametrów w training_job
                    cur.execute("""
                        UPDATE training_job 
                        SET instrument_id = (SELECT id FROM instrument WHERE name = %s),
                            timeframe_id = (SELECT id FROM timeframe WHERE name = %s),
                            data_source_id = (SELECT id FROM data_source WHERE name = %s)
                        WHERE job_uuid = %s
                    """, (params['instrument'], params['timeframe'], params['data_source'], job_uuid))

                    # 2. Aktualizacja parameter_set
                    cur.execute("""
                        UPDATE parameter_set 
                        SET samples_limit = %s, train_ratio = %s, seed = %s, 
                            epochs = %s, train_noise = %s, learning_rate = %s
                        WHERE training_job_uuid = %s
                    """, (params['samples_limit'], params['train_ratio'], params['seed'], 
                          params['epochs'], params['train_noise'], params['learning_rate'], job_uuid))

                    # 3. Odświeżenie target_def (usuń i wstaw)
                    cur.execute("DELETE FROM target_def WHERE training_job_uuid = %s", (job_uuid,))
                    for target in params['targets']:
                        cur.execute("""
                            INSERT INTO target_def (training_job_uuid, target_type_id, shift) 
                            VALUES (%s, (SELECT id FROM target_type WHERE name = %s), %s)
                        """, (job_uuid, target['type'], target['shift']))

                    # 4. Odświeżenie feature_def (usuń i wstaw)
                    cur.execute("DELETE FROM feature_def WHERE training_job_uuid = %s", (job_uuid,))
                    for feature in params['features']:
                        cur.execute("""
                            INSERT INTO feature_def (training_job_uuid, feature_type_id, start_from, stop_at, step, shift) 
                            VALUES (%s, (SELECT id FROM feature_type WHERE name = %s), %s, %s, %s, %s)
                        """, (job_uuid, feature['type'], feature['start_from'], feature['stop_at'], feature['step'], feature['shift']))

                    # 5. Odświeżenie architektur (usuń i wstaw)
                    cur.execute("DELETE FROM training_job_architecture WHERE training_job_uuid = %s", (job_uuid,))
                    for architecture in params['architectures']:
                        cur.execute("""
                            INSERT INTO training_job_architecture (training_job_uuid, architecture_id) 
                            VALUES (%s, (SELECT id FROM architecture WHERE name = %s))
                        """, (job_uuid, architecture))

                    conn.commit()
                    print(f"[update_training_job_config] Zaktualizowano konfigurację zadania: {job_uuid}")
                    return True
        except Exception as e:
            print(f"[update_training_job_config] Błąd: {e}")
            return False
        
