import psycopg2
import uuid

class DatabaseManager:
    def __init__(self, db_config):
        self.config = db_config

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
            return []

    def del_training_job(self, job_uuid):
        try:
            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM training_job WHERE job_uuid = %s", (job_uuid,))
                    
                    if cur.rowcount > 0:
                        conn.commit()
                        return True
                    else:
                        return False
        except Exception as e:
            return False

    def get_training_config(self, job_uuid):
        try:
            with psycopg2.connect(**self.config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT 
                            instrument.name, 
                            instrument.ticker,
                            timeframe.name,
                            data_source.name,
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
                        'instrument_name': row[0],
                        'instrument_ticker': row[1],
                        'timeframe': row[2],
                        'data_source': row[3],
                        'samples_limit': row[4],
                        'train_ratio': row[5],
                        'seed': row[6],
                        'epochs': row[7],
                        'train_noise': row[8],
                        'learning_rate': row[9],
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
                        config['targets'].append(f"{r[0]}:{r[1]}")
                    config['targets'] = ", ".join(config['targets'])

                    cur.execute("""
                        SELECT feature_type.name, feature_def.start_from, 
                               feature_def.stop_at, feature_def.step, feature_def.shift 
                        FROM feature_def
                        JOIN feature_type ON feature_def.feature_type_id = feature_type.id
                        WHERE feature_def.training_job_uuid = %s
                    """, (job_uuid,))
                    for r in cur.fetchall():
                        config['features'].append(f"{r[0]}:{r[1]}:{r[2]}:{r[3]}:{r[4]}")
                    config['features'] = ", ".join(config['features'])

                    cur.execute("""
                        SELECT architecture.name 
                        FROM training_job_architecture
                        JOIN architecture ON training_job_architecture.architecture_id = architecture.id
                        WHERE training_job_architecture.training_job_uuid = %s
                    """, (job_uuid,))
                    config['architectures'] = [r[0] for r in cur.fetchall()]
                    config['architectures'] = ", ".join(config['architectures'])

                    return config
        except Exception as e:
            return None

    def add_training_job(self, params):
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
                    """, (job_uuid, params['instrument_name'], params['timeframe'], params['data_source']))

                    cur.execute("""
                        INSERT INTO parameter_set (training_job_uuid, samples_limit, train_ratio, seed, epochs, train_noise, learning_rate) 
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (job_uuid, params['samples_limit'], params['train_ratio'], params['seed'], params['epochs'], params['train_noise'], params['learning_rate']))

                    targets = [item.strip() for item in params['targets'].split(",")]
                    for target in targets:
                        t = [item.strip() for item in target.split(":")]
                        cur.execute("""
                            INSERT INTO target_def (training_job_uuid, target_type_id, shift) 
                            VALUES (%s, (SELECT id FROM target_type WHERE name = %s), %s)
                        """, (job_uuid, t[0], t[1]))

                    features = [item.strip() for item in params['features'].split(",")]
                    for feature in features:
                        f = [item.strip() for item in feature.split(":")]
                        cur.execute("""
                            INSERT INTO feature_def (training_job_uuid, feature_type_id, start_from, stop_at, step, shift) 
                            VALUES (%s, (SELECT id FROM feature_type WHERE name = %s), %s, %s, %s, %s)
                        """, (job_uuid, f[0], f[1], f[2], f[3], f[4]))

                    architectures = [item.strip() for item in params['architectures'].split(",")]
                    for architecture in architectures:
                        cur.execute("""
                            INSERT INTO training_job_architecture (training_job_uuid, architecture_id) 
                            VALUES (%s, (SELECT id FROM architecture WHERE name = %s))
                        """, (job_uuid, architecture))

                    conn.commit()
                    
                    return job_uuid
        except Exception as e:
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
                    return True
        except Exception as e:
            return False
        
