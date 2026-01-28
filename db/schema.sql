CREATE TABLE instrument (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    ticker TEXT UNIQUE NOT NULL
);

CREATE TABLE timeframe (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
	range TEXT NOT NULL,
    check_period TEXT,
	min_count INTEGER
);

CREATE TABLE architecture (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL
);

CREATE TABLE base_column (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL
);

CREATE TABLE feature_type (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL
);

CREATE TABLE status (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL
);

CREATE TABLE data_source (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL
);

CREATE TABLE training_job (
    job_uuid UUID PRIMARY KEY,
    instrument_id INTEGER REFERENCES instrument(id) NOT NULL,
    timeframe_id INTEGER REFERENCES timeframe(id) NOT NULL,
    status_id INTEGER REFERENCES status(id) NOT NULL,
	data_source_id INTEGER REFERENCES data_source(id) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE parameter_set (
    training_job_uuid UUID PRIMARY KEY REFERENCES training_job(job_uuid) ON DELETE CASCADE,
    samples_limit INTEGER NOT NULL CHECK (samples_limit > 0),
    test_samples INTEGER NOT NULL CHECK (test_samples >= 0 AND test_samples < samples_limit),
    seed INTEGER NOT NULL CHECK (seed > 0),
    epochs INTEGER NOT NULL CHECK (epochs > 0),
    train_noise REAL NOT NULL CHECK (train_noise >= 0 AND train_noise < 1),
    learning_rate REAL NOT NULL CHECK (learning_rate > 0 AND learning_rate < 1)
);

CREATE TABLE feature_def (
    id SERIAL PRIMARY KEY,
    training_job_uuid UUID REFERENCES training_job(job_uuid) ON DELETE CASCADE,
    feature_type_id INTEGER REFERENCES feature_type(id) NOT NULL,
    feature_periods INTEGER[] NOT NULL,
    shift INTEGER NOT NULL CHECK (shift > 0),
    CONSTRAINT uq_feature_def UNIQUE (training_job_uuid, feature_type_id, feature_periods, shift),
    CONSTRAINT feature_periods_not_empty CHECK (array_length(feature_periods, 1) > 0),
    CONSTRAINT feature_periods_positive CHECK (0 < ALL(feature_periods))
);

CREATE TABLE target_def (
    id SERIAL PRIMARY KEY,
    training_job_uuid UUID REFERENCES training_job(job_uuid) ON DELETE CASCADE,
    base_column_id INTEGER REFERENCES base_column(id) NOT NULL,
    shift INTEGER NOT NULL CHECK (shift < 0),
    CONSTRAINT uq_target_def UNIQUE (training_job_uuid, base_column_id, shift)
    
);

CREATE TABLE training_job_architecture (
    training_job_uuid UUID REFERENCES training_job(job_uuid) ON DELETE CASCADE,
    architecture_id INTEGER REFERENCES architecture(id),
    PRIMARY KEY (training_job_uuid, architecture_id)
);

CREATE TABLE statistic (
    id SERIAL PRIMARY KEY,
    training_job_uuid UUID NOT NULL REFERENCES training_job(job_uuid) ON DELETE CASCADE,
    column_name TEXT NOT NULL,
    stat_name TEXT NOT NULL,
    stat_value REAL NOT NULL,
    CONSTRAINT uq_stat_entry UNIQUE (training_job_uuid, column_name, stat_name)
);

CREATE TABLE model (
    id SERIAL PRIMARY KEY,
    training_job_uuid UUID NOT NULL REFERENCES training_job(job_uuid) ON DELETE CASCADE,
    architecture_id INTEGER NOT NULL REFERENCES architecture(id),
    weights BYTEA NOT NULL,
    mse_loss REAL,
    mae_loss REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_model_job_arch UNIQUE (training_job_uuid, architecture_id)
);

CREATE TABLE simulation (
    sim_uuid UUID PRIMARY KEY,
    training_job_uuid UUID NOT NULL REFERENCES training_job(job_uuid) ON DELETE CASCADE,
    status_id INTEGER NOT NULL REFERENCES status(id),
    samples_simulation INTEGER NOT NULL CHECK (samples_simulation > 0),
    predicted_samples INTEGER NOT NULL CHECK (predicted_samples > 0),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_predicted_samples_limit CHECK (predicted_samples <= samples_simulation)
);

CREATE TABLE strategy (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL
);

CREATE TABLE simulation_strategy (
    simulation_uuid UUID REFERENCES simulation(sim_uuid) ON DELETE CASCADE,
    strategy_id INTEGER REFERENCES strategy(id) ON DELETE CASCADE,
    PRIMARY KEY (simulation_uuid, strategy_id)
);

CREATE TABLE result (
    sim_uuid UUID NOT NULL REFERENCES simulation(sim_uuid) ON DELETE CASCADE,
    strategy_id INTEGER NOT NULL REFERENCES strategy(id),
    architecture_id INTEGER NOT NULL REFERENCES architecture(id),
    data BYTEA NOT NULL,
    PRIMARY KEY (sim_uuid, strategy_id, architecture_id)
);

CREATE INDEX idx_simulation_training_uuid ON simulation(training_job_uuid);
CREATE INDEX idx_sim_strat_sim_uuid ON simulation_strategy(simulation_uuid);
CREATE INDEX idx_model_job_uuid ON model(training_job_uuid);
CREATE INDEX idx_statistic_job_uuid ON statistic(training_job_uuid);
CREATE INDEX idx_target_def_job_uuid ON target_def(training_job_uuid);
CREATE INDEX idx_feature_def_job_uuid ON feature_def(training_job_uuid);
CREATE INDEX idx_tj_arch_job_uuid ON training_job_architecture(training_job_uuid);

INSERT INTO status (name) VALUES ('pending'), ('running'), ('completed'), ('failed');
INSERT INTO instrument (name, ticker) VALUES ('EURUSD', 'EURUSD=X');
INSERT INTO base_column (name) VALUES ('close'), ('high'), ('low'), ('open');
INSERT INTO feature_type (name) VALUES ('sma'), ('ema'), ('rsi');
INSERT INTO architecture (name) VALUES ('ModelV1'), ('ModelV2'), ('ModelV3'), ('ModelV4'), ('ModelV5'), ('ModelV6'), ('ModelV7'), ('ModelV8'), ('ModelV9'), ('ModelV10'),
('ModelV11'), ('ModelV12'), ('ModelV13'), ('ModelV14'), ('ModelV15'), ('ModelV16'), ('ModelV17'), ('ModelV18'), ('ModelV19'), ('ModelV20'),
('ModelV21'), ('ModelV22'), ('ModelV23'), ('ModelV24'), ('ModelV25'), ('ModelV26'), ('ModelV27'), ('ModelV28'), ('ModelV29'), ('ModelV30'),
('ModelV31'), ('ModelV32'), ('ModelV33'), ('ModelV34'), ('ModelV35'), ('ModelV36'), ('ModelV37'), ('ModelV38'), ('ModelV39'), ('ModelV40'),
('ModelV41'), ('ModelV42'), ('ModelV43'), ('ModelV44'), ('ModelV45'), ('ModelV46'), ('ModelV47'), ('ModelV48'), ('ModelV49'), ('ModelV50'),
('ModelV51'), ('ModelV52'), ('ModelV53'), ('ModelV54'), ('ModelV55'), ('ModelV56'), ('ModelV57'), ('ModelV58'), ('ModelV59'), ('ModelV60'), 
('ModelV61'), ('ModelV62'), ('ModelV63'), ('ModelV64'), ('ModelV65'), ('ModelV66'), ('ModelV67'), ('ModelV68'), ('ModelV69'), ('ModelV70'), 
('ModelV71'), ('ModelV72'), ('ModelV73'), ('ModelV74'), ('ModelV75'), ('ModelV76'), ('ModelV77'), ('ModelV78'), ('ModelV79'), ('ModelV80'), 
('ModelV81'), ('ModelV82'), ('ModelV83'), ('ModelV84'), ('ModelV85'), ('ModelV86'), ('ModelV87'), ('ModelV88'), ('ModelV89'), ('ModelV90'), 
('ModelV91'), ('ModelV92'), ('ModelV93'), ('ModelV94'), ('ModelV95'), ('ModelV96'), ('ModelV97'), ('ModelV98'), ('ModelV99'), ('ModelV100'),
('ModelV101'), ('ModelV102'), ('ModelV103'), ('ModelV104'), ('ModelV105'), ('ModelV106'), ('ModelV107'), ('ModelV108'), ('ModelV109'), ('ModelV110'),
('ModelV111'), ('ModelV112'), ('ModelV113'), ('ModelV114'), ('ModelV115'), ('ModelV116'), ('ModelV117'), ('ModelV118'), ('ModelV119'), ('ModelV120');
INSERT INTO timeframe (name, range, check_period, min_count) VALUES ('1d', 'max', 'M', 18);
INSERT INTO data_source (name) VALUES ('YF');
INSERT INTO strategy (name) VALUES ('only_predict');
