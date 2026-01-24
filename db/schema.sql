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
    shift INTEGER NOT NULL,
    CONSTRAINT uq_feature_def UNIQUE (training_job_uuid, feature_type_id, feature_periods, shift),
    CONSTRAINT feature_periods_not_empty CHECK (array_length(feature_periods, 1) > 0),
    CONSTRAINT feature_periods_positive CHECK (0 < ALL(feature_periods))
);

CREATE TABLE target_def (
    id SERIAL PRIMARY KEY,
    training_job_uuid UUID REFERENCES training_job(job_uuid) ON DELETE CASCADE,
    base_column_id INTEGER REFERENCES base_column(id) NOT NULL,
    shift INTEGER NOT NULL,
    CONSTRAINT uq_target_def UNIQUE (training_job_uuid, base_column_id, shift)
    
);

CREATE TABLE training_job_architecture (
    training_job_uuid UUID REFERENCES training_job(job_uuid) ON DELETE CASCADE,
    architecture_id INTEGER REFERENCES architecture(id),
    PRIMARY KEY (training_job_uuid, architecture_id)
);

CREATE TABLE experiment (
    training_job_uuid UUID REFERENCES training_job(job_uuid) ON DELETE CASCADE,
    architecture_id INTEGER REFERENCES architecture(id),
    file_path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (training_job_uuid, architecture_id)
);

CREATE INDEX idx_target_def_job_uuid ON target_def(training_job_uuid);
CREATE INDEX idx_feature_def_job_uuid ON feature_def(training_job_uuid);
CREATE INDEX idx_experiment_job_uuid ON experiment(training_job_uuid);
CREATE INDEX idx_tj_arch_job_uuid ON training_job_architecture(training_job_uuid);

INSERT INTO status (name) VALUES ('pending'), ('running'), ('completed'), ('failed');
INSERT INTO instrument (name, ticker) VALUES ('EURUSD', 'EURUSD=X');
INSERT INTO base_column (name) VALUES ('close'), ('high'), ('low'), ('open');
INSERT INTO feature_type (name) VALUES ('sma'), ('ema'), ('rsi');
INSERT INTO architecture (name) VALUES ('MLP_Base'), ('MLP_Extended');
INSERT INTO timeframe (name, range, check_period, min_count) VALUES ('1d', 'max', 'M', 18);
INSERT INTO data_source (name) VALUES ('YF');
