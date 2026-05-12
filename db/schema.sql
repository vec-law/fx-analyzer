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

CREATE TABLE calculated_column (
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

CREATE TABLE role (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL
);

CREATE TABLE app_user (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    password_hash BYTEA NOT NULL,
    role_id INTEGER REFERENCES role(id) NOT NULL,
    session_token UUID DEFAULT NULL
);

CREATE TABLE training (
    train_uuid UUID PRIMARY KEY,
    instrument_id INTEGER REFERENCES instrument(id) NOT NULL,
    timeframe_id INTEGER REFERENCES timeframe(id) NOT NULL,
    status_id INTEGER REFERENCES status(id) NOT NULL,
	data_source_id INTEGER REFERENCES data_source(id) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE parameter_set (
    train_uuid UUID PRIMARY KEY REFERENCES training(train_uuid) ON DELETE CASCADE,
    all_samples INTEGER NOT NULL CHECK (all_samples > 0),
    test_samples INTEGER NOT NULL CHECK (test_samples >= 0 AND test_samples < all_samples),
    seed INTEGER NOT NULL CHECK (seed > 0),
    epochs INTEGER NOT NULL CHECK (epochs > 0),
    train_noise REAL NOT NULL CHECK (train_noise >= 0 AND train_noise < 1),
    learning_rate REAL NOT NULL CHECK (learning_rate > 0 AND learning_rate < 1)
);

CREATE TABLE feature_def (
    id SERIAL PRIMARY KEY,
    train_uuid UUID REFERENCES training(train_uuid) ON DELETE CASCADE,
    feature_type_id INTEGER REFERENCES feature_type(id) NOT NULL,
    feature_periods INTEGER[] NOT NULL,
    shift INTEGER NOT NULL CHECK (shift > 0),
    CONSTRAINT uq_feature_def UNIQUE (train_uuid, feature_type_id, feature_periods, shift),
    CONSTRAINT feature_periods_not_empty CHECK (array_length(feature_periods, 1) > 0),
    CONSTRAINT feature_periods_positive CHECK (0 < ALL(feature_periods))
);

CREATE TABLE target_def (
    id SERIAL PRIMARY KEY,
    train_uuid UUID REFERENCES training(train_uuid) ON DELETE CASCADE,
    base_column_id INTEGER REFERENCES base_column(id),
    calculated_column_id INTEGER REFERENCES calculated_column(id),
    shift INTEGER NOT NULL CHECK (shift < 0),
    CONSTRAINT uq_target_def UNIQUE (train_uuid, base_column_id, calculated_column_id, shift),
    CONSTRAINT check_single_source CHECK (
        (base_column_id IS NOT NULL AND calculated_column_id IS NULL) OR 
        (base_column_id IS NULL AND calculated_column_id IS NOT NULL)
    )
);

CREATE TABLE training_architecture (
    train_uuid UUID REFERENCES training(train_uuid) ON DELETE CASCADE,
    architecture_id INTEGER REFERENCES architecture(id),
    PRIMARY KEY (train_uuid, architecture_id)
);

CREATE TABLE statistic (
    id SERIAL PRIMARY KEY,
    train_uuid UUID NOT NULL REFERENCES training(train_uuid) ON DELETE CASCADE,
    column_name TEXT NOT NULL,
    stat_name TEXT NOT NULL,
    stat_value REAL NOT NULL,
    CONSTRAINT uq_stat_entry UNIQUE (train_uuid, column_name, stat_name)
);

CREATE TABLE model (
    id SERIAL PRIMARY KEY,
    train_uuid UUID NOT NULL REFERENCES training(train_uuid) ON DELETE CASCADE,
    architecture_id INTEGER NOT NULL REFERENCES architecture(id),
    weights BYTEA NOT NULL,
    mse_loss REAL,
    mae_loss REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_model_arch UNIQUE (train_uuid, architecture_id)
);

CREATE TABLE prediction (
    pred_uuid UUID PRIMARY KEY,
    train_uuid UUID NOT NULL REFERENCES training(train_uuid) ON DELETE CASCADE,
    status_id INTEGER NOT NULL REFERENCES status(id),
    all_samples INTEGER NOT NULL CHECK (all_samples > 0),
    predicted_samples INTEGER NOT NULL CHECK (predicted_samples > 0),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT chk_samples CHECK (predicted_samples <= all_samples)
);

CREATE TABLE prediction_result (
    pred_uuid UUID NOT NULL REFERENCES prediction(pred_uuid) ON DELETE CASCADE,
    architecture_id INTEGER NOT NULL REFERENCES architecture(id),
    data BYTEA NOT NULL,
    PRIMARY KEY (pred_uuid, architecture_id)
);

CREATE TABLE app_user_training (
    app_user_id INTEGER NOT NULL REFERENCES app_user(id) ON DELETE CASCADE,
    train_uuid UUID NOT NULL REFERENCES training(train_uuid) ON DELETE CASCADE,
    PRIMARY KEY (app_user_id, train_uuid)
);


CREATE INDEX idx_pred_train_uuid ON prediction(train_uuid);
CREATE INDEX idx_model_train_uuid ON model(train_uuid);
CREATE INDEX idx_statistic_train_uuid ON statistic(train_uuid);
CREATE INDEX idx_target_def_train_uuid ON target_def(train_uuid);
CREATE INDEX idx_feature_def_train_uuid ON feature_def(train_uuid);
CREATE INDEX idx_train_arch_uuid ON training_architecture(train_uuid);

INSERT INTO status (name) VALUES ('pending'), ('running'), ('completed'), ('failed');
INSERT INTO instrument (name, ticker) VALUES ('EURUSD', 'EURUSD=X'), ('US100', '^NDX');
INSERT INTO base_column (name) VALUES ('close'), ('high'), ('low'), ('open');
INSERT INTO calculated_column (name) VALUES ('typical'), ('median'), ('weighted'), ('ohlc');
INSERT INTO feature_type (name) VALUES ('sma'), ('ema'), ('rsi'), ('pct');
INSERT INTO architecture (name) VALUES ('ModelV1');
INSERT INTO timeframe (name, range, check_period, min_count) VALUES ('1d', 'max', 'M', 18);
INSERT INTO timeframe (name, range) VALUES ('2m', '60d'), ('5m', '60d'), ('15m', '60d');
INSERT INTO data_source (name) VALUES ('YF');
INSERT INTO role (name) VALUES ('admin'), ('user');
