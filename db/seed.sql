INSERT INTO status (name) VALUES ('pending'), ('running'), ('completed'), ('failed'), ('stopping'), ('created');
INSERT INTO base_column (name) VALUES ('close'), ('high'), ('low'), ('open');
INSERT INTO calculated_column (name) VALUES ('typical'), ('median'), ('weighted'), ('ohlc');
INSERT INTO feature_type (name) VALUES ('sma'), ('ema'), ('rsi'), ('pct');
INSERT INTO architecture (name) VALUES ('ModelV1');
INSERT INTO timeframe (name, range, check_period, min_count) VALUES ('1d', 'max', 'M', 18);
INSERT INTO timeframe (name, range) VALUES ('2m', '60d'), ('5m', '60d'), ('15m', '60d');
INSERT INTO data_source (name) VALUES ('YF');
INSERT INTO role (name) VALUES ('admin'), ('user');

INSERT INTO instrument (name, ticker) VALUES
('EURUSD', 'EURUSD=X'), ('USDJPY', 'JPY=X'), ('GBPUSD', 'GBPUSD=X'),
('AUDUSD', 'AUDUSD=X'), ('USDCAD', 'CAD=X'), ('USDCHF', 'CHF=X'),
('NZDUSD', 'NZDUSD=X'), ('EURJPY', 'EURJPY=X'), ('GBPJPY', 'GBPJPY=X'),
('EURGBP', 'EURGBP=X'), ('EURCHF', 'EURCHF=X'), ('AUDJPY', 'AUDJPY=X'),
('USDCNH', 'CNY=X'), ('USDHKD', 'HKD=X'), ('USDSGD', 'SGD=X'),
('USDTRY', 'TRY=X'), ('USDMXN', 'MXN=X'), ('USDZAR', 'ZAR=X'),
('USDBRL', 'BRL=X'), ('USDINR', 'INR=X'), ('EURAUD', 'EURAUD=X'),
('EURCAD', 'EURCAD=X'), ('GBPAUD', 'GBPAUD=X'), ('GBPCAD', 'GBPCAD=X'),
('AUDNZD', 'AUDNZD=X'), ('CADJPY', 'CADJPY=X'), ('CHFJPY', 'CHFJPY=X'),
('NZDJPY', 'NZDJPY=X'), ('USDNOK', 'NOK=X'), ('USDSEK', 'SEK=X');

INSERT INTO instrument (name, ticker) VALUES
('US500', '^GSPC'), ('US100', '^NDX'), ('US30', '^DJI'),
('DE40', '^GDAXI'), ('UK100', '^FTSE'), ('JP225', '^N225'),
('EU50', '^STOXX50E'), ('HANGSENG', '^HSI'), ('FRA40', '^FCHI'),
('ASX200', '^AXJO'), ('VIX', '^VIX'), ('US2000', '^RUT'),
('NIFTY50', '^NSEI'), ('TSX', '^GSPTSE'), ('CHINAA50', '000300.SS'),
('BOVESPA', '^BVSP');

INSERT INTO instrument (name, ticker) VALUES
('WTI', 'CL=F'), ('BRENT', 'BZ=F'), ('GOLD', 'GC=F'),
('NATGAS', 'NG=F'), ('SILVER', 'SI=F'), ('COPPER', 'HG=F'),
('CORN', 'ZC=F'), ('SOYBEAN', 'ZS=F'), ('WHEAT', 'ZW=F'),
('PLATINUM', 'PL=F'), ('PALLADIUM', 'PA=F'), ('ALUMINIUM', 'ALI=F'),
('NICKEL', 'NI=F'), ('ZINC', 'ZN=F'), ('COFFEE', 'KC=F'),
('SUGAR', 'SB=F');

INSERT INTO instrument (name, ticker) VALUES
('BITCOIN', 'BTC-USD'), ('ETHEREUM', 'ETH-USD'), ('SOLANA', 'SOL-USD'),
('BINANCECOIN', 'BNB-USD'), ('RIPPLE', 'XRP-USD'), ('CARDANO', 'ADA-USD'),
('DOGECOIN', 'DOGE-USD'), ('TONCOIN', 'TON-USD'), ('AVALANCHE', 'AVAX-USD'),
('TRON', 'TRX-USD'), ('POLKADOT', 'DOT-USD'), ('CHAINLINK', 'LINK-USD'),
('POLYGON', 'MATIC-USD');
