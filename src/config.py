DATASET_NAME = 'santosh3110/credit_card_fraud_transactions'
RANDOM_STATE = 42
CUSTOMER_COL = 'cc_num'
TIME_COL = 'trans_date_trans_time'
AMOUNT_COL = 'amt'
CAT_COL = 'category'
TARGET = 'is_fraud'
HISTORY_WEEKS = [1,4,12]  # 7d,30d,90d approx
HORIZON_DAYS = 30
MIN_HISTORY_TX = 5
SEQ_LEN = 20
BATCH_SIZE = 256
EPOCHS = 5
LR = 1e-3

