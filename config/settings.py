"""Configuration settings for the trading system"""

# Data settings
DATA_PATH = "data/prices.txt"

# Trading parameters
N_INSTRUMENTS = 50
COMMISSION_RATE = 0.0005
DOLLAR_POSITION_LIMIT = 10000

# Evaluation parameters
DEFAULT_TEST_DAYS = 750
PENALTY_FACTOR = 0.1

# Output settings
OUTPUT_DIR = "./output"