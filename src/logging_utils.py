"""
Logging utilities for the RL game agent.
Provides structured logging for different components.
"""
import logging
from logging.handlers import RotatingFileHandler
from src.config import ENABLE_LOGGING, LOG_FILENAME

# ─── LOGGING SETUP ─────────────────────────────────────────────────────────
# Create a custom logger
logger = logging.getLogger('GameDebug')
logger.setLevel(logging.DEBUG)

# Console handler (always enabled)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

if ENABLE_LOGGING:
    # File handler with rotation
    file_handler = RotatingFileHandler(
        LOG_FILENAME,
        maxBytes=200*1024*1024,  # 200MB per file
        backupCount=10,          # Keep 10 files (2GB total)
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    
    # Create formatter
    file_formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add formatter to handler
    file_handler.setFormatter(file_formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)

# Create separate loggers for different components
round_logger = logging.getLogger('GameDebug.Round')
match_logger = logging.getLogger('GameDebug.Match')
state_logger = logging.getLogger('GameDebug.State')
learner_logger = logging.getLogger('GameDebug.Learner')

# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────
def log_round(message, *args, **kwargs):
    """Log round-related messages"""
    round_logger.info(message, *args, **kwargs)

def log_match(message, *args, **kwargs):
    """Log match-related messages"""
    match_logger.info(message, *args, **kwargs)

def log_state(message, *args, **kwargs):
    """Log state changes"""
    state_logger.info(message, *args, **kwargs)

def log_learner(message, *args, **kwargs):
    """Log learner messages"""
    learner_logger.info(message, *args, **kwargs)

def log_debug(message, *args, **kwargs):
    """Log debug messages (only to file)"""
    logger.debug(message, *args, **kwargs)