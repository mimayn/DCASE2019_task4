import logging
import sys
import config as cfg
import os


def create_logger(logger_name, log_file):
    '''
    Create a logger.
    The same logger object will be active all through out the python
    interpreter process.
    https://docs.python.org/2/howto/logging-cookbook.html
    Use   logger = logging.getLogger(logger_name) to obtain logging all
    through out
    '''
    logger = logging.getLogger(logger_name)
    # Remove the stdout handler
    logger_handlers = logger.handlers[:]
    for handler in logger_handlers:
        if handler.name == 'std_out':
            logger.removeHandler(handler)
    logger.setLevel(logging.DEBUG)
    file_h = logging.FileHandler(log_file)
    file_h.setLevel(logging.DEBUG)
    file_h.set_name('file_handler')
    terminal_h = logging.StreamHandler(sys.stdout)
    terminal_h.setLevel(logging.INFO)
    terminal_h.set_name('stdout')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    tool_formatter = logging.Formatter(' %(levelname)s - %(message)s')
    file_h.setFormatter(formatter)
    terminal_h.setFormatter(tool_formatter)
    logger.addHandler(file_h)
    logger.addHandler(terminal_h)
    return logger


use_weak = cfg.use_weak
use_synthetic = cfg.use_synthetic

if use_weak and use_synthetic:
    add_dir_path = "_synthetic_and_weak"
     
elif use_weak:
    add_dir_path = "_weak_only"
    
elif use_synthetic:
    add_dir_path = "_synthetic_only"    

if cfg.add_perturbations:
    perturb_type = 'random_mask'
    add_dir_path += "_{}_pertubations".format(cfg.n_perturb) 
else:
    perturb_type = None

store_dir = os.path.join("stored_data", "logs")
if not os.path.exists(store_dir):
    os.makedirs(store_dir)
logger_path ="simple_{}".format(cfg.model_name) + add_dir_path+".log"
LOG = create_logger("model-log", os.path.join(store_dir,logger_path))
