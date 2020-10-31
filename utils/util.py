import os, sys
import logging
import time
import torch

def set_logger(args, exp_path):
    try:
		# FIXME(https://github.com/abseil/abseil-py/issues/99)
		# FIXME(https://github.com/abseil/abseil-py/issues/102)
		# Unfortunately, many libraries that include absl (including Tensorflow)
		# will get bitten by double-logging due to absl's incorrect use of
		# the python logging library:
		#   2019-07-19 23:47:38,829 my_logger   779 : test
		#   I0719 23:47:38.829330 139904865122112 foo.py:63] test
		#   2019-07-19 23:47:38,829 my_logger   779 : test
		#   I0719 23:47:38.829469 139904865122112 foo.py:63] test
		# The code below fixes this double-logging.  FMI see:
		#   https://github.com/tensorflow/tensorflow/issues/26691#issuecomment-500369493
        import absl.logging
        logging.root.removeHandler(absl.logging._absl_handler)
        absl.logging._warn_preinit_stderr = False
    except Exception as e:
        print("Failed to fix absl logging bug", e)
        pass

    logFormatter = logging.Formatter('%(message)s') #('%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.DEBUG)
    if args.testing:
        fileHandler = logging.FileHandler('%s/log_test.txt' % (exp_path), mode='w')
    else:
        fileHandler = logging.FileHandler('%s/log_train.txt' % (exp_path), mode='w')
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)
    if not args.noStdout:
        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setFormatter(logFormatter)
        logger.addHandler(consoleHandler)
    logger.info(args)
    logger.info("Experiment path: %s" % (exp_path))
    logger.info(time.asctime(time.localtime(time.time())))
    return logger

def setup_device(deviceId, logger):
    if deviceId >= 0:
        torch.cuda.set_device(deviceId)
        device = torch.device("cuda") # is equivalent to torch.device('cuda:X') where X is the result of torch.cuda.current_device()
    else:
        logger.info("CPU is used.")
        device = torch.device("cpu")
    return device
