# -*- coding: utf-8 -*-
import os
import logging


def create_logger(name):
    
    if not os.path.exists("saves/logs/"):
        os.mkdir("saves/logs/")
        
    logger = logging.getLogger(name)
    hdlr = logging.FileHandler("saves/logs/%s.log" % name)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    # logger.setLevel(logging.INFO)
    
    return logger

