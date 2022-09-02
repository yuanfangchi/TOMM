import collections

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from code_20.model.local_trainer import local_train
from code_20.options import read_options
from code_20.data.data_distributor import DataDistributor
import logging
import json
import time
import sys

logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.ERROR)

@tff.federated_computation(tff.type_at_clients(tf.float32))
def get_average_temperature(sensor_readings):
  return tff.federated_mean(sensor_readings)

def prepare_options():
  # read command line options
  options_raw = read_options("test_multi_agent_" + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time())))

  options = {}
  for option in options_raw:
    options[option] = tf.constant(options_raw[option])

  if options['distributed_training']:
    # agent_names = ['agent_1', 'agent_2', 'agent_3', 'agent_4', 'agent_5', 'agent_6', 'agent_7', 'agent_8',
    #               'agent_9','agent_10', 'agent_11', 'agent_12']
    agent_names = ['agent_1', 'agent_2', 'agent_3']
  else:
    agent_names = ['agent_full']

  # Set logging
  logger.setLevel(logging.INFO)
  fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                          '%m/%d/%Y %I:%M:%S %p')
  console = logging.StreamHandler()
  console.setFormatter(fmt)
  logger.addHandler(console)
  logfile = logging.FileHandler(options['log_file_name'].numpy().decode('ascii'), 'w')
  logfile.setFormatter(fmt)
  logger.addHandler(logfile)
  # read the vocab files, it will be used by many classes hence global scope
  logger.info('reading vocab files...')

  relation_vocabs = json.load(open(options['vocab_dir'].numpy().decode('ascii') + '/relation_vocab.json'))
  entity_vocabs = json.load(open(options['vocab_dir'].numpy().decode('ascii') + '/entity_vocab.json'))

  options['entity_vocab'] = {}
  for entity_vocab in entity_vocabs:
    options['entity_vocab'][entity_vocab] = tf.constant(entity_vocabs[entity_vocab])

  options['relation_vocab'] = {}
  for relation_vocab in relation_vocabs:
    options['relation_vocab'][relation_vocab] = tf.constant(relation_vocabs[relation_vocab])


  logger.info('Reading mid to name map')
  mid_to_word = {}
  # with open('/iesl/canvas/rajarshi/data/RL-Path-RNN/FB15k-237/fb15k_names', 'r') as f:
  #     mid_to_word = json.load(f)
  logger.info('Done..')
  logger.info('Total number of entities {}'.format(len(options['entity_vocab'])))
  logger.info('Total number of relations {}'.format(len(options['relation_vocab'])))
  save_path = ''
  config = tf.compat.v1.ConfigProto()
  config.gpu_options.allow_growth = False
  config.log_device_placement = False

  triple_count_array = [100, 200, 300, 400, 500, 600, 700, 800, 900]

  return options, agent_names

def prepare_episode_data(options, agent_names):
  data_splitter = DataDistributor()
  data_splitter.split(options, agent_names)

  return data_splitter

@tff.federated_computation
def federated_train():
  options, agent_names = prepare_options()
  data_splitter = prepare_episode_data(options, agent_names)

  local_train()


federated_train()
