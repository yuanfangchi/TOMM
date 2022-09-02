import pandas as pd
import matplotlib.pyplot as plt
import random
import csv
import json
import os

root_dir = '../../'
vocab_dir = root_dir+'datasets/data_preprocessed/WN18RR/vocab/'
dir = root_dir+'datasets/data_preprocessed/WN18RR/'

#os.makedirs(vocab_dir)
entity_dict = {}

entity_vocab = {}
relation_vocab = {}

entity_vocab['PAD'] = len(entity_vocab)
entity_vocab['UNK'] = len(entity_vocab)
relation_vocab['PAD'] = len(relation_vocab)
relation_vocab['DUMMY_START_RELATION'] = len(relation_vocab)
relation_vocab['NO_OP'] = len(relation_vocab)
relation_vocab['UNK'] = len(relation_vocab)

graph = []
train_candidate = []
train = []
test = []

agent_relation_vocab = {}
agent_entity_vocab = {}

agent_entity_vocab_total = []

with open(dir + 'graph_agent_1.txt') as kg_name:
    triple_file = list(csv.reader(kg_name, delimiter='\t'))
    agent_entity_vocab['graph_agent_1'] = []
    agent_relation_vocab['graph_agent_1'] = []
    for triple in triple_file:
        if not triple[0] in agent_entity_vocab['graph_agent_1']:
            agent_entity_vocab['graph_agent_1'].append(triple[0])
        if not triple[2] in agent_entity_vocab['graph_agent_1']:
            agent_entity_vocab['graph_agent_1'].append(triple[2])

        if not triple[1] in agent_relation_vocab['graph_agent_1']:
            agent_relation_vocab['graph_agent_1'].append(triple[1])

        if not triple[0] in agent_entity_vocab_total:
            agent_entity_vocab_total.append(triple[0])
        if not triple[2] in agent_entity_vocab_total:
            agent_entity_vocab_total.append(triple[2])

    print(len( agent_entity_vocab['graph_agent_1']))
    print(len(agent_relation_vocab['graph_agent_1']))

with open(dir + 'graph_agent_2.txt') as kg_name:
    triple_file = list(csv.reader(kg_name, delimiter='\t'))
    agent_entity_vocab['graph_agent_2'] = []
    agent_relation_vocab['graph_agent_2'] = []
    for triple in triple_file:
        if not triple[0] in agent_entity_vocab['graph_agent_2']:
            agent_entity_vocab['graph_agent_2'].append(triple[0])
        if not triple[2] in agent_entity_vocab['graph_agent_2']:
            agent_entity_vocab['graph_agent_2'].append(triple[2])

        if not triple[1] in agent_relation_vocab['graph_agent_2']:
            agent_relation_vocab['graph_agent_2'].append(triple[1])

        if not triple[0] in agent_entity_vocab_total:
            agent_entity_vocab_total.append(triple[0])
        if not triple[2] in agent_entity_vocab_total:
            agent_entity_vocab_total.append(triple[2])

    print(len( agent_entity_vocab['graph_agent_2']))
    print(len(agent_relation_vocab['graph_agent_2']))

with open(dir + 'graph_agent_3.txt') as kg_name:
    triple_file = list(csv.reader(kg_name, delimiter='\t'))
    agent_entity_vocab['graph_agent_3'] = []
    agent_relation_vocab['graph_agent_3'] = []
    for triple in triple_file:
        if not triple[0] in agent_entity_vocab['graph_agent_3']:
            agent_entity_vocab['graph_agent_3'].append(triple[0])
        if not triple[2] in agent_entity_vocab['graph_agent_3']:
            agent_entity_vocab['graph_agent_3'].append(triple[2])

        if not triple[1] in agent_relation_vocab['graph_agent_3']:
            agent_relation_vocab['graph_agent_3'].append(triple[1])

        if not triple[0] in agent_entity_vocab_total:
            agent_entity_vocab_total.append(triple[0])
        if not triple[2] in agent_entity_vocab_total:
            agent_entity_vocab_total.append(triple[2])

    print(len( agent_entity_vocab['graph_agent_3']))
    print(len(agent_relation_vocab['graph_agent_3']))

unique_entity_3_not_2 = []
unique_entity_2_not_3 = []

for agent_entity in agent_entity_vocab['graph_agent_3']:
    if agent_entity in agent_entity_vocab['graph_agent_1']:
        if agent_entity not in agent_entity_vocab['graph_agent_2']:
            unique_entity_3_not_2.append(agent_entity)

    if agent_entity in agent_entity_vocab['graph_agent_2']:
        if agent_entity not in agent_entity_vocab['graph_agent_1']:
            unique_entity_2_not_3.append(agent_entity)

print(len(unique_entity_3_not_2))
print(len(unique_entity_2_not_3))

print(len(agent_entity_vocab_total))

