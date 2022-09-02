import pandas as pd
import matplotlib.pyplot as plt
import random
import csv
import json
import os

root_dir = '../../'
vocab_dir = root_dir+'datasets/data_preprocessed/tep/vocab/'
dir = root_dir+'datasets/data_preprocessed/tep/'

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

iot_data = {}
fault_mode = {}
fault_reason = {}
fault_effect = {}
fault_action = {}
fault_test = {}

with open('/Users/YuanfangChi/Library/Application Support/Neo4j Desktop/Application/relate-data/dbmss/dbms-27be5539-8878-49fb-8d70-4f1ca1ed3d76/import/' + 'tep_kg.csv') as tep_kg_raw:
    tep_kg_file = list(csv.reader(tep_kg_raw, delimiter='\t'))

    for i in range(1, len(tep_kg_file)):
        tep_kg_line = tep_kg_file[i][0].split(",")
        if tep_kg_line[0]:
            entity_dict[tep_kg_line[0]] = tep_kg_line[2].replace('"','')
            entity_vocab[tep_kg_line[2].replace('"','')] = len(entity_vocab)
            if "INPUT" in tep_kg_line[1]:
                iot_data[int(tep_kg_line[0])] = tep_kg_line[2].replace('"','')
            elif "OUTPUT" in tep_kg_line[1]:
                iot_data[int(tep_kg_line[0])] = tep_kg_line[2].replace('"', '')
            elif "FAULT" in tep_kg_line[1]:
                fault_mode[int(tep_kg_line[0])] = tep_kg_line[2].replace('"', '')
        else:
            e1 = entity_dict[tep_kg_line[3].replace('"','')]
            e2 = entity_dict[tep_kg_line[4].replace('"','')]
            r = tep_kg_line[5].replace('"','')

            if r not in relation_vocab.keys():
                relation_vocab[r] = len(relation_vocab)

            graph_line = []
            graph_line.append(e1)
            graph_line.append(r)
            graph_line.append(e2)

            print(graph_line)
            graph.append(graph_line)

            e2 = entity_dict[tep_kg_line[3].replace('"', '')]
            e1 = entity_dict[tep_kg_line[4].replace('"', '')]
            r = "_" + tep_kg_line[5].replace('"', '')

            if r not in relation_vocab.keys():
                relation_vocab[r] = len(relation_vocab)

            graph_line = []
            graph_line.append(e1)
            graph_line.append(r)
            graph_line.append(e2)

            print(graph_line)
            graph.append(graph_line)

# fault_mode + iot_data
fault_ratio = 0.2
for i in fault_mode.keys():
    fault_number = int(len(iot_data) * fault_ratio)
    exist_relation = []
    for j in range(0, fault_number):
        idx = random.choice(list(iot_data.keys()))

        while True:
            if iot_data[idx] not in exist_relation:
                exist_relation.append(iot_data[idx])
                break
            idx = random.choice(list(iot_data.keys()))

        e1 = iot_data[idx]
        e2 = fault_mode[i]
        r = "RELATED_TO"

        if r not in relation_vocab.keys():
            relation_vocab[r] = len(relation_vocab)


        graph_line = []
        graph_line.append(e1)
        graph_line.append(r)
        graph_line.append(e2)

        print(graph_line)
        graph.append(graph_line)

        e2 = iot_data[idx]
        e1 = fault_mode[i]
        r = "_RELATED_TO"

        if r not in relation_vocab.keys():
            relation_vocab[r] = len(relation_vocab)

        graph_line = []
        graph_line.append(e1)
        graph_line.append(r)
        graph_line.append(e2)

        print(graph_line)
        graph.append(graph_line)

# fault_reason + fault_mode
fault_reason_number = 10
for i in range(1, fault_reason_number + 1):
    e2 = "FaultReason_" + str(i)
    entity_vocab[e2] = len(entity_vocab)

    fault_reason[i] = e2
    fault_number = int(len(fault_mode) * fault_ratio)
    exist_fault = []
    for j in range(0, fault_number):
        idx = random.choice(list(fault_mode.keys()))

        while True:
            if fault_mode[idx] not in exist_fault:
                exist_fault.append(fault_mode[idx])
                break
            idx = random.choice(list(fault_mode.keys()))

        e1 = fault_mode[idx]
        r = "HAS_REASON"
        if r not in relation_vocab.keys():
            relation_vocab[r] = len(relation_vocab)

        graph_line = []
        graph_line.append(e1)
        graph_line.append(r)
        graph_line.append(e2)

        print(graph_line)
        graph.append(graph_line)

        r = "_HAS_REASON"
        if r not in relation_vocab.keys():
            relation_vocab[r] = len(relation_vocab)

        graph_line = []
        graph_line.append(e2)
        graph_line.append(r)
        graph_line.append(e1)

        print(graph_line)
        graph.append(graph_line)

# fault_action + fault_reason
fault_action_number = 10
for i in range(1, fault_action_number + 1):
    e2 = "FaultAction_" + str(i)
    entity_vocab[e2] = len(entity_vocab)

    fault_action[i] = e2
    fault_reason_number = int(len(fault_reason) * fault_ratio)
    exist_fault = []
    for j in range(0, fault_reason_number):
        idx = random.choice(list(fault_reason.keys()))

        while True:
            if fault_reason[idx] not in exist_fault:
                exist_fault.append(fault_reason[idx])
                break
            idx = random.choice(list(fault_reason.keys()))

        e1 = fault_reason[idx]
        r = "HAS_ACTION"
        if r not in relation_vocab.keys():
            relation_vocab[r] = len(relation_vocab)

        graph_line = []
        graph_line.append(e1)
        graph_line.append(r)
        graph_line.append(e2)

        print(graph_line)
        graph.append(graph_line)

        r = "_HAS_ACTION"
        if r not in relation_vocab.keys():
            relation_vocab[r] = len(relation_vocab)

        graph_line = []
        graph_line.append(e2)
        graph_line.append(r)
        graph_line.append(e1)

        print(graph_line)
        graph.append(graph_line)

# fault_effect + fault_mode
fault_effect_number = 5
for i in range(1, fault_effect_number + 1):
    e2 = "FaultEffect_" + str(i)
    entity_vocab[e2] = len(entity_vocab)

    fault_effect[i] = e2
    fault_number = int(len(fault_mode) * fault_ratio)
    exist_fault = []
    for j in range(0, fault_number):
        idx = random.choice(list(fault_mode.keys()))

        while True:
            if fault_mode[idx] not in exist_fault:
                exist_fault.append(fault_mode[idx])
                break
            idx = random.choice(list(fault_mode.keys()))

        e1 = fault_mode[idx]
        r = "HAS_EFFECT"
        if r not in relation_vocab.keys():
            relation_vocab[r] = len(relation_vocab)

        graph_line = []
        graph_line.append(e1)
        graph_line.append(r)
        graph_line.append(e2)

        print(graph_line)
        graph.append(graph_line)

        r = "_HAS_EFFECT"
        if r not in relation_vocab.keys():
            relation_vocab[r] = len(relation_vocab)

        graph_line = []
        graph_line.append(e2)
        graph_line.append(r)
        graph_line.append(e1)

        print(graph_line)
        graph.append(graph_line)

# # TEP + fault_effect
# for i in fault_effect.keys():
#     e1 = "TEP"
#     e2 = fault_effect[i]
#     r = "HAS_PRODUCT_EFFECT"
#     if r not in relation_vocab.keys():
#         relation_vocab[r] = len(relation_vocab)
#
#     graph_line = []
#     graph_line.append(e1)
#     graph_line.append(r)
#     graph_line.append(e2)
#
#     print(graph_line)
#     graph.append(graph_line)
#
#     r = "_HAS_PRODUCT_EFFECT"
#     if r not in relation_vocab.keys():
#         relation_vocab[r] = len(relation_vocab)
#
#     graph_line = []
#     graph_line.append(e2)
#     graph_line.append(r)
#     graph_line.append(e1)
#
#     print(graph_line)
#     graph.append(graph_line)

# iot_data + fault_action
fault_ratio = 0.2
test_ratio = 0.1
for i in fault_action.keys():
    fault_action_number = int(len(iot_data) * fault_ratio)
    exist_relation = []
    for j in range(0, fault_action_number):
        idx = random.choice(list(iot_data.keys()))

        while True:
            if iot_data[idx] not in exist_relation:
                exist_relation.append(iot_data[idx])
                break
            idx = random.choice(list(iot_data.keys()))

        e1 = iot_data[idx]
        e2 = fault_action[i]
        r = "ACTED_WITH"
        if r not in relation_vocab.keys():
            relation_vocab[r] = len(relation_vocab)

        graph_line = []
        graph_line.append(e1)
        graph_line.append(r)
        graph_line.append(e2)

        print(graph_line)
        graph.append(graph_line)
        train_candidate.append(graph_line)

        r = "_ACTED_WITH"
        if r not in relation_vocab.keys():
            relation_vocab[r] = len(relation_vocab)

        graph_line = []
        graph_line.append(e2)
        graph_line.append(r)
        graph_line.append(e1)

        print(graph_line)
        graph.append(graph_line)

    test_action_number = int(len(iot_data) * test_ratio)
    for t in range(0, test_action_number):
        idx = random.choice(list(iot_data.keys()))

        while True:
            if iot_data[idx] not in exist_relation:
                exist_relation.append(iot_data[idx])
                break
            idx = random.choice(list(iot_data.keys()))

        e1 = iot_data[idx]
        e2 = fault_action[i]
        r = "ACTED_WITH"

        test_line = []
        test_line.append(e1)
        test_line.append(r)
        test_line.append(e2)

        print(test_line)
        test.append(test_line)

# iot_data + fault_test
fault_test_number = 5
for i in range(1, fault_test_number + 1):
    e2 = "FaultTest_" + str(i)
    entity_vocab[e2] = len(entity_vocab)

    fault_test[i] = e2
    fault_iotData_number = int(len(iot_data) * fault_ratio)
    exist_iotData = []
    for j in range(0, fault_iotData_number):
        idx = random.choice(list(iot_data.keys()))

        while True:
            if iot_data[idx] not in exist_iotData:
                exist_iotData.append(iot_data[idx])
                break
            idx = random.choice(list(iot_data.keys()))

        e1 = iot_data[idx]
        r = "HAS_TEST_METHOD"
        if r not in relation_vocab.keys():
            relation_vocab[r] = len(relation_vocab)

        graph_line = []
        graph_line.append(e1)
        graph_line.append(r)
        graph_line.append(e2)

        print(graph_line)
        graph.append(graph_line)

        r = "_HAS_TEST_METHOD"
        if r not in relation_vocab.keys():
            relation_vocab[r] = len(relation_vocab)

        graph_line = []
        graph_line.append(e2)
        graph_line.append(r)
        graph_line.append(e1)

        print(graph_line)
        graph.append(graph_line)

# fault_test + fault_mode
for i in fault_test.keys():

    fault_faultMode_number = int(len(fault_mode) * fault_ratio)
    exist_faultMode = []
    for j in range(0, fault_faultMode_number):
        idx = random.choice(list(fault_mode.keys()))

        while True:
            if fault_mode[idx] not in exist_faultMode:
                exist_faultMode.append(fault_mode[idx])
                break
            idx = random.choice(list(fault_mode.keys()))

        e1 = fault_test[i]
        e2 = fault_mode[idx]
        r = "TEST_WITH"
        if r not in relation_vocab.keys():
            relation_vocab[r] = len(relation_vocab)

        graph_line = []
        graph_line.append(e1)
        graph_line.append(r)
        graph_line.append(e2)

        print(graph_line)
        graph.append(graph_line)

        r = "_TEST_WITH"
        if r not in relation_vocab.keys():
            relation_vocab[r] = len(relation_vocab)

        graph_line = []
        graph_line.append(e2)
        graph_line.append(r)
        graph_line.append(e1)

        print(graph_line)
        graph.append(graph_line)

# TEP + fault_effect
fault_ratio = 0.4
test_ratio = 0.2

fault_effect_number = int(len(fault_effect) * fault_ratio)
exist_relation = []
for j in range(0, fault_effect_number):
    idx = random.choice(list(fault_effect.keys()))

    while True:
        if fault_effect[idx] not in exist_relation:
            exist_relation.append(fault_effect[idx])
            break
        idx = random.choice(list(fault_effect.keys()))

    e1 = "TEP"
    e2 = fault_effect[idx]
    r = "PRODUCT_EFFECT"
    if r not in relation_vocab.keys():
        relation_vocab[r] = len(relation_vocab)

    graph_line = []
    graph_line.append(e1)
    graph_line.append(r)
    graph_line.append(e2)

    print(graph_line)
    graph.append(graph_line)
    train_candidate.append(graph_line)

    r = "_PRODUCT_EFFECT"
    if r not in relation_vocab.keys():
        relation_vocab[r] = len(relation_vocab)

    graph_line = []
    graph_line.append(e2)
    graph_line.append(r)
    graph_line.append(e1)

    print(graph_line)
    graph.append(graph_line)

test_effect_number = int(len(fault_effect) * test_ratio)
for t in range(0, test_effect_number):
    idx = random.choice(list(fault_effect.keys()))

    while True:
        if fault_effect[idx] not in exist_relation:
            exist_relation.append(fault_effect[idx])
            break
        idx = random.choice(list(fault_effect.keys()))

    e1 = "TEP"
    e2 = fault_effect[idx]
    r = "PRODUCT_EFFECT"

    test_line = []
    test_line.append(e1)
    test_line.append(r)
    test_line.append(e2)

    print(test_line)
    test.append(test_line)

# fault_test + fault_action
fault_ratio = 0.2
test_ratio = 0.1

for i in fault_test.keys():
    fault_action_number = int(len(fault_action) * fault_ratio)
    exist_relation = []
    for j in range(0, fault_action_number):
        idx = random.choice(list(fault_action.keys()))

        while True:
            if fault_action[idx] not in exist_relation:
                exist_relation.append(fault_action[idx])
                break
            idx = random.choice(list(fault_action.keys()))

        e1 = fault_test[i]
        e2 = fault_action[idx]
        r = "TEST_ACTION"
        if r not in relation_vocab.keys():
            relation_vocab[r] = len(relation_vocab)

        graph_line = []
        graph_line.append(e1)
        graph_line.append(r)
        graph_line.append(e2)

        print(graph_line)
        graph.append(graph_line)
        train_candidate.append(graph_line)

        r = "_TEST_ACTION"
        if r not in relation_vocab.keys():
            relation_vocab[r] = len(relation_vocab)

        graph_line = []
        graph_line.append(e2)
        graph_line.append(r)
        graph_line.append(e1)

        print(graph_line)
        graph.append(graph_line)

    test_action_number = int(len(fault_action) * test_ratio)
    for t in range(0, test_action_number):
        idx = random.choice(list(fault_action.keys()))

        while True:
            if fault_action[idx] not in exist_relation:
                exist_relation.append(fault_action[idx])
                break
            idx = random.choice(list(fault_action.keys()))

        e1 = fault_test[i]
        e2 = fault_action[idx]
        r = "TEST_ACTION"

        test_line = []
        test_line.append(e1)
        test_line.append(r)
        test_line.append(e2)

        print(test_line)
        test.append(test_line)


train_ratio = 0.8
train_number = int(len(train_candidate) * train_ratio)
exist_train = []
for i in range(0, train_number):
    train_line = random.choice(train_candidate)

    print(train_line)
    train.append(train_line)
    train_candidate.remove(train_line)

with open(dir + 'graph.txt', 'w') as tep_kg_name:
    writer = csv.writer(tep_kg_name, delimiter='\t')
    for i in graph:
        writer.writerow(i)

with open(dir + 'train.txt', 'w') as tep_kg_name:
    writer = csv.writer(tep_kg_name, delimiter='\t')
    for i in train:
        writer.writerow(i)

with open(dir + 'dev.txt', 'w') as tep_kg_name:
    writer = csv.writer(tep_kg_name, delimiter='\t')
    for i in test:
        writer.writerow(i)

with open(dir + 'test.txt', 'w') as tep_kg_name:
    writer = csv.writer(tep_kg_name, delimiter='\t')
    for i in test:
        writer.writerow(i)

with open(vocab_dir + 'entity_vocab.json', 'w') as fout:
    json.dump(entity_vocab, fout)

with open(vocab_dir + 'relation_vocab.json', 'w') as fout:
    json.dump(relation_vocab, fout)

