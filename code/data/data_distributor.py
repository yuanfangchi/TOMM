from collections import defaultdict
import logging
import numpy as np
import csv
import json
import os
import random

logger = logging.getLogger(__name__)


class DataDistributor:
    def __init__(self, triple_per_agent_limit=None):
        self.triple_per_agent = {}
        self.agent_entity_vocab = {}
        self.agent_relation_vocab = {}
        self.triple_per_agent_limit = triple_per_agent_limit

    def split(self, params, agent_names):
        if params['split_random']:
            self.split_grapher_triple_random(params, agent_names)
        #elif params['abs_relationship']:
        #    self.split_grapher_triple_abs_relation(params, agent_names)
        else:
            self.split_grapher_triple(params, agent_names)
        self.split_batcher_triple_from_graph(params, agent_names)
        #self.split_batcher_aa_triple(params, agent_names)

        # self.create_vocab(params, agent_names)

    def set_triple_per_agent_limit(self, triple_per_agent_limit):
        self.triple_per_agent_limit = triple_per_agent_limit

    def get_grapher_triple_per_count(self):
        return self.triple_per_agent

    def get_grapher_entity_per_count(self):
        return self.agent_entity_vocab

    def get_grapher_relation_per_count(self):
        return self.agent_relation_vocab

    def split_grapher_triple(self, params, agent_names):
        with open(params['data_input_dir'] + '/' + 'graph.txt') as triple_file_raw:
            triple_file = list(csv.reader(triple_file_raw, delimiter='\t'))

            triple_count_start_idx = 0
            if self.triple_per_agent_limit:
                triple_count_per_agent = self.triple_per_agent_limit
            else:
                triple_count_per_agent = int(len(triple_file) / len(agent_names))
            self.triple_per_agent = {}
            self.agent_entity_vocab = {}
            self.agent_relation_vocab = {}

            for agent in agent_names:
                self.triple_per_agent[agent] = triple_count_per_agent
                self.agent_entity_vocab[agent] = []
                self.agent_relation_vocab[agent] = []
                with open(params['data_input_dir'] + '/' + 'graph_' + agent + '.txt', 'w') as triple_file_name:
                    writer = csv.writer(triple_file_name, delimiter='\t')
                    for i in range(triple_count_start_idx, triple_count_start_idx + triple_count_per_agent):

                        if not triple_file[i][0] in self.agent_entity_vocab[agent]:
                            self.agent_entity_vocab[agent].append(triple_file[i][0])
                        if not triple_file[i][1] in self.agent_relation_vocab[agent]:
                            self.agent_relation_vocab[agent].append(triple_file[i][1])
                        if not triple_file[i][2] in self.agent_entity_vocab[agent]:
                            self.agent_entity_vocab[agent].append(triple_file[i][2])

                        writer.writerow(triple_file[i])
                    triple_count_start_idx = triple_count_start_idx + triple_count_per_agent

    def split_grapher_triple_random(self, params, agent_names):
        with open(params['data_input_dir'] + '/' + 'graph.txt') as triple_file_raw:
            triple_file = list(csv.reader(triple_file_raw, delimiter='\t'))
            total_count = len(triple_file)
            # # 打乱数据集
            # random.shuffle(triple_file)
            # 计算正态分布数据集
            print("total_count", total_count)
            smoothness = 1000  # 平滑度，n个agent生成正态分布不够平滑，一般给1000倍agent数量的点足够，最后按索引截取即可
            # 实测方差为1，再经过下面3步变形处理后起伏较为平缓，不突兀
            print("variance_count", 1)  # 打印方差
            normal_sorted = sorted(np.random.normal(0, 1, len(agent_names)*smoothness))
            target_index_suffix = str(smoothness // 2)  # 每多少个index相间取一个正态值 1000/2=500 '500'
            index_suffix_long = len(target_index_suffix)  # 判断index 如 500 1500 2500时就拿出来
            print('index_suffix_long:', index_suffix_long)
            normal_tmp = [i for i in normal_sorted if str(normal_sorted.index(i))[-index_suffix_long:] == target_index_suffix]

            # normal_tmp
            # |                         --/
            # |                     --/
            # |                --/
            # 0----------------------
            # |         --/
            # |    --/
            # |--/

            print('normal_tmp:', normal_tmp)

            normal_tmp = [abs(i) for i in normal_tmp]  # 折叠正态分布

            # normal_tmp
            # |--\                     --/
            # |    --\             --/
            # |         --\   --/
            # 0----------------------

            max_count = int(max(normal_tmp)) + 1  # 找到最大值并进位，
            normal_tmp = [max_count - i for i in normal_tmp]  # 反转正态分布
            print('normal_tmp:', normal_tmp)

            # normal_tmp
            # |          --/   \--
            # |     --/             \--
            # | --/                     \--
            # 0----------------------
            # 计算每个比例
            triple_count_per_agent = [i/sum(normal_tmp) for i in normal_tmp]
            # 计算每个agent总数
            # 计算完了算出起始索引去切片原数据集，保证不重复的分配
            normal_count = [int(i*total_count) for i in triple_count_per_agent]
            print('normal_count:', normal_count)
            # 170000 大约 8000 19000 26000 30000 30000 26000 19000 8000
            # triple_count_per_agent = [1/len(agent_names) for i in agent_names]
            # # 计算每个agent总数
            # # 计算完了算出起始索引去切片原数据集，保证不重复的分配
            # normal_count = [int(i*total_count) for i in triple_count_per_agent]
            # print('normal_count:', normal_count)
            # 170000 大约 20000 20000 20000 20000 20000 20000 20000 20000




            self.triple_per_agent = {}
            self.agent_entity_vocab = {}
            self.agent_relation_vocab = {}
            # agent_triple_spilt_param = {}
            # agent_triple_spilt_param[agent_names[0]] = 0.10
            # agent_triple_spilt_param[agent_names[1]] = 0.20
            # agent_triple_spilt_param[agent_names[2]] = 0.30
            start_idx = 0
            # offset = 0
            for idx, agent in enumerate(agent_names):
                print(agent)
                # triple_count_per_agent = int(len(triple_file) * random.random())
                # 每个agent数据占比
                self.triple_per_agent[agent] = "{}*{}={}".format(total_count, triple_count_per_agent[idx], normal_count[idx])
                # 每个agent数据量
                self.agent_entity_vocab[agent] = []
                self.agent_relation_vocab[agent] = []

                # 正态分布标准差0
                with open(params['data_input_dir'] + '/' + 'graph_' + agent + '.txt', 'w') as triple_file_name:
                    writer = csv.writer(triple_file_name, delimiter='\t')
                    # start_idx = 0
                    offset = normal_count[idx]
                    end_idx = start_idx + offset
                    if idx == len(agent_names):
                        # 防止normal_count中的 比例*总数损失了个别int，最后一位agent的情况取到完为止
                        end_idx = total_count
                    for target_line in triple_file[start_idx:end_idx]:
                        writer.writerow(target_line)
                    # 不断根据当前数量平移切片的索引
                    start_idx += offset


                    # for i in range(triple_count_per_agent):
                    #     idx = random.randint(1, len(triple_file) - 1)

                        # if not triple_file[idx][0] in self.agent_entity_vocab[agent]:
                        #     self.agent_entity_vocab[agent].append(triple_file[idx][0])
                        # if not triple_file[idx][1] in self.agent_relation_vocab[agent]:
                        #     self.agent_relation_vocab[agent].append(triple_file[idx][1])
                        # if not triple_file[idx][2] in self.agent_entity_vocab[agent]:
                        #     self.agent_entity_vocab[agent].append(triple_file[idx][2])

                        # writer.writerow(triple_file[idx])
                        # triple_file.remove(triple_file[idx])


    def split_grapher_triple_abs_relation(self, params, agent_names):

        self.triple_per_agent = {}
        self.agent_entity_vocab = {}
        self.agent_relation_vocab = {}

        # split master regions
        params['relation_vocab'] = json.load(open(params['vocab_dir'] + '/relation_vocab.json'))
        relation_voc_keys = list(params['relation_vocab'].keys())
        num_agent = len(agent_names)
        relations_master_in = [[] for ele in range(num_agent)]
        for i in range(len(relation_voc_keys)):
            ele = relation_voc_keys[i]
            if ele not in ["NO_OP", "DUMMY_START_RELATION", "PAD", "UNK"]:
                agent_index = i % num_agent
                relations_master_in[agent_index].append(ele)
        # build look up to spped up
        belongs_to = {}
        for agent_idx in range(num_agent):
            for rela in relations_master_in[agent_idx]:
                belongs_to[rela] = agent_idx
        # feed graph
        sub_graphs = [[] for ele in range(num_agent)]
        with open(params['data_input_dir'] + '/' + 'graph.txt') as triple_file_raw:
            triple_file = list(csv.reader(triple_file_raw, delimiter='\t'))
            for ele in triple_file:
                idx = belongs_to[ele[1]]
                sub_graphs[idx].append(ele)
            # write into files
            for agent_idx in range(num_agent):
                agent = agent_names[agent_idx]

                this_sub_graph = sub_graphs[agent_idx]

                self.triple_per_agent[agent] = len(this_sub_graph)
                self.agent_entity_vocab[agent] = []
                self.agent_relation_vocab[agent] = []

                with open(params['data_input_dir'] + '/' + 'graph_' + agent + '.txt', 'w') as triple_file_name:
                    writer = csv.writer(triple_file_name, delimiter='\t')
                    for i in range(len(this_sub_graph)):

                        if not this_sub_graph[i][0] in self.agent_entity_vocab[agent]:
                            self.agent_entity_vocab[agent].append(this_sub_graph[i][0])
                        if not this_sub_graph[i][1] in self.agent_relation_vocab[agent]:
                            self.agent_relation_vocab[agent].append(this_sub_graph[i][1])
                        if not this_sub_graph[i][2] in self.agent_entity_vocab[agent]:
                            self.agent_entity_vocab[agent].append(this_sub_graph[i][2])

                        writer.writerow(this_sub_graph[i])

    def split_batcher_aa_triple(self, params, agent_names):
            # triple_count_per_agent = int(len(triple_file) / len(agent_names))
            for agent in agent_names:
                with open(params['data_input_dir'] + '/' + 'graph_' + agent + '.txt') as triple_file_raw:
                    triple_file = list(csv.reader(triple_file_raw, delimiter='\t'))
                    triple_count_per_agent = int(len(triple_file) * 0.3)
                    triple_count_per_agent_dev = int(len(triple_file)*0.7)
                    with open(params['data_input_dir'] + '/' + 'train_' + agent + '.txt', 'w') as triple_file_name:
                        writer = csv.writer(triple_file_name, delimiter='\t')
                        for i in range(0, triple_count_per_agent):
                            writer.writerow(triple_file[i])
                    with open(params['data_input_dir'] + '/' + 'dev_' + agent + '.txt', 'w') as triple_file_name:
                        writer = csv.writer(triple_file_name, delimiter='\t')
                        for i in range(triple_count_per_agent, len(triple_file)):
                            writer.writerow(triple_file[i])


    def split_batcher_triple(self, params, agent_names):
        with open(params['data_input_dir'] + '/' + 'train.txt') as triple_file_raw:
            triple_file = list(csv.reader(triple_file_raw, delimiter='\t'))

            triple_count_start_idx = 0
            triple_count_per_agent = int(len(triple_file) / len(agent_names))
            #triple_count_per_agent = len(triple_file)

            for agent in agent_names:
                with open(params['data_input_dir'] + '/' + 'train_' + agent + '.txt', 'w') as triple_file_name:
                    writer = csv.writer(triple_file_name, delimiter='\t')
                    for i in range(triple_count_start_idx, triple_count_start_idx + triple_count_per_agent):
                        writer.writerow(triple_file[i])
                    triple_count_start_idx = triple_count_start_idx + triple_count_per_agent

        with open(params['data_input_dir'] + '/' + 'dev.txt') as triple_file_raw:
            dev_triple_file = list(csv.reader(triple_file_raw, delimiter='\t'))

            dev_triple_count_start_idx = 0
            dev_triple_count_per_agent = int(len(dev_triple_file) / len(agent_names))
            for agent in agent_names:
                with open(params['data_input_dir'] + '/' + 'dev_' + agent + '.txt', 'w') as triple_file_name:
                    writer = csv.writer(triple_file_name, delimiter='\t')
                    for i in range(dev_triple_count_start_idx, dev_triple_count_start_idx + dev_triple_count_per_agent):
                        writer.writerow(dev_triple_file[i])
                    dev_triple_count_start_idx = dev_triple_count_start_idx + dev_triple_count_per_agent

    def split_batcher_triple_from_graph(self, params, agent_names):
        for agent in agent_names:
            with open(params['data_input_dir'] + '/' + 'graph_' + agent + '.txt') as triple_file_raw:
                triple_file = list(csv.reader(triple_file_raw, delimiter='\t'))
                train_data_len = int(len(triple_file) * 0.5)
                batcher_idx = np.random.randint(len(triple_file), size=(1, train_data_len))
                with open(params['data_input_dir'] + '/' + 'train_' + agent + '.txt', 'w') as triple_file_name:
                    writer = csv.writer(triple_file_name, delimiter='\t')
                    for i in range(0, len(batcher_idx[0])):
                        writer.writerow(triple_file[batcher_idx[0][i]])

        with open(params['data_input_dir'] + '/' + 'dev.txt') as triple_file_raw:
            dev_triple_file = list(csv.reader(triple_file_raw, delimiter='\t'))

            dev_triple_count_start_idx = 0
            dev_triple_count_per_agent = int(len(dev_triple_file) / len(agent_names))
            for agent in agent_names:
                with open(params['data_input_dir'] + '/' + 'dev_' + agent + '.txt', 'w') as triple_file_name:
                    writer = csv.writer(triple_file_name, delimiter='\t')
                    for i in range(dev_triple_count_start_idx, dev_triple_count_start_idx + dev_triple_count_per_agent):
                        writer.writerow(dev_triple_file[i])
                    dev_triple_count_start_idx = dev_triple_count_start_idx + dev_triple_count_per_agent

    def create_vocab(self, params, agent_names):
        params['relation_vocab'] = json.load(open(params['vocab_dir'] + '/relation_vocab.json'))
        params['entity_vocab'] = json.load(open(params['vocab_dir'] + '/entity_vocab.json'))

        print('Total Entity')
        print(len(params['entity_vocab']))
        print('Total Relation')
        print(len(params['relation_vocab']))

        for agent in agent_names:
            with open(params['data_input_dir'] + '/' + 'graph_' + agent + '.txt') as triple_file_raw:
                triple_file = csv.reader(triple_file_raw, delimiter='\t')

                agent_entity_vocab = {}
                agent_relation_vocab = {}
                for line in triple_file:
                    if not line[0] in agent_entity_vocab.keys():
                        agent_entity_vocab[line[0]] = params['entity_vocab'][line[0]]
                    if not line[1] in agent_relation_vocab.keys():
                        agent_relation_vocab[line[1]] = params['relation_vocab'][line[1]]
                    if not line[2] in agent_entity_vocab.keys():
                        agent_entity_vocab[line[2]] = params['entity_vocab'][line[2]]

                print(agent + ' Entity')
                print(len(agent_entity_vocab))
                print(agent + ' Relation')
                print(len(agent_relation_vocab))

                if not os.path.exists(params['vocab_dir'] + '/' + agent):
                    os.makedirs(params['vocab_dir'] + '/' + agent)

                with open(params['vocab_dir'] + '/' + agent + '/entity_vocab.json', 'w') as jsonWriter:
                    json.dump(agent_entity_vocab, jsonWriter)
                with open(params['vocab_dir'] + '/' + agent + '/relation_vocab.json', 'w') as jsonWriter:
                    json.dump(agent_relation_vocab, jsonWriter)