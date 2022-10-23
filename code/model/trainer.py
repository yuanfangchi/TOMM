from __future__ import absolute_import
from __future__ import division

import copy
import random

from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple
from tqdm import tqdm
import json
import time
import os
import logging
import numpy as np
import tensorflow as tf
from code.model.agent import Agent
from code.options import read_options
from code.model.environment import env
import codecs
from collections import defaultdict
import gc
import resource
import sys
from code.model.baseline import ReactiveBaseline
from code.model.nell_eval import nell_eval
from scipy.special import logsumexp as lse
from pprint import pprint
from code.data.data_distributor import DataDistributor
#from code.model.blackboard import Blackboard
import csv

logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.ERROR)

tf.compat.v1.disable_eager_execution()


class Trainer(object):

    def __init__(self, params, agent=None, isTrainHandover=False):

        # transfer parameters to self
        for key, val in params.items(): setattr(self, key, val);

        self.agent = Agent(params)
        self.save_path = None
        self.test_rollouts = None
        self.path_logger_file_ = None
        self.train_environment = env(params, agent, 'train')
        self.dev_test_environment = env(params, agent, 'dev')
        self.test_test_environment = env(params, agent, 'test')
        self.test_environment = self.dev_test_environment
        self.rev_relation_vocab = self.train_environment.grapher.rev_relation_vocab
        self.rev_entity_vocab = self.train_environment.grapher.rev_entity_vocab
        self.max_hits_at_10 = 0
        self.max_num_actions = params['max_num_actions']
        self.ePAD = self.entity_vocab['PAD']
        self.rPAD = self.relation_vocab['PAD']
        # optimize
        self.baseline = ReactiveBaseline(l=self.Lambda)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
        self.use_entity_embeddings = params['use_entity_embeddings']
        if self.use_entity_embeddings:
            self.m = 4
        else:
            self.m = 2

        if agent is not None:
            self.output_dir = self.output_dir + '/' + agent
            if not os.path.isdir(self.output_dir):
                os.mkdir(self.output_dir)
            if isTrainHandover:
                self.output_dir = self.output_dir + '/handover'
        else:
            self.output_dir = self.output_dir + '/test'
        self.model_dir = self.output_dir + '/' + 'model/'
        self.path_logger_file = self.output_dir
        self.log_file_name = self.output_dir + '/log.txt'

        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        if not os.path.isdir(self.model_dir):
            os.mkdir(self.model_dir)



    def calc_reinforce_loss(self):

        loss = tf.stack(self.per_example_loss, axis=1)  # [B, T]

        self.tf_baseline = self.baseline.get_baseline_value()
        # self.pp = tf.Print(self.tf_baseline)
        # multiply with rewards
        final_reward = self.cum_discounted_reward - self.tf_baseline
        # reward_std = tf.sqrt(tf.reduce_mean(tf.square(final_reward))) + 1e-5 # constant addded for numerical stability
        reward_mean, reward_var = tf.nn.moments(x=final_reward, axes=[0, 1])
        # Constant added for numerical stability
        reward_std = tf.sqrt(reward_var) + 1e-6
        final_reward = tf.compat.v1.div(final_reward - reward_mean, reward_std)

        loss = tf.multiply(loss, final_reward)  # [B, T]
        loss_list = tf.unstack(loss, axis=1)
        for i in range(len(loss_list)):
            loss_list[i] = tf.multiply(loss_list[i], tf.dtypes.cast(self.valid_loss_idx, tf.float32))
        loss = tf.stack(loss_list, axis=1)

        self.loss_before_reg = loss

        total_loss = tf.reduce_mean(input_tensor=loss) - self.decaying_beta * self.entropy_reg_loss(self.per_example_logits)  # scalar

        return total_loss

    def entropy_reg_loss(self, all_logits):
        all_logits = tf.stack(all_logits, axis=2)  # [B, MAX_NUM_ACTIONS, T]
        entropy_policy = - tf.reduce_mean(input_tensor=tf.reduce_sum(input_tensor=tf.multiply(tf.exp(all_logits), all_logits), axis=1))  # scalar
        return entropy_policy


    def initialize(self, restore=None, sess=None):

        logger.info("Creating TF graph...")
        self.candidate_relation_sequence = []
        self.candidate_entity_sequence = []
        self.input_path = []
        self.first_state_of_test = tf.compat.v1.placeholder(tf.bool, name="is_first_state_of_test")
        self.query_relation = tf.compat.v1.placeholder(tf.int32, [None], name="query_relation")
        self.range_arr = tf.compat.v1.placeholder(tf.int32, shape=[None, ], name="range_arr")
        self.global_step = tf.Variable(0, trainable=False)
        self.decaying_beta = tf.compat.v1.train.exponential_decay(self.beta, self.global_step,
                                                   200, 0.90, staircase=False)
        self.entity_sequence = []

        # to feed in the discounted reward tensor
        self.cum_discounted_reward = tf.compat.v1.placeholder(tf.float32, [None, self.path_length],
                                                    name="cumulative_discounted_reward")
        self.valid_loss_idx = tf.compat.v1.placeholder(tf.int32, None,
                                                    name="valid_loss_idx")

        for t in range(self.path_length):
            next_possible_relations = tf.compat.v1.placeholder(tf.int32, [None, self.max_num_actions],
                                                   name="next_relations_{}".format(t))
            next_possible_entities = tf.compat.v1.placeholder(tf.int32, [None, self.max_num_actions],
                                                     name="next_entities_{}".format(t))
            input_label_relation = tf.compat.v1.placeholder(tf.int32, [None], name="input_label_relation_{}".format(t))
            start_entities = tf.compat.v1.placeholder(tf.int32, [None, ])
            mem_shape = self.agent.get_mem_shape()
            self.input_path.append(input_label_relation)
            self.candidate_relation_sequence.append(next_possible_relations)
            self.candidate_entity_sequence.append(next_possible_entities)
            self.entity_sequence.append(start_entities)

        self.loss_before_reg = tf.constant(0.0)

        self.per_example_loss, self.per_example_logits, self.action_idx, self.rnn_state, self.rnn_output, self.chosen_relations= self.agent(
            self.candidate_relation_sequence,
            self.candidate_entity_sequence, self.entity_sequence, self.query_relation, self.range_arr)

        self.loss_op = self.calc_reinforce_loss()

        # backprop
        self.train_op = self.bp(self.loss_op)

        # Building the test graph
        self.prev_state = tf.compat.v1.placeholder(tf.float32, self.agent.get_mem_shape(), name="memory_of_agent")
        self.prev_relation = tf.compat.v1.placeholder(tf.int32, [None, ], name="previous_relation")
        self.query_embedding = tf.nn.embedding_lookup(params=self.agent.relation_lookup_table, ids=self.query_relation)  # [B, 2D]
        layer_state = tf.unstack(self.prev_state, self.LSTM_layers)
        formated_state = [tf.unstack(s, 2) for s in layer_state]
        self.next_relations = tf.compat.v1.placeholder(tf.int32, shape=[None, self.max_num_actions])
        self.next_entities = tf.compat.v1.placeholder(tf.int32, shape=[None, self.max_num_actions])

        self.handover_rnn_output = tf.compat.v1.placeholder(tf.float32, shape=[self.batch_size * self.num_rollouts, self.m * self.hidden_size])
        self.handover_rnn_state = tf.compat.v1.placeholder(tf.float32, mem_shape)
        self.run_handover_step = tf.compat.v1.placeholder(tf.bool)

        self.current_entities = tf.compat.v1.placeholder(tf.int32, shape=[None,])

        with tf.compat.v1.variable_scope("global_policy_steps_unroll") as scope:
            scope.reuse_variables()
            self.test_loss, test_state, self.test_logits, test_output, self.test_action_idx, self.chosen_relation = self.agent.step(
                self.next_relations, self.next_entities, formated_state, self.prev_relation, self.query_embedding,
                self.current_entities, self.range_arr)
            self.test_state = tf.stack(test_state)

        logger.info('TF Graph creation done..')
        self.model_saver = tf.compat.v1.train.Saver(max_to_keep=2)

        # return the variable initializer Op.
        if not restore:
            return tf.compat.v1.global_variables_initializer()
        else:
            return  self.model_saver.restore(sess, restore)

    def initialize_pretrained_embeddings(self, sess):
        if self.pretrained_embeddings_action != '':
            embeddings = np.loadtxt(open(self.pretrained_embeddings_action))
            _ = sess.run((self.agent.relation_embedding_init),
                         feed_dict={self.agent.action_embedding_placeholder: embeddings})
        if self.pretrained_embeddings_entity != '':
            embeddings = np.loadtxt(open(self.pretrained_embeddings_entity))
            _ = sess.run((self.agent.entity_embedding_init),
                         feed_dict={self.agent.entity_embedding_placeholder: embeddings})

    def bp(self, cost):
        self.baseline.update(tf.reduce_mean(input_tensor=self.cum_discounted_reward))
        tvars = tf.compat.v1.trainable_variables()
        grads = tf.gradients(ys=cost, xs=tvars)
        grads, _ = tf.clip_by_global_norm(grads, self.grad_clip_norm)
        train_op = self.optimizer.apply_gradients(zip(grads, tvars))
        with tf.control_dependencies([train_op]):  # see https://github.com/tensorflow/tensorflow/issues/1899
            self.dummy = tf.constant(0)
        return train_op

    def calc_cum_discounted_reward(self, rewards):
        """
        calculates the cumulative discounted reward.
        :param rewards:
        :param T:
        :param gamma:
        :return:
        """
        running_add = np.zeros([rewards.shape[0]])  # [B]
        cum_disc_reward = np.zeros([rewards.shape[0], self.path_length])  # [B, T]
        cum_disc_reward[:,
        self.path_length - 1] = rewards  # set the last time step to the reward received at the last state
        for t in reversed(range(self.path_length)):
            running_add = self.gamma * running_add + cum_disc_reward[:, t]
            cum_disc_reward[:, t] = running_add
        return cum_disc_reward

    def gpu_io_setup(self):
        # create fetches for partial_run_setup
        fetches = self.per_example_loss  + self.action_idx + [self.loss_op] + self.per_example_logits + [self.dummy] + self.rnn_state + self.chosen_relations + self.rnn_output
        feeds =  [self.first_state_of_test] + self.candidate_relation_sequence+ self.candidate_entity_sequence + self.input_path + \
                [self.query_relation] + [self.cum_discounted_reward] + [self.range_arr] + self.entity_sequence + \
                 [self.valid_loss_idx]


        feed_dict = [{} for _ in range(self.path_length)]

        feed_dict[0][self.first_state_of_test] = False
        feed_dict[0][self.query_relation] = None
        feed_dict[0][self.range_arr] = np.arange(self.batch_size*self.num_rollouts)
        for i in range(self.path_length):
            feed_dict[i][self.input_path[i]] = np.zeros(self.batch_size * self.num_rollouts)  # placebo
            feed_dict[i][self.candidate_relation_sequence[i]] = None
            feed_dict[i][self.candidate_entity_sequence[i]] = None
            feed_dict[i][self.entity_sequence[i]] = None

        return fetches, feeds, feed_dict

    def update_valid_loss_idx(self, next_entities):
        valid_loss_idx = next_entities[:, 0]
        condlist = [valid_loss_idx > 0]
        choicelist = [valid_loss_idx - valid_loss_idx + 1]
        valid_loss_idx = np.select(condlist, choicelist)

        return valid_loss_idx

    def train(self, sess):
        # import pdb
        # pdb.set_trace()
        fetches, feeds, feed_dict = self.gpu_io_setup()

        train_loss = 0.0
        start_time = time.time()
        self.batch_counter = 0
        batch_loss = {}
        memory_use = {}
        episode_handovers = defaultdict(list)
        for episode in self.train_environment.get_episodes():
            self.batch_counter += 1
            h = sess.partial_run_setup(fetches=fetches, feeds=feeds)
            feed_dict[0][self.query_relation] = episode.get_query_relation()

            # get initial state
            state = episode.get_state()
            valid_loss_idx = self.update_valid_loss_idx(state['next_entities'])

            # for each time step
            loss_before_regularization = []
            logits = []
            handover_idx = None
            for i in range(self.path_length):
                current_entities_at_t = state['current_entities']
                next_relations_at_t = state['next_relations']
                next_entities_at_t = state['next_entities']

                feed_dict[i][self.candidate_relation_sequence[i]] = state['next_relations']
                feed_dict[i][self.candidate_entity_sequence[i]] = state['next_entities']
                feed_dict[i][self.entity_sequence[i]] = state['current_entities']

                per_example_loss, per_example_logits, idx, rnn_state, rnn_output, chosen_relation = sess.partial_run(h, [self.per_example_loss[i],
                                                                                                                         self.per_example_logits[i], self.action_idx[i], self.rnn_state[i], self.rnn_output[i], self.chosen_relations[i]],
                                                  feed_dict=feed_dict[i])

                loss_before_regularization.append(per_example_loss)
                logits.append(per_example_logits)
                # action = np.squeeze(action, axis=1)  # [B,]
                state = episode(idx)

                current_entities_handover = []
                for j in range(len(state['current_entities'])):
                    if current_entities_at_t[j] != 0 and state['current_entities'][j] == 0:
                        current_entities_handover.append(current_entities_at_t[j])
                        handover_idx = i
                    else:
                        current_entities_handover.append(0)

                episode_handover_state = {}
                episode_handover_state['current_entities'] = current_entities_at_t
                episode_handover_state['next_relations'] = next_relations_at_t
                episode_handover_state['next_entities'] = next_entities_at_t
                episode_handover_state['handover_entities'] = current_entities_handover
                episode_handover_state['handover_idx'] = handover_idx
                episode_handover_state['rnn_state'] = rnn_state
                episode_handover_state['rnn_output'] = rnn_output
                episode_handovers[episode].append((i, episode_handover_state))

            loss_before_regularization = np.stack(loss_before_regularization, axis=1)

            # get the final reward from the environment
            rewards = episode.get_reward()

            # computed cumulative discounted reward
            cum_discounted_reward = self.calc_cum_discounted_reward(rewards)  # [B, T]


            # backprop
            batch_total_loss, _ = sess.partial_run(h, [self.loss_op, self.dummy],
                                                   feed_dict={self.cum_discounted_reward: cum_discounted_reward,
                                                              self.valid_loss_idx: valid_loss_idx})

            # print statistics
            train_loss = 0.98 * train_loss + 0.02 * batch_total_loss
            avg_reward = np.mean(rewards)
            # now reshape the reward to [orig_batch_size, num_rollouts], I want to calculate for how many of the
            # entity pair, atleast one of the path get to the right answer
            reward_reshape = np.reshape(rewards, (self.batch_size, self.num_rollouts))  # [orig_batch, num_rollouts]
            reward_reshape = np.sum(reward_reshape, axis=1)  # [orig_batch]
            reward_reshape = (reward_reshape > 0)
            num_ep_correct = np.sum(reward_reshape)

            if np.isnan(train_loss):
                raise ArithmeticError("Error in computing loss")

            logger.info("batch_counter: {0:4d}, num_hits: {1:7.4f}, avg. reward per batch {2:7.4f}, "
                        "num_ep_correct {3:4d}, avg_ep_correct {4:7.4f}, train loss {5:7.4f}".
                        format(self.batch_counter, np.sum(rewards), avg_reward, num_ep_correct,
                               (num_ep_correct / self.batch_size),
                               train_loss))

            if self.batch_counter%self.eval_every == 0:
                batch_loss[self.batch_counter] = train_loss
                memory_use[self.batch_counter] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            #     with open(self.output_dir + '/scores.txt', 'a') as score_file:
            #         score_file.write("Score for iteration " + str(self.batch_counter) + "\n")
            #     os.mkdir(self.path_logger_file + "/" + str(self.batch_counter))
            #     self.path_logger_file_ = self.path_logger_file + "/" + str(self.batch_counter) + "/paths"
            #
            #     self.test(sess, beam=True, print_paths=False)

            logger.info('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            self.save_path = self.model_saver.save(sess, self.model_dir + "model" + '.ckpt')

            with open(self.pretrained_embeddings_action_dir, 'w') as out:
                pprint(self.agent.relation_lookup_table, stream=out)
            self.pretrained_embeddings_action = self.pretrained_embeddings_action_dir

            with open(self.pretrained_embeddings_entity_dir, 'w') as out:
                pprint(self.agent.entity_lookup_table, stream=out)
            self.pretrained_embeddings_entity = self.pretrained_embeddings_entity_dir
            #  ll pre*
            #  <tf.Variable 'action_lookup_table/relation_lookup_table:0' shape=(26, 100) dtype=float32>
            #  <tf.Variable 'entity_lookup_table/entity_lookup_table:0' shape=(40945, 100) dtype=float32>

            gc.collect()
            if self.batch_counter >= self.total_iterations:
                break

        return episode_handovers, batch_loss, memory_use

    def train_noop_episode(self, sess, episode_handovers):
        fetches, feeds, feed_dict = self.gpu_io_setup()

        train_loss = 0.0
        start_time = time.time()

        self.batch_counter = 0
        batch_loss = {}
        memory_use = {}
        for episode_handover in episode_handovers:
            reconstruct_state_map = {}
            for i, episode_handover_state in episode_handovers[episode_handover]:
                if episode_handover_state['handover_idx'] is None or episode_handover_state['handover_idx'] < i:
                    reconstruct_state_map[i] = episode_handover_state
                    pass
                else:
                    # get initial state
                    episode = self.train_environment.get_handover_episodes(episode_handover)

                    self.batch_counter += 1
                    h = sess.partial_run_setup(fetches=fetches, feeds=feeds)
                    feed_dict[0][self.query_relation] = episode_handover.get_query_relation()

                    loss_before_regularization = []
                    logits = []

                    reconstruct_path_idx = 0
                    path_idx_offset = False
                    for path_idx in reconstruct_state_map.keys():
                        feed_dict[path_idx][self.candidate_relation_sequence[path_idx]] = reconstruct_state_map[path_idx]['next_relations']
                        feed_dict[path_idx][self.candidate_entity_sequence[path_idx]] = reconstruct_state_map[path_idx]['next_entities']
                        feed_dict[path_idx][self.entity_sequence[path_idx]] = reconstruct_state_map[path_idx]['current_entities']

                        per_example_loss, per_example_logits, idx, rnn_state, rnn_output, chosen_relation = sess.partial_run(h, [
                            self.per_example_loss[path_idx],
                            self.per_example_logits[path_idx], self.action_idx[path_idx], self.rnn_state[path_idx], self.rnn_output[path_idx],
                            self.chosen_relations[path_idx]], feed_dict=feed_dict[path_idx])
                        path_idx_offset = True
                        reconstruct_path_idx = path_idx
                    reconstruct_state_map[i] = episode_handover_state
                    new_state = episode.return_next_actions(np.array(episode_handover_state['handover_entities']),
                                                            episode_handover_state['handover_idx'])
                    valid_loss_idx = self.update_valid_loss_idx(new_state['next_entities'])

                    if path_idx_offset:
                        reconstruct_path_idx = reconstruct_path_idx + 1

                    for j in range(reconstruct_path_idx, self.path_length):
                        feed_dict[j][self.candidate_relation_sequence[j]] = new_state['next_relations']
                        feed_dict[j][self.candidate_entity_sequence[j]] = new_state['next_entities']
                        feed_dict[j][self.entity_sequence[j]] = new_state['current_entities']

                        per_example_loss, per_example_logits, idx, rnn_state, chosen_relation = sess.partial_run(h, [
                            self.per_example_loss[j], self.per_example_logits[j], self.action_idx[j], self.rnn_state[j],
                            self.chosen_relations[j]],
                                                                                                                 feed_dict=
                                                                                                                 feed_dict[
                                                                                                                     j])
                        new_state = episode(idx)
                    loss_before_regularization.append(per_example_loss)
                    logits.append(per_example_logits)

                    loss_before_regularization = np.stack(loss_before_regularization, axis=1)

                    # get the final reward from the environment
                    rewards = episode.get_reward()

                    # computed cumulative discounted reward
                    cum_discounted_reward = self.calc_cum_discounted_reward(rewards)  # [B, T]

                    # backprop
                    batch_total_loss, _ = sess.partial_run(h, [self.loss_op, self.dummy],
                                                           feed_dict={self.cum_discounted_reward: cum_discounted_reward,
                                                                      self.valid_loss_idx: valid_loss_idx})

                    # print statistics
                    train_loss = 0.98 * train_loss + 0.02 * batch_total_loss

                    avg_reward = np.mean(rewards)
                    # now reshape the reward to [orig_batch_size, num_rollouts], I want to calculate for how many of the
                    # entity pair, atleast one of the path get to the right answer
                    reward_reshape = np.reshape(rewards, (self.batch_size, self.num_rollouts))  # [orig_batch, num_rollouts]
                    reward_reshape = np.sum(reward_reshape, axis=1)  # [orig_batch]
                    reward_reshape = (reward_reshape > 0)
                    num_ep_correct = np.sum(reward_reshape)


                    if np.isnan(train_loss):
                        raise ArithmeticError("Error in computing loss")

                    logger.info("episode handover task, batch_counter: {0:4d}, num_hits: {1:7.4f}, avg. reward per batch {2:7.4f}, "
                                "num_ep_correct {3:4d}, avg_ep_correct {4:7.4f}, train loss {5:7.4f}".
                                format(self.batch_counter, np.sum(rewards), avg_reward, num_ep_correct,
                                       (num_ep_correct / self.batch_size),
                                       train_loss))

                    if self.batch_counter % self.eval_every == 0:
                        batch_loss[self.batch_counter] = train_loss
                        memory_use[self.batch_counter] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

                    self.save_path = self.model_saver.save(sess, self.model_dir + "model" + '.ckpt')
                    logger.info('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

                    gc.collect()
        return batch_loss, memory_use

    def train_full_episode(self, sess, episode_handovers):
        fetches, feeds, feed_dict = self.gpu_io_setup()

        train_loss = 0.0
        start_time = time.time()

        self.batch_counter = 0
        batch_loss = {}
        memory_use = {}
        episode_handovers_on_handover_node = defaultdict(list)
        for episode_handover in episode_handovers:
            reconstruct_state_map = {}
            for i, episode_handover_state in episode_handovers[episode_handover]:
                self.batch_counter += 1

                reconstruct_state_map[i] = episode_handover_state

                h = sess.partial_run_setup(fetches=fetches, feeds=feeds)
                feed_dict[0][self.query_relation] = episode_handover.get_query_relation()

                # get initial state
                episode = self.train_environment.get_handover_episodes(episode_handover)

                loss_before_regularization = []
                logits = []


                episode_handover_state_on_handover_node = {}

                if i > 0:
                    break
                    # for path_idx in range(i):
                    #     current_entities_at_t = reconstruct_state_map[path_idx]['current_entities']
                    #     next_relations_at_t = reconstruct_state_map[path_idx]['next_relations']
                    #     next_entities_at_t = reconstruct_state_map[path_idx]['next_entities']
                    #
                    #     feed_dict[path_idx][self.candidate_relation_sequence[path_idx]] = \
                    #     reconstruct_state_map[path_idx]['next_relations']
                    #     feed_dict[path_idx][self.candidate_entity_sequence[path_idx]] = reconstruct_state_map[path_idx][
                    #         'next_entities']
                    #     feed_dict[path_idx][self.entity_sequence[path_idx]] = reconstruct_state_map[path_idx][
                    #         'current_entities']
                    #
                    #     per_example_loss, per_example_logits, idx, rnn_state, rnn_output, chosen_relation = sess.partial_run(h, [
                    #         self.per_example_loss[path_idx],
                    #         self.per_example_logits[path_idx], self.action_idx[path_idx], self.rnn_state[path_idx], self.rnn_output[path_idx],
                    #         self.chosen_relations[path_idx]], feed_dict=feed_dict[path_idx])
                    #
                    #     episode_handover_state_on_handover_node['current_entities'] = current_entities_at_t
                    #     episode_handover_state_on_handover_node['next_relations'] = next_relations_at_t
                    #     episode_handover_state_on_handover_node['next_entities'] = next_entities_at_t
                    #     episode_handovers_on_handover_node[episode_handover].append((i, episode_handover_state_on_handover_node))

                new_state = episode.return_next_actions(np.array(episode_handover_state['current_entities']), i)
                valid_loss_idx = self.update_valid_loss_idx(new_state['next_entities'])

                for j in range(i, self.path_length):
                    current_entities_at_t = new_state['current_entities']
                    next_relations_at_t = new_state['next_relations']
                    next_entities_at_t = new_state['next_entities']

                    feed_dict[j][self.candidate_relation_sequence[j]] = new_state['next_relations']
                    feed_dict[j][self.candidate_entity_sequence[j]] = new_state['next_entities']
                    feed_dict[j][self.entity_sequence[j]] = new_state['current_entities']

                    per_example_loss, per_example_logits, idx, rnn_state, chosen_relation = sess.partial_run(h, [
                        self.per_example_loss[j], self.per_example_logits[j], self.action_idx[j], self.rnn_state[j],
                        self.chosen_relations[j]],
                                                                                                             feed_dict=
                                                                                                             feed_dict[
                                                                                                                 j])

                    new_state = episode(idx)
                    episode_handover_state_on_handover_node['current_entities'] = current_entities_at_t
                    episode_handover_state_on_handover_node['next_relations'] = next_relations_at_t
                    episode_handover_state_on_handover_node['next_entities'] = next_entities_at_t
                    episode_handovers_on_handover_node[episode_handover].append((i, episode_handover_state_on_handover_node))

                loss_before_regularization.append(per_example_loss)
                logits.append(per_example_logits)


                loss_before_regularization = np.stack(loss_before_regularization, axis=1)
                # get the final reward from the environment
                rewards = episode.get_reward()

                # computed cumulative discounted reward
                cum_discounted_reward = self.calc_cum_discounted_reward(rewards)  # [B, T]

                # backprop
                batch_total_loss, _ = sess.partial_run(h, [self.loss_op, self.dummy],
                                                       feed_dict={self.cum_discounted_reward: cum_discounted_reward,
                                                                  self.valid_loss_idx:valid_loss_idx})

                # print statistics
                train_loss = 0.98 * train_loss + 0.02 * batch_total_loss

                avg_reward = np.mean(rewards)
                # now reshape the reward to [orig_batch_size, num_rollouts], I want to calculate for how many of the
                # entity pair, atleast one of the path get to the right answer
                reward_reshape = np.reshape(rewards, (self.batch_size, self.num_rollouts))  # [orig_batch, num_rollouts]
                reward_reshape = np.sum(reward_reshape, axis=1)  # [orig_batch]
                reward_reshape = (reward_reshape > 0)
                num_ep_correct = np.sum(reward_reshape)


                if np.isnan(train_loss):
                    raise ArithmeticError("Error in computing loss")

                logger.info("episode handover task, batch_counter: {0:4d}, num_hits: {1:7.4f}, avg. reward per batch {2:7.4f}, "
                            "num_ep_correct {3:4d}, avg_ep_correct {4:7.4f}, train loss {5:7.4f}".
                            format(self.batch_counter, np.sum(rewards), avg_reward, num_ep_correct,
                                   (num_ep_correct / self.batch_size),
                                   train_loss))

                if self.batch_counter % self.eval_every == 0:
                    batch_loss[self.batch_counter] = train_loss
                    memory_use[self.batch_counter] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                # with open(self.output_dir + '/scores.txt', 'a') as score_file:
                #     score_file.write("Score for iteration " + str(self.batch_counter) + "\n")
                # os.mkdir(self.path_logger_file + "/" + str(self.batch_counter))
                # self.path_logger_file_ = self.path_logger_file + "/" + str(self.batch_counter) + "/paths"

                #self.test(sess, beam=True, print_paths=False)
                self.save_path = self.model_saver.save(sess, self.model_dir + "model" + '.ckpt')

                with open(self.pretrained_embeddings_action_dir, 'w') as out:
                    pprint(self.agent.relation_lookup_table, stream=out)
                self.pretrained_embeddings_action = self.pretrained_embeddings_action_dir

                with open(self.pretrained_embeddings_entity_dir, 'w') as out:
                    pprint(self.agent.entity_lookup_table, stream=out)
                self.pretrained_embeddings_entity = self.pretrained_embeddings_entity_dir

                logger.info('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

                gc.collect()
        return episode_handovers_on_handover_node, batch_loss, memory_use

    def test(self, sess, beam=False, print_paths=False, save_model = True, auc = False):
        batch_counter = 0
        paths = defaultdict(list)
        answers = []
        feed_dict = {}
        all_final_reward_1 = 0
        all_final_reward_3 = 0
        all_final_reward_5 = 0
        all_final_reward_10 = 0
        all_final_reward_20 = 0
        auc = 0

        total_examples = self.test_environment.total_no_examples
        for episode in tqdm(self.test_environment.get_episodes()):
            batch_counter += 1

            temp_batch_size = episode.no_examples

            self.qr = episode.get_query_relation()
            feed_dict[self.query_relation] = self.qr
            # set initial beam probs
            beam_probs = np.zeros((temp_batch_size * self.test_rollouts, 1))
            # get initial state
            state = episode.get_state()
            mem = self.agent.get_mem_shape()
            agent_mem = np.zeros((mem[0], mem[1], temp_batch_size*self.test_rollouts, mem[3]) ).astype('float32')
            previous_relation = np.ones((temp_batch_size * self.test_rollouts, ), dtype='int64') * self.relation_vocab[
                'DUMMY_START_RELATION']
            feed_dict[self.range_arr] = np.arange(temp_batch_size * self.test_rollouts)
            feed_dict[self.input_path[0]] = np.zeros(temp_batch_size * self.test_rollouts)

            ####logger code####
            if print_paths:
                self.entity_trajectory = []
                self.relation_trajectory = []
            ####################

            self.log_probs = np.zeros((temp_batch_size*self.test_rollouts,)) * 1.0

            # for each time step
            for i in range(self.path_length):
                if i == 0:
                    feed_dict[self.first_state_of_test] = True
                feed_dict[self.next_relations] = state['next_relations']
                feed_dict[self.next_entities] = state['next_entities']
                feed_dict[self.current_entities] = state['current_entities']
                feed_dict[self.prev_state] = agent_mem
                feed_dict[self.prev_relation] = previous_relation
                feed_dict[self.handover_rnn_output] = np.zeros(
                    [self.batch_size * self.num_rollouts, self.m * self.hidden_size])
                feed_dict[self.handover_rnn_state] = [[
                    np.zeros([self.batch_size * self.num_rollouts, self.m * self.hidden_size]),
                    np.zeros([self.batch_size * self.num_rollouts, self.m * self.hidden_size])]]
                feed_dict[self.run_handover_step] = False

                loss, agent_mem, test_scores, test_action_idx, chosen_relation = sess.run(
                    [ self.test_loss, self.test_state, self.test_logits, self.test_action_idx, self.chosen_relation],
                    feed_dict=feed_dict)


                if beam:
                    k = self.test_rollouts
                    new_scores = test_scores + beam_probs
                    if i == 0:
                        idx = np.argsort(new_scores)
                        idx = idx[:, -k:]
                        ranged_idx = np.tile([b for b in range(k)], temp_batch_size)
                        idx = idx[np.arange(k*temp_batch_size), ranged_idx]
                    else:
                        idx = self.top_k(new_scores, k)

                    y = idx//self.max_num_actions
                    x = idx%self.max_num_actions

                    y += np.repeat([b*k for b in range(temp_batch_size)], k)
                    state['current_entities'] = state['current_entities'][y]
                    state['next_relations'] = state['next_relations'][y,:]
                    state['next_entities'] = state['next_entities'][y, :]
                    agent_mem = agent_mem[:, :, y, :]
                    test_action_idx = x
                    chosen_relation = state['next_relations'][np.arange(temp_batch_size*k), x]
                    beam_probs = new_scores[y, x]
                    beam_probs = beam_probs.reshape((-1, 1))
                    if print_paths:
                        for j in range(i):
                            self.entity_trajectory[j] = self.entity_trajectory[j][y]
                            self.relation_trajectory[j] = self.relation_trajectory[j][y]
                previous_relation = chosen_relation

                ####logger code####
                if print_paths:
                    self.entity_trajectory.append(state['current_entities'])
                    self.relation_trajectory.append(chosen_relation)
                ####################
                state = episode(test_action_idx)
                self.log_probs += test_scores[np.arange(self.log_probs.shape[0]), test_action_idx]
            if beam:
                self.log_probs = beam_probs

            ####Logger code####

            if print_paths:
                self.entity_trajectory.append(
                    state['current_entities'])


            # ask environment for final reward
            rewards = episode.get_reward()  # [B*test_rollouts]
            reward_reshape = np.reshape(rewards, (temp_batch_size, self.test_rollouts))  # [orig_batch, test_rollouts]
            self.log_probs = np.reshape(self.log_probs, (temp_batch_size, self.test_rollouts))
            sorted_indx = np.argsort(-self.log_probs)
            final_reward_1 = 0
            final_reward_3 = 0
            final_reward_5 = 0
            final_reward_10 = 0
            final_reward_20 = 0
            AP = 0
            ce = episode.state['current_entities'].reshape((temp_batch_size, self.test_rollouts))
            se = episode.start_entities.reshape((temp_batch_size, self.test_rollouts))
            for b in range(temp_batch_size):
                answer_pos = None
                seen = set()
                pos=0
                if self.pool == 'max':
                    for r in sorted_indx[b]:
                        if reward_reshape[b,r] == self.positive_reward:
                            answer_pos = pos
                            break
                        if ce[b, r] not in seen:
                            seen.add(ce[b, r])
                            pos += 1
                if self.pool == 'sum':
                    scores = defaultdict(list)
                    answer = ''
                    for r in sorted_indx[b]:
                        scores[ce[b,r]].append(self.log_probs[b,r])
                        if reward_reshape[b,r] == self.positive_reward:
                            answer = ce[b,r]
                    final_scores = defaultdict(float)
                    for e in scores:
                        final_scores[e] = lse(scores[e])
                    sorted_answers = sorted(final_scores, key=final_scores.get, reverse=True)
                    if answer in  sorted_answers:
                        answer_pos = sorted_answers.index(answer)
                    else:
                        answer_pos = None


                if answer_pos != None:
                    if answer_pos < 20:
                        final_reward_20 += 1
                        if answer_pos < 10:
                            final_reward_10 += 1
                            if answer_pos < 5:
                                final_reward_5 += 1
                                if answer_pos < 3:
                                    final_reward_3 += 1
                                    if answer_pos < 1:
                                        final_reward_1 += 1
                if answer_pos == None:
                    AP += 0
                else:
                    AP += 1.0/((answer_pos+1))
                if print_paths:
                    qr = self.train_environment.grapher.rev_relation_vocab[self.qr[b * self.test_rollouts]]
                    start_e = self.rev_entity_vocab[episode.start_entities[b * self.test_rollouts]]
                    end_e = self.rev_entity_vocab[episode.end_entities[b * self.test_rollouts]]
                    paths[str(qr)].append(str(start_e) + "\t" + str(end_e) + "\n")
                    paths[str(qr)].append("Reward:" + str(1 if answer_pos != None and answer_pos < 10 else 0) + "\n")
                    for r in sorted_indx[b]:
                        indx = b * self.test_rollouts + r
                        if rewards[indx] == self.positive_reward:
                            rev = 1
                        else:
                            rev = -1
                        answers.append(self.rev_entity_vocab[se[b,r]]+'\t'+ self.rev_entity_vocab[ce[b,r]]+'\t'+ str(self.log_probs[b,r])+'\n')
                        paths[str(qr)].append(
                            '\t'.join([str(self.rev_entity_vocab[e[indx]]) for e in
                                       self.entity_trajectory]) + '\n' + '\t'.join(
                                [str(self.rev_relation_vocab[re[indx]]) for re in self.relation_trajectory]) + '\n' + str(
                                rev) + '\n' + str(
                                self.log_probs[b, r]) + '\n___' + '\n')
                    paths[str(qr)].append("#####################\n")

            all_final_reward_1 += final_reward_1
            all_final_reward_3 += final_reward_3
            all_final_reward_5 += final_reward_5
            all_final_reward_10 += final_reward_10
            all_final_reward_20 += final_reward_20
            auc += AP

        all_final_reward_1 /= total_examples
        all_final_reward_3 /= total_examples
        all_final_reward_5 /= total_examples
        all_final_reward_10 /= total_examples
        all_final_reward_20 /= total_examples
        auc /= total_examples
        if save_model:
            if all_final_reward_10 >= self.max_hits_at_10:
                self.max_hits_at_10 = all_final_reward_10
                #self.save_path = self.model_saver.save(sess, self.model_dir + "model" + '.ckpt')

        if print_paths:
            logger.info("[ printing paths at {} ]".format(self.output_dir+'/test_beam/'))
            for q in paths:
                j = q.replace('/', '-')
                with codecs.open(self.path_logger_file_ + '_' + j, 'a', 'utf-8') as pos_file:
                    for p in paths[q]:
                        pos_file.write(p)
            with open(self.path_logger_file_ + 'answers', 'w') as answer_file:
                for a in answers:
                    answer_file.write(a)

        score = {}
        score["Hits@1"] = all_final_reward_1
        score["Hits@3"] = all_final_reward_3
        score["Hits@5"] = all_final_reward_5
        score["Hits@10"] = all_final_reward_10
        score["Hits@20"] = all_final_reward_20
        score["auc"] = auc

        # with open(self.output_dir + '/scores.txt', 'a') as score_file:
        #     score_file.write("Hits@1: {0:7.4f}".format(all_final_reward_1))
        #     score_file.write("\n")
        #     score_file.write("Hits@3: {0:7.4f}".format(all_final_reward_3))
        #     score_file.write("\n")
        #     score_file.write("Hits@5: {0:7.4f}".format(all_final_reward_5))
        #     score_file.write("\n")
        #     score_file.write("Hits@10: {0:7.4f}".format(all_final_reward_10))
        #     score_file.write("\n")
        #     score_file.write("Hits@20: {0:7.4f}".format(all_final_reward_20))
        #     score_file.write("\n")
        #     score_file.write("auc: {0:7.4f}".format(auc))
        #     score_file.write("\n")
        #     score_file.write("\n")

        logger.info("Hits@1: {0:7.4f}".format(all_final_reward_1))
        logger.info("Hits@3: {0:7.4f}".format(all_final_reward_3))
        logger.info("Hits@5: {0:7.4f}".format(all_final_reward_5))
        logger.info("Hits@10: {0:7.4f}".format(all_final_reward_10))
        logger.info("Hits@20: {0:7.4f}".format(all_final_reward_20))
        logger.info("auc: {0:7.4f}".format(auc))

        return score

    def top_k(self, scores, k):
        scores = scores.reshape(-1, k * self.max_num_actions)  # [B, (k*max_num_actions)]
        idx = np.argsort(scores, axis=1)
        idx = idx[:, -k:]  # take the last k highest indices # [B , k]
        return idx.reshape((-1))

def test_auc(options, save_path, path_logger_file, output_dir, data_input_dir=None):
    trainer = Trainer(options)
    # 直接读取模型
    if options['load_model']:
        save_path = options['model_load_dir']
        path_logger_file = trainer.path_logger_file
        output_dir = trainer.output_dir

    # Testing
    with tf.compat.v1.Session(config=config) as sess:
        trainer.initialize(restore=save_path, sess=sess)

        trainer.test_rollouts = 20

        if not os.path.isdir(path_logger_file + "/" + "test_beam"):
            os.mkdir(path_logger_file + "/" + "test_beam")
        trainer.path_logger_file_ = path_logger_file + "/" + "test_beam" + "/paths"
        with open(output_dir + '/scores.txt', 'a') as score_file:
            score_file.write("Test (beam) scores with best model from " + save_path + "\n")
        trainer.test_environment = trainer.test_test_environment
        trainer.test_environment.test_rollouts = 20

        score = trainer.test(sess, beam=True, print_paths=True, save_model=False)

        if options['nell_evaluation'] == 1:
            nell_eval(path_logger_file + "/" + "test_beam/" + "pathsanswers",
                      data_input_dir + '/sort_test.pairs')

    tf.compat.v1.reset_default_graph()
    return score
def test_auc_avg(save_path, path_logger_file, output_dir, trainer, sess, data_input_dir=None):
    # trainer = Trainer(options)

    # Testing
    # with tf.compat.v1.Session(config=config) as sess:
    # trainer.initialize(restore=save_path, sess=sess)

    trainer.test_rollouts = 20

    if not os.path.isdir(path_logger_file + "/" + "test_beam"):
        os.mkdir(path_logger_file + "/" + "test_beam")
    trainer.path_logger_file_ = path_logger_file + "/" + "test_beam" + "/paths"
    with open(output_dir + '/scores.txt', 'a') as score_file:
        score_file.write("Test (beam) scores with best model from " + save_path + "\n")
    trainer.test_environment = trainer.test_test_environment
    trainer.test_environment.test_rollouts = 20

    score = trainer.test(sess, beam=True, print_paths=True, save_model=False)
    # tf.compat.v1.reset_default_graph()
    return score

def train_multi_agents(options, agent_names, triple_count_max=None, iter=None):
    episode_handovers = {}
    evaluation = {}
    batch_loss = {}
    memory_use = {}
    model_path = {}
    evaluation[1] = {}
    batch_loss[1] = {}
    memory_use[1] = {}
    model_path[1] = {}

    ho_count = {1: {}}
    save_path = ""
    for i in range(len(agent_names)):
        trainer = Trainer(options, agent_names[i])

        with tf.compat.v1.Session(config=config) as sess:
            # 初始化训练模型
            if i == 0:
                sess.run(trainer.initialize())
            else:
                sess.run(trainer.initialize())
            trainer.initialize_pretrained_embeddings(sess=sess)

            # 训练
            episode_handover_for_agent, batch_loss_for_agent, memory_use_for_agent = trainer.train(sess)
            episode_handovers[agent_names[i]] = episode_handover_for_agent
            save_path = trainer.save_path
            path_logger_file = trainer.path_logger_file
            output_dir = trainer.output_dir

        tf.compat.v1.reset_default_graph()

        score = test_auc(options, save_path, path_logger_file, output_dir)

        iter_string = ""
        if triple_count_max:
            iter_string = iter_string + "_" + str(triple_count_max)
        if iter:
            iter_string = iter_string + "_" + str(iter)
        evaluation[1][agent_names[i] + iter_string] = score
        batch_loss[1][agent_names[i] + iter_string] = batch_loss_for_agent
        memory_use[1][agent_names[i] + iter_string] = memory_use_for_agent
        model_path[1][agent_names[i] + iter_string] = save_path
        if i == 0:
            # 打头的计算信心值，后续一并在打头的基础上计算信心值
            for idx in range(len(agent_names)):
                count, used_entities_value_set = calc_confident_indicator(options, agent_names, {}, 1, idx,
                                                 episode_handover_for_agent)
                ho_count[1][agent_names[idx]] = count


    return evaluation, batch_loss, memory_use, model_path,ho_count

def train_multi_agents_with_transfer(options, agent_names, agent_training_order, triple_count_max=None, iter=None):
    episode_handovers = {}
    evaluation = {}
    batch_loss = {}
    memory_use = {}
    save_path = ""
    for agent_order in agent_training_order:
        evaluation[agent_order] = {}
        batch_loss[agent_order] = {}
        memory_use[agent_order] = {}
        for i in agent_training_order[agent_order]:
            i = i-1
            trainer = Trainer(options, agent_names[i])

            with tf.compat.v1.Session(config=config) as sess:
                # 初始化训练模型
                if save_path == "":
                    sess.run(trainer.initialize())
                else:
                    trainer.initialize(restore=save_path, sess=sess)
                trainer.initialize_pretrained_embeddings(sess=sess)

                # 训练
                episode_handover_for_agent, batch_loss_for_agent, memory_use_for_agent = trainer.train(sess)
                episode_handovers[agent_names[i]] = episode_handover_for_agent
                save_path = trainer.save_path
                path_logger_file = trainer.path_logger_file
                output_dir = trainer.output_dir

            tf.compat.v1.reset_default_graph()

            score = test_auc(options, save_path, path_logger_file, output_dir)

            iter_string = ""
            if triple_count_max:
                iter_string = iter_string + "_" + str(triple_count_max)
            if iter:
                iter_string = iter_string + "_" + str(iter)
            evaluation[agent_order][agent_names[i] + " transferred " + iter_string] = score
            batch_loss[agent_order][agent_names[i] + " transferred " + iter_string] = batch_loss_for_agent
            memory_use[agent_order][agent_names[i] + " transferred " + iter_string] = memory_use_for_agent

    return evaluation, batch_loss, memory_use

def train_multi_agents_with_handover_query(options, agent_names, agent_training_order, agent_training_non_coo=None,triple_count_max=None, iter=None,
                                           sorted_flag=False, sorted_flag_with_non_coo=False, more_loop_count=0):
    episode_handovers = {}
    evaluation = {}
    batch_loss = {}
    memory_use = {}
    ho_count = {}
    ho_ratio = {}

    for agent_order in agent_training_order:
        evaluation[agent_order] = {}
        batch_loss[agent_order] = {}
        memory_use[agent_order] = {}
        ho_count[agent_order] = {}
        ho_ratio[agent_order] = {}
        save_path = None

        i = agent_training_order[agent_order][0] - 1
        trainer = Trainer(options, agent_names[i])

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = False
        config.log_device_placement = False

        with tf.compat.v1.Session(config=config) as sess:
            # 初始化训练模型
            if save_path is None:
                sess.run(trainer.initialize())
            else:
                sess.run(trainer.initialize())
            trainer.initialize_pretrained_embeddings(sess=sess)

            # 训练
            episode_handover_for_agent, batch_loss_for_agent, memory_use_for_agent = trainer.train(sess)
            tvars = tf.compat.v1.trainable_variables()
            episode_handovers[agent_names[i]] = episode_handover_for_agent
            save_path = trainer.save_path
            path_logger_file = trainer.path_logger_file
            output_dir = trainer.output_dir

        tf.compat.v1.reset_default_graph()

        score = test_auc(options, save_path, path_logger_file, output_dir)

        iter_string = ""
        if triple_count_max:
            iter_string = iter_string + "_" + str(triple_count_max)
        if iter:
            iter_string = iter_string + "_" + str(iter)
        evaluation[agent_order][agent_names[i] + iter_string] = score
        batch_loss[agent_order][agent_names[i] + iter_string] = batch_loss_for_agent
        memory_use[agent_order][agent_names[i] + iter_string] = memory_use_for_agent
        sorted_flag = sorted_flag
        sorted_flag_with_non_coo = sorted_flag_with_non_coo
        agent_training_non_coo = agent_training_non_coo

        if sorted_flag:
            print("sorted_flag:", sorted_flag)
            counts = []
            used_entities_value_array = []
            for idx in range(len(agent_training_order[agent_order])):

                count, used_entities_value_set = calc_confident_indicator(options, agent_names, agent_training_order, agent_order, idx,
                                         episode_handover_for_agent)
                if idx == 0:
                    ho_count[agent_order][agent_names[agent_training_order[agent_order][idx] - 1]] = count
                else:
                    ho_count[agent_order]["continued on " + agent_names[agent_training_order[agent_order][idx] - 1]] = count
                counts.append(count)
                used_entities_value_array.append(used_entities_value_set)
                #print(count)

            query_ratio = {}
            for c in range(len(counts)):
                query_ratio[counts[c]] = c
                if c == 0:
                    ho_ratio[agent_order][agent_names[agent_training_order[agent_order][c] - 1]] = counts[c]
                else:
                    ho_ratio[agent_order]["continued on " + agent_names[agent_training_order[agent_order][c] - 1]] = counts[c]

            sorted(query_ratio.keys())
            print("counts:", counts)
            print("query_ratio:", query_ratio)
            for used_entities_value in used_entities_value_array:
                print("used_entities_value_set len", len(used_entities_value))
            sort_count_list = sorted(query_ratio.keys(), reverse=True)
            sorted_count_and_agent_name_map = [{count_: agent_training_order[agent_order][query_ratio[count_]]} for count_ in sort_count_list]
            # 打印排序后 信息值与agent name的对应关系
            print("query_ratio sorted_count_and_agent_name_map:", sorted_count_and_agent_name_map)
            for count_agent_name in sorted_count_and_agent_name_map:
                print("count_agent_name: ", count_agent_name)
            print("sorted(query_ratio.keys(), reverse=True):", sorted(query_ratio.keys(), reverse=True))

            # continue
            # return evaluation, batch_loss, memory_use, ho_count, ho_ratio
            if sorted_flag_with_non_coo:
                # agent_training_non_coo = {
                #         1: [3], # 不合作的
                #         2: [2],
                #         3: [3],
                #         4: [1],
                #         5: [2],
                #         6: [1]
                #         # 4: [4, 1, 2, 3, 5, 6, 7, 8],
                #         # 5: [5, 1, 2, 3, 4, 6, 7, 8],
                #         # 6: [6, 1, 2, 3, 4, 5, 7, 8],
                #         # 7: [7, 1, 2, 3, 4, 5, 6, 8],
                #         # 8: [8, 1, 2, 3, 4, 5, 6, 7]
                #     }

                while query_ratio:  # 合作的训练完就算结束
                    agent_key_with_max_auc = max(query_ratio)
                    if query_ratio[agent_key_with_max_auc] == 0:
                        # 去除打头的agent
                        del query_ratio[agent_key_with_max_auc]
                        continue
                    else:
                        # 还有未训练的合作的agent
                        # 不合作的举手数量 示例：5个agent 有可能 0，1，2，3，4，5个举手的，相当于range(6)
                        # 与下面的竞争搭配后，相当于大家举手完全独立随机
                        hands_up_count = random.choice(range(len(agent_training_non_coo[agent_order])+1))
                        # 先拿出来不合作的举手的agent列表
                        non_coo_whit_hands_up = random.choices(agent_training_non_coo[agent_order], k=hands_up_count)
                        # 信心值最大的 与 举手的不合作的竞争 概率拉平随机拿一个
                        coo_inx = agent_training_order[agent_order][query_ratio[agent_key_with_max_auc]]
                        choice_agent_idx = random.choice(non_coo_whit_hands_up + [coo_inx])
                        non_coo = False if choice_agent_idx == coo_inx else True
                        continue_training_with_handover_query(options, agent_names, agent_training_order, agent_order,
                                                              choice_agent_idx,
                                                              episode_handover_for_agent, evaluation, batch_loss,
                                                              memory_use, save_path,
                                                              config, whit_non_coo=True, non_coo=non_coo)
                        if choice_agent_idx == coo_inx:
                            # 选到了合作的 去掉
                            del query_ratio[agent_key_with_max_auc]
                        else:
                            # 选到了不合作的 去掉
                            print("agent_training_non_coo:", choice_agent_idx)
                            agent_training_non_coo[agent_order].remove(choice_agent_idx)
                more_loop_count = more_loop_count
                # more_loop_count = 2  # 合作的跑完了后，还有不合作的存在的话，再取两次
                while more_loop_count and agent_training_non_coo[agent_order]:
                    more_loop_count -= 1
                    hands_up_count = random.choice(range(len(agent_training_non_coo[agent_order]) + 1))
                    non_coo_whit_hands_up = random.choices(agent_training_non_coo[agent_order], k=hands_up_count)
                    if not non_coo_whit_hands_up:  # 没有举手的
                        continue
                    # if not random.choice([0, 1]):  # 没有被选上
                    #     continue
                    choice_agent_idx = random.choice(non_coo_whit_hands_up)
                    continue_training_with_handover_query(options, agent_names, agent_training_order, agent_order,
                                                          choice_agent_idx,
                                                          episode_handover_for_agent, evaluation, batch_loss,
                                                          memory_use, save_path,
                                                          config, whit_non_coo=True, non_coo=True)
                    # 选到了不合作的 去掉
                    print("agent_training_non_coo:", choice_agent_idx)
                    agent_training_non_coo[agent_order].remove(choice_agent_idx)


            if not sorted_flag_with_non_coo:
                #order_index = list(query_ratio.keys())
                #random.shuffle(order_index)
                for q_r_sorted in sorted(query_ratio.keys()):
                #for q_r_sorted in order_index:
                    if query_ratio[q_r_sorted] == 0:
                        continue
                    continue_training_with_handover_query(options, agent_names, agent_training_order, agent_order, query_ratio[q_r_sorted],
                                                      episode_handover_for_agent, evaluation, batch_loss, memory_use, save_path,
                                                      config)
        else:
            for agent_idx in range(len(agent_training_order[agent_order])):
                if agent_idx == 0:
                    continue
                j = agent_training_order[agent_order][agent_idx] - 1

                trainer = Trainer(options, agent_names[j], isTrainHandover=True)

                with tf.compat.v1.Session(config=config) as sess:
                    # 初始化训练模型

                    trainer.initialize(restore=save_path, sess=sess)
                    trainer.initialize_pretrained_embeddings(sess=sess)

                    # 训练
                    episode_handovers_on_handover_node, batch_loss_for_agent, memory_use_for_agent = trainer.train_full_episode(
                        sess, episode_handover_for_agent)
                    # episode_handover_for_agent = episode_handovers_on_handover_node

                    save_path = trainer.save_path
                    path_logger_file = trainer.path_logger_file
                    output_dir = trainer.output_dir

                tf.compat.v1.reset_default_graph()
                if save_path:
                    score = test_auc(options, save_path, path_logger_file, output_dir)
                    evaluation[agent_order]["continued on " + agent_names[j]] = score
                    batch_loss[agent_order]["continued on " + agent_names[j]] = batch_loss_for_agent
                    memory_use[agent_order]["continued on " + agent_names[j]] = memory_use_for_agent

    return evaluation, batch_loss, memory_use, ho_count, ho_ratio

def continue_training_with_handover_query(options, agent_names, agent_training_order, order_idx, agent_idx,
                                          episode_handover_for_agent, evaluation, batch_loss, memory_use, save_path, config,
                                          whit_non_coo=False, non_coo=False):
    # query_ratio


    p = ""
    # for idx in range(len(agent_training_order[order_idx])):
    #     if idx == 0:
    #         continue
    # 取索引有两种方式
    if whit_non_coo:
        # 有不合作的
        j = agent_idx - 1
    else:
        j = agent_training_order[order_idx][agent_idx] - 1

    trainer = Trainer(options, agent_names[j], isTrainHandover=True)

    with tf.compat.v1.Session(config=config) as sess:
        # 初始化训练模型

        trainer.initialize(restore=save_path, sess=sess)
        trainer.initialize_pretrained_embeddings(sess=sess)

        # 训练
        episode_handovers_on_handover_node, batch_loss_for_agent, memory_use_for_agent = trainer.train_full_episode(sess, episode_handover_for_agent)
        #episode_handover_for_agent = episode_handovers_on_handover_node

        save_path = trainer.save_path
        path_logger_file = trainer.path_logger_file
        output_dir = trainer.output_dir

    tf.compat.v1.reset_default_graph()
    if save_path:
        score = test_auc(options, save_path, path_logger_file, output_dir)
        if not non_coo:
            evaluation[order_idx]["continued on " + agent_names[j]] = score
            batch_loss[order_idx]["continued on " + agent_names[j]] = batch_loss_for_agent
            memory_use[order_idx]["continued on " + agent_names[j]] = memory_use_for_agent
        else:
            evaluation[order_idx]["continued on non_coo " + agent_names[j]] = score
            batch_loss[order_idx]["continued on non_coo " + agent_names[j]] = batch_loss_for_agent
            memory_use[order_idx]["continued on non_coo " + agent_names[j]] = memory_use_for_agent

    return evaluation, batch_loss, memory_use

def calc_confident_indicator(options, agent_names, agent_training_order, order_idx, agent_idx, episode_handover_for_agent):

    count = 0
    if agent_training_order:
        j = agent_training_order[order_idx][agent_idx] - 1
    else:
        j = agent_idx
    print(agent_names[j])
    train_environment = env(options, agent_names[j])
    # 初始空set装所有用过的
    used_entities_value_set = set()
    for episode_handovers in episode_handover_for_agent:
        for I, handover in episode_handover_for_agent[episode_handovers]:
            # 深拷贝，不影响原数据
            np_tmp_1x1700 = copy.deepcopy(handover['current_entities'])  # 1*1700
            # 不重复的重新组数据
            handled_entities = []
            for i in np_tmp_1x1700:
                # 不重复原值 重复给0
                tmp_entities = i if i not in used_entities_value_set else 0
                handled_entities.append(tmp_entities)
                used_entities_value_set.add(i)
            # 转回np数组不影响后续
            handled_entities = np.array(handled_entities)
            # action_entity = train_environment.grapher.array_store[handover['current_entities'],:,:]
            for i in handled_entities:
                if i > 1:
                    action_entity = train_environment.grapher.array_store[i,:,:]
                    if len(action_entity):
                        # print("action_entity:", action_entity)
                        for action in action_entity:
                            if action[0] > 0 and action[1] > 2:
                                # print("action[0] > 0 and action[1] > 2:", action)
                                count += 1
                                break
                    else:
                        print("len(action_entity) == 0 ", action_entity)

            np_1 = handover['current_entities']
            np_2 = handled_entities
            #print("np_1 handover:", np_1.shape, np_1.dtype, np_1.size, np_1.ndim, np_1)
            # print("np_2 handover:", np_2.shape, np_2.dtype, np_2.size, np_2.ndim, np_2)
            # a = np.array([[[1,2,3,4],[5,6,7,8],[9,10,11,12]], \
            #               [[13,14,15,16],[17,18,19,20],[21,22,23,24]], \
            #               [[25,26,27,28],[29,30,31,32],[33,34,35,36]]])
            # b = np.array([[[],[],[]],[[],[],[]],[[],[],[]]])

            # for current_entity in action_entity:
            #     for action in current_entity:
            #         if action[0] > 0 and action[1] > 0:
            #             count += 1

    return count, used_entities_value_set


def save_result_to_excel(data_splitter, evaluation, batch_loss, memory_use, ho_count = None, ho_ratio = None):
    with open(options['output_dir'] + '/test/8_8scores.csv', 'w') as evaluation_score:
        writer = csv.writer(evaluation_score, delimiter=',')
        writer.writerow(
            ["item", "triple_count", "entity_count", "relation_count", "ho_count", "ho_ratio", "Hits@1", "Hits@3", "Hits@5", "Hits@10",
             "Hits@20", "auc"])
        for round in evaluation:
            for i in evaluation[round]:
                row = []
                row.append(i)
                grapher_triple_per_count = data_splitter.get_grapher_triple_per_count()
                if i in grapher_triple_per_count.keys():
                    row.append(grapher_triple_per_count[i])
                else:
                    row.append("")
                if i in data_splitter.get_grapher_entity_per_count().keys():
                    row.append(len(data_splitter.get_grapher_entity_per_count()[i]))
                else:
                    row.append("")
                if i in data_splitter.get_grapher_relation_per_count().keys():
                    row.append(len(data_splitter.get_grapher_relation_per_count()[i]))
                else:
                    row.append("")
                if ho_count is not None and i in ho_count[round].keys():
                    row.append(ho_count[round][i])
                else:
                    row.append("")
                if ho_ratio is not None and i in ho_ratio[round].keys():
                    row.append(ho_ratio[round][i])
                else:
                    row.append("")
                for v in evaluation[round][i].values():
                    row.append(v)
                writer.writerow(row)
            row = []
            row.append("line break")
            writer.writerow(row)
    with open(options['output_dir'] + '/test/batch_loss.csv', 'w') as batch_loss_writer:
        writer = csv.writer(batch_loss_writer, delimiter=',')
        writer.writerow(["item", "batch_count", "loss"])
        for round in batch_loss:
            for i in batch_loss[round]:
                for j in batch_loss[round][i]:
                    row = []
                    row.append(i)
                    row.append(j)
                    row.append(batch_loss[round][i][j])
                    writer.writerow(row)
    with open(options['output_dir'] + '/test/memory_use.csv', 'w') as memory_use_writer:
        writer = csv.writer(memory_use_writer, delimiter=',')
        writer.writerow(["item", "batch_count", "memory_use"])
        for round in memory_use:
            for i in memory_use[round]:
                for j in memory_use[round][i]:
                    row = []
                    row.append(i)
                    row.append(j)
                    row.append(memory_use[round][i][j])
                    writer.writerow(row)
def save_avg_model_auc_to_excel(evaluation):
    with open(options['output_dir'] + '/test/8_8scores.csv', 'a') as evaluation_score:
        writer = csv.writer(evaluation_score, delimiter=',')
        for round in evaluation:
            for i in evaluation[round]:
                row = []
                row.append(i)
                for _ in range(5):
                    row.append("")
                for v in evaluation[round][i].values():
                    row.append(v)
                writer.writerow(row)
            row = []
            row.append("line break")
            writer.writerow(row)

def test_avg_model(agent_name, trainer, sess):
    episode_handovers = {}
    evaluation = {}
    batch_loss = {}
    memory_use = {}
    model_path = {}
    evaluation[1] = {}
    batch_loss[1] = {}
    memory_use[1] = {}
    model_path[1] = {}

    save_path = ""
    # for i in range(len(agent_names)):
    # trainer = Trainer(options, agent_names[i])
    save_path = trainer.save_path if trainer.save_path else ""
    path_logger_file = trainer.path_logger_file
    output_dir = trainer.output_dir
    trainer.test_rollouts = 20

    # tf.compat.v1.reset_default_graph()

    score = test_auc_avg(save_path, path_logger_file, output_dir, trainer, sess)
    print("avg score:", score)
    # print("avg score:", score.values())

    evaluation[1][agent_name] = score
    return evaluation


if __name__ == '__main__':
    # read command line options
    options = read_options("test_multi_agent_" + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time())))

    if options['distributed_training']:
        # agent_names = ['agent_1', 'agent_2', 'agent_3']
        agent_names = ['agent_1', 'agent_2', 'agent_3', 'agent_4', 'agent_5', 'agent_6', 'agent_7', 'agent_8']
        # agent_names = ['agent_1', 'agent_5', 'agent_6', 'agent_8', 'agent_7', 'agent_2', 'agent_4', 'agent_3']
    else:
        agent_names = ['agent_full']

    data_splitter = DataDistributor()
    # data_splitter.split(options, agent_names)

    # Set logging
    logger.setLevel(logging.WARNING)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    logfile = logging.FileHandler(options['log_file_name'], 'w')
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)
    # read the vocab files, it will be used by many classes hence global scope
    logger.info('reading vocab files...')
    options['relation_vocab'] = json.load(open(options['vocab_dir'] + '/relation_vocab.json'))
    options['entity_vocab'] = json.load(open(options['vocab_dir'] + '/entity_vocab.json'))
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

    # 2后随机挑1个合作的，其余为不合作的，跑第一次 如 [2, 1]+[3, 4, 5, 6, 7, 8] agent_training_order + agent_training_non_coo
    # 2后随机挑1个合作的，其余为不合作的，跑第二次 如 [2, 6]+[1, 3, 4, 5, 7, 8] agent_training_order + agent_training_non_coo
    # 2后随机挑2个合作的，其余为不合作的，跑第一次 如 [2, 1, 8]+[3, 4, 5, 6, 7] agent_training_order + agent_training_non_coo
    # 2后随机挑2个合作的，其余为不合作的，跑第二次 如 [2, 3, 4]+[1, 5, 6, 7, 8] agent_training_order + agent_training_non_coo
    # 2后随机挑3个合作的，其余为不合作的，跑第一次 如 [2, 3, 6, 8]+[1, 5, 7, 8] agent_training_order + agent_training_non_coo
    # 2后随机挑3个合作的，其余为不合作的，跑第二次 如 [2, 1, 5, 6]+[3, 4, 7, 8] agent_training_order + agent_training_non_coo
    # 2后随机挑4个合作的，其余为不合作的，跑第一次 如 [2, 1, 3 ,4, 5]+[6, 7, 8] agent_training_order + agent_training_non_coo
    # 2后随机挑4个合作的，其余为不合作的，跑第二次 如 [2, 3, 5, 7, 8]+[1, 4, 6] agent_training_order + agent_training_non_coo
    # 挑n个合作的，n一般 <= count(agent)-1，如8个agent，n一般小于等于7
    # 跑x次
    # agent_training_order = {}  # 头部+后续合作的
    agent_training_non_coo = {}  # 不合作的
    # coo_count_loop = [1, 2, 3, 4]  # 每次挑出n个合作的情况
    # order_start_idx = 1  # 起始索引：agent_training_order 和 agent_training_non_coo 的
    # batch_count = 2  # 跑x次
    # start_agent = 2  # 起始agent：agent_training_order的
    # other_agent = [1, 3, 4, 5, 6, 7, 8]
    # for coo_count in coo_count_loop:  # 每次挑出n个合作的情况
    #     for i in range(batch_count):  # 跑x次
    #         random.shuffle(other_agent)  # 每次打乱，取前n个作为合作的，剩下为不合作的
    #         # 取前n个作为合作的, sorted 排序下比较好看些
    #         agent_training_order[order_start_idx] = [start_agent] + sorted(other_agent[:coo_count])
    #         # 剩下为不合作的, sorted 排序下比较好看些
    #         agent_training_non_coo[order_start_idx] = sorted(other_agent[coo_count:])
    #         order_start_idx += 1
    # print("agent_training_order:")
    # for k_, v_ in agent_training_order.items():
    #     print(k_, ":", v_)
    # print("agent_training_non_coo:")
    # for k_, v_ in agent_training_non_coo.items():
    #     print(k_, ":", v_)



    agent_training_order = {
    #     # 1: [1, 2, 3, 4], # 头部+后续合作的
    #     # 2: [2, 1, 3, 4],
    #     # 3: [3, 1, 2, 4],
    #     # 4: [4, 1, 2, 3],
    #     # 5: [5, 6, 7, 8],
    #     # 6: [6, 5, 7, 8],
    #     # 7: [7, 5, 6, 8],
    #     # 8: [8, 5, 6, 7],
    #     1: [2, 1, 3, 4, 5, 6, 7, 8],  # 2后随机挑1个合作的，其余为不合作的，跑第一次 如 [2, 1] [3, 4, 5, 6, 7, 8]
    #     2: [2, 1, 3, 4, 5, 6, 7, 8],  # 2后随机挑1个合作的，其余为不合作的，跑第二次 如 [2, 6] [1, 3, 4, 5, 7, 8]
    #     3: [2, 1, 3, 4, 5, 6, 7, 8],  # 2后随机挑2个合作的，其余为不合作的，跑第一次 如 [2, 1, 8] [3, 4, 5, 6, 7]
    #     4: [2, 1, 3, 4, 5, 6, 7, 8],  # 2后随机挑2个合作的，其余为不合作的，跑第二次 如 [2, 3, 4] [1, 5, 6, 7, 8]
    #     5: [2, 1, 3, 4, 5, 6, 7, 8],  # 2后随机挑3个合作的，其余为不合作的，跑第一次 如 [2, 3, 6, 8] [1, 5, 7, 8]
    #     6: [2, 1, 3, 4, 5, 6, 7, 8],  # 2后随机挑3个合作的，其余为不合作的，跑第二次 如 [2, 1, 5, 6] [3, 4, 7, 8]
    #     7: [2, 1, 3, 4, 5, 6, 7, 8],  # 2后随机挑4个合作的，其余为不合作的，跑第一次 如 [2, 1, 3 ,4, 5] [6, 7, 8]
    #     8: [2, 1, 3, 4, 5, 6, 7, 8]   # 2后随机挑4个合作的，其余为不合作的，跑第二次 如 [2, 3, 5, 7, 8] [1, 4, 6]
        1: [1, 5, 6, 8, 7, 4, 2, 3],
        2: [2, 1, 3, 4, 5, 6, 7, 8],
        3: [3, 1, 2, 4, 5, 6, 7, 8],
        4: [4, 1, 2, 3, 5, 6, 7, 8],
        5: [5, 1, 2, 3, 4, 6, 7, 8],
        6: [6, 1, 2, 3, 4, 5, 7, 8],
        7: [7, 1, 2, 3, 4, 5, 6, 8],
        8: [8, 1, 2, 3, 4, 5, 6, 7]
    }
    #
    # agent_training_non_coo = {
    #     1: [5, 6, 7, 8], # 不合作的
    #     2: [5, 6, 7, 8],
    #     3: [5, 6, 7, 8],
    #     4: [5, 6, 7, 8],
    #     5: [1, 2, 3, 4],
    #     6: [1, 2, 3, 4],
    #     7: [1, 2, 3, 4],
    #     8: [1, 2, 3, 4]
    #     # 4: [4, 1, 2, 3, 5, 6, 7, 8],
    #     # 5: [5, 1, 2, 3, 4, 6, 7, 8],
    #     # 6: [6, 1, 2, 3, 4, 5, 7, 8],
    #     # 7: [7, 1, 2, 3, 4, 5, 6, 8],
    #     # 8: [8, 1, 2, 3, 4, 5, 6, 7]
    # }
    # 【测试修改模型变量】
    # agent = "agent_1"
    # cal_trainable_variables = {}
    # trainer = Trainer(options, agent)
    # with tf.compat.v1.Session(config=config) as sess:
    #     path = "output/WN18RR/test_multi_agent_2022-10-08-15-45-23/agent_1/model/model.ckpt"
    #     trainer.initialize(restore=path, sess=sess)
    #     cal_trainable_variables[agent] = {}
    #     new_var_to_restore_list = []
    #     for i in range(len(tf.compat.v1.trainable_variables())):
    #         # 取模型文件的变量数据
    #         current_var = tf.compat.v1.trainable_variables()[i]
    #         cal_trainable_variables[agent][current_var.name] = current_var + 1
    #         var_name = current_var.name.split(":")[0]
    #         print("current_var.name:", var_name)
    #         print("current_var:", current_var.eval())
    #         # 预览变量运算的效果
    #         print("current_var+1:", (current_var + 1).eval())
    #         var_new = tf.compat.v1.convert_to_tensor(current_var + 1)
    #         var_assign = tf.compat.v1.assign(current_var, var_new)
    #         new_var_to_restore_list.append(var_assign)
    #         # else:
    #         #     new_var_to_restore_list.append(v)
    #     for new_var_to_restore in new_var_to_restore_list:
    #         sess.run(new_var_to_restore)
    #
    #     for i in range(len(tf.compat.v1.trainable_variables())):
    #         # 再次取模型文件的变量数据
    #         current_var = tf.compat.v1.trainable_variables()[i]
    #         cal_trainable_variables[agent][current_var.name] = current_var.eval()
    #         var_name = current_var.name.split(":")[0]
    #         var_value =current_var.eval()
    #         print("after current_var.name:", var_name)
    #         print("after current_var.eval():", var_value)
    #
    #     # print("cal_trainable_variables:", cal_trainable_variables)
    # exit(1)

    # Training
    # 不直接读取模型
    if not options['load_model']:
        episode_handovers = {}

        if not options['transferred_training']:
            evaluation_per_agent, batch_loss_per_agent, memory_use_per_agent, model_path, ho_count = train_multi_agents(options, agent_names)
            if options['distributed_training']:
                save_result_to_excel(data_splitter, evaluation_per_agent, batch_loss_per_agent, memory_use_per_agent, ho_count)

            cal_trainable_variables = {}
            avg_trainable_variables = {}
            avg_length_loop = [2, 3, 4, 5, 6, 7, 8]
            # 前n个agent做平均后再测试
            for avg_length in avg_length_loop:
                cal_trainable_variables[avg_length] = {}
                avg_trainable_variables[avg_length] = {}
                # 1.加载相应模型，并获取变量
                for train_order in model_path.keys():
                    for agent in model_path[train_order]:
                        if agent not in agent_names[:avg_length]: # 不在范围内的不做计算
                            print("计算前", avg_length, "个agent平均值，当前agent：", agent, "不在计算之内")
                            continue
                        trainer = Trainer(options, agent)
                        with tf.compat.v1.Session(config=config) as sess:
                            # 初始化训练模型
                            if model_path[train_order][agent] == "":
                                sess.run(trainer.initialize())
                            else:
                                print("model_path[train_order][agent]:", model_path[train_order][agent])
                                trainer.initialize(restore=model_path[train_order][agent], sess=sess)
                                cal_trainable_variables[avg_length][agent] = {}
                                for i in range(len(tf.compat.v1.trainable_variables())):
                                    current_var = tf.compat.v1.trainable_variables()[i]
                                    var_name = current_var.name.split(":")[0]
                                    if var_name not in avg_trainable_variables[avg_length]:
                                        avg_trainable_variables[avg_length][var_name] = []
                                    avg_trainable_variables[avg_length][var_name].append(current_var.eval())
                                    cal_trainable_variables[avg_length][agent][var_name] = current_var.eval()
                                    # print("current_var.name:", current_var.name)
                                    # print("current_var.eval():", current_var.eval())
                        # tf.compat.v1.enable_eager_execution()
                        # tf.compat.v1.disable_v2_behavior()
                        tf.compat.v1.reset_default_graph()
                # 2.计算变量平均值
                for k_, var_list in avg_trainable_variables[avg_length].items():
                    print("k_:", k_)
                    print("len(var_list):", len(var_list))
                    avg_trainable_variables[avg_length][k_] = sum(var_list)/len(var_list)
                    # print("avg(var_list):", avg_trainable_variables[avg_length][k_])

                # print(cal_trainable_variables)
                # print(avg_trainable_variables)
                # 3.将agent_1的模型拿出来载入变量的平均值
                # 跑多次看平均是不是一致
                last_agent_name = agent_names[avg_length-1]  # 每次取最后一个agent作为基础模型并测试
                avg_loop = [(last_agent_name, "avg_base_" + last_agent_name)]
                for base_agent, avg_agent in avg_loop:
                    # base_agent = "agent_1"
                    # avg_agent = "agent_avg_base_1"
                    # base_agent = "agent_2"
                    # avg_agent = "agent_avg_base_2"
                    cal_trainable_variables[avg_length] = {}
                    trainer = Trainer(options, base_agent)
                    with tf.compat.v1.Session(config=config) as sess:
                        path = model_path[1][base_agent]
                        # path = "output/WN18RR/test_multi_agent_2022-10-08-15-45-23/agent_1/model/model.ckpt"
                        trainer.initialize(restore=path, sess=sess)
                        cal_trainable_variables[avg_length][avg_agent] = {}
                        new_var_to_restore_list = []
                        for i in range(len(tf.compat.v1.trainable_variables())):
                            # 取模型文件的变量数据
                            current_var = tf.compat.v1.trainable_variables()[i]
                            var_name = current_var.name.split(":")[0]
                            if var_name in avg_trainable_variables[avg_length]:
                                cal_trainable_variables[avg_length][avg_agent][current_var.name] = avg_trainable_variables[avg_length][var_name]
                                print("avg_var.name:", var_name)
                                print("avg_var:", avg_trainable_variables[avg_length][var_name])
                                var_new = tf.compat.v1.convert_to_tensor(avg_trainable_variables[avg_length][var_name])
                                var_assign = tf.compat.v1.assign(current_var, var_new)
                            else:
                                var_assign = tf.compat.v1.assign(current_var, current_var)
                            new_var_to_restore_list.append(var_assign)

                        print("count_of_var_to_restore:", len(new_var_to_restore_list))
                        for new_var_to_restore in new_var_to_restore_list:
                            sess.run(new_var_to_restore)

                        for i in range(len(tf.compat.v1.trainable_variables())):
                            # 再次取模型文件的变量数据 查看是否正确修改且赋值
                            current_var = tf.compat.v1.trainable_variables()[i]
                            cal_trainable_variables[avg_length][avg_agent][current_var.name] = current_var.eval()
                            var_name = current_var.name.split(":")[0]
                            var_value =current_var.eval()
                            print("after avg_var.name:", var_name)
                            print("after avg_var:", var_value)

                        print("cal_trainable_variables:", cal_trainable_variables[avg_length])

                        # 4.测试avg_auc并存入文件
                        evaluation_per_agent = test_avg_model(avg_agent, trainer, sess)
                        save_avg_model_auc_to_excel(evaluation_per_agent)
                    tf.compat.v1.reset_default_graph()




        # sorted_flag 按信心值大到小排序训练
        # sorted_flag_with_non_coo 按信息值排序+带不合作的agent，为true时必须带上 agent_training_non_coo
        # agent_training_non_coo 不合作的agent节点
        if options['transferred_training']:
            more_loop_count = 2  # 跑完合作的之后，再跑n轮不合作的
            #evaluation, batch_loss, memory_use = train_multi_agents_with_transfer(options,agent_names, agent_training_order)
            evaluation, batch_loss, memory_use, ho_count, ho_ratio = train_multi_agents_with_handover_query(options,
                                                                agent_names, agent_training_order, sorted_flag=True,
                                                                agent_training_non_coo=agent_training_non_coo,
                                                               sorted_flag_with_non_coo=False, more_loop_count=more_loop_count)
            save_result_to_excel(data_splitter, evaluation, batch_loss, memory_use, ho_count, ho_ratio)

    # 直接读取模型
    # Testing on test with best model
    else:
        logger.info("Skipping training")
        logger.info("Loading model from {}".format(options["model_load_dir"]))

