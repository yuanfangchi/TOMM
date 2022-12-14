import numpy as np
import tensorflow as tf
import logging
import sys


logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class Agent(object):

    def __init__(self, params):
        super(Agent, self).__init__()

        self.action_vocab_size = len(params['relation_vocab'])
        self.entity_vocab_size = len(params['entity_vocab'])
        self.embedding_size = params['embedding_size']
        self.hidden_size = params['hidden_size']
        self.ePAD = tf.constant(params['entity_vocab']['PAD'], dtype=tf.int32)
        self.rPAD = tf.constant(params['relation_vocab']['PAD'], dtype=tf.int32)
        if params['use_entity_embeddings']:
            self.entity_initializer = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
        else:
            self.entity_initializer = tf.compat.v1.zeros_initializer()
        self.train_entities = params['train_entity_embeddings']
        self.train_relations = params['train_relation_embeddings']

        self.path_length = params['path_length']
        self.num_rollouts = params['num_rollouts']
        self.test_rollouts = params['test_rollouts']
        self.LSTM_Layers = params['LSTM_layers']
        self.batch_size = params['batch_size']
        self.max_num_actions = params['max_num_actions']
        self.dummy_start_label = tf.constant(
            np.ones(self.batch_size * self.num_rollouts, dtype='int64') * params['relation_vocab']['DUMMY_START_RELATION'])

        self.entity_embedding_size = self.embedding_size
        self.use_entity_embeddings = params['use_entity_embeddings']
        if self.use_entity_embeddings:
            self.m = 4
        else:
            self.m = 2

        with tf.compat.v1.variable_scope("action_lookup_table", reuse=tf.compat.v1.AUTO_REUSE):
            self.action_embedding_placeholder = tf.compat.v1.placeholder(tf.float32,
                                                               [self.action_vocab_size, 2 * self.embedding_size])

            self.relation_lookup_table = tf.compat.v1.get_variable("relation_lookup_table",
                                                         shape=[self.action_vocab_size, 2 * self.embedding_size],
                                                         dtype=tf.float32,
                                                         initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                                                         trainable=self.train_relations)
            self.relation_embedding_init = self.relation_lookup_table.assign(self.action_embedding_placeholder)

        with tf.compat.v1.variable_scope("entity_lookup_table", reuse=tf.compat.v1.AUTO_REUSE):
            self.entity_embedding_placeholder = tf.compat.v1.placeholder(tf.float32,
                                                               [self.entity_vocab_size, 2 * self.embedding_size])
            self.entity_lookup_table = tf.compat.v1.get_variable("entity_lookup_table",
                                                       shape=[self.entity_vocab_size, 2 * self.entity_embedding_size],
                                                       dtype=tf.float32,
                                                       initializer=self.entity_initializer,
                                                       trainable=self.train_entities)
            self.entity_embedding_init = self.entity_lookup_table.assign(self.entity_embedding_placeholder)

        with tf.compat.v1.variable_scope("global_policy_step", reuse=tf.compat.v1.AUTO_REUSE):
            cells = []
            for _ in range(self.LSTM_Layers):
                cells.append(tf.compat.v1.nn.rnn_cell.LSTMCell(self.m * self.hidden_size, use_peepholes=True,
                                                               state_is_tuple=True))
            self.global_rnn = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

        self.global_hidden_layer = tf.keras.layers.Dense(4 * self.hidden_size, activation='relu')
        self.global_output_layer = tf.keras.layers.Dense(self.m * self.embedding_size, activation='relu')

    def get_mem_shape(self):
        return (self.LSTM_Layers, 2, None, self.m * self.hidden_size)

    def action_encoder(self, next_relations, next_entities):
        with tf.compat.v1.variable_scope("lookup_table_edge_encoder", reuse=tf.compat.v1.AUTO_REUSE):
            relation_embedding = tf.nn.embedding_lookup(params=self.relation_lookup_table, ids=next_relations)
            entity_embedding = tf.nn.embedding_lookup(params=self.entity_lookup_table, ids=next_entities)
            if self.use_entity_embeddings:
                action_embedding = tf.concat([relation_embedding, entity_embedding], axis=-1)
            else:
                action_embedding = relation_embedding
        return action_embedding

    def step(self, next_relations, next_entities, prev_state, prev_relation, query_embedding, current_entities, range_arr):

        prev_action_embedding = self.action_encoder(prev_relation, current_entities)

        # 1. one step of rnn

        output, new_state = self.global_rnn(prev_action_embedding, prev_state)
        # output: [B, 4D]


        # Get state vector
        prev_entity = tf.nn.embedding_lookup(params=self.entity_lookup_table, ids=current_entities)
        if self.use_entity_embeddings:
            state = tf.concat([output, prev_entity], axis=-1)
        else:
            state = output

        candidate_action_embeddings = self.action_encoder(next_relations, next_entities)
        state_query_concat = tf.concat([state, query_embedding], axis=-1)

        # MLP for policy#
        hidden_layer_output = self.global_hidden_layer(state_query_concat)
        output = self.global_output_layer(hidden_layer_output)

        output_expanded = tf.expand_dims(output, axis=1)  # [B, 1, 2D]
        prelim_scores = tf.reduce_sum(input_tensor=tf.multiply(candidate_action_embeddings, output_expanded), axis=2)

        # Masking PAD actions

        comparison_tensor = tf.ones_like(next_relations, dtype=tf.int32) * self.rPAD  # matrix to compare
        mask = tf.equal(next_relations, comparison_tensor)  # The mask
        dummy_scores = tf.ones_like(prelim_scores) * -99999.0  # the base matrix to choose from if dummy relation
        scores = tf.compat.v1.where(mask, dummy_scores, prelim_scores)  # [B, MAX_NUM_ACTIONS]

        # 4 sample action
        action = tf.cast(tf.random.categorical(logits=scores, num_samples=1), dtype=tf.int32)  # [B, 1]

        # loss
        # 5a.
        label_action =  tf.squeeze(action, axis=1)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=label_action)  # [B,]

        # 6. Map back to true id
        action_idx = tf.squeeze(action)
        chosen_relation = tf.gather_nd(next_relations, tf.transpose(a=tf.stack([range_arr, action_idx])))
        return loss, new_state, tf.nn.log_softmax(scores), output, action_idx, chosen_relation

    def __call__(self, candidate_relation_sequence, candidate_entity_sequence, current_entities, query_relation, range_arr, T=3):
        self.baseline_inputs = []
        # get the query vector
        query_embedding = tf.nn.embedding_lookup(params=self.relation_lookup_table, ids=query_relation)  # [B, 2D]
        state = self.global_rnn.zero_state(batch_size=self.batch_size * self.num_rollouts, dtype=tf.float32)

        prev_relation = self.dummy_start_label

        all_loss = []  # list of loss tensors each [B,]
        all_logits = []  # list of actions each [B,]
        action_idx = []
        rnn_state = []
        chosen_relations = []
        rnn_output = []

        with tf.compat.v1.variable_scope("global_policy_steps_unroll", reuse=tf.compat.v1.AUTO_REUSE) as scope:
            for t in range(T):
                if t > 0:
                    scope.reuse_variables()
                next_possible_relations = candidate_relation_sequence[t]  # [B, MAX_NUM_ACTIONS, MAX_EDGE_LENGTH]
                next_possible_entities = candidate_entity_sequence[t]
                current_entities_t = current_entities[t]

                loss, state, logits, output, idx, chosen_relation = self.step(next_possible_relations,
                                                                              next_possible_entities,
                                                                              state, prev_relation,
                                                                              query_embedding,
                                                                              current_entities_t,
                                                                              range_arr=range_arr)

                all_loss.append(loss)
                all_logits.append(logits)
                action_idx.append(idx)
                rnn_state.append(state)
                prev_relation = chosen_relation
                chosen_relations.append(chosen_relation)
                rnn_output.append(output)
            # [(B, T), 4D]

        return all_loss, all_logits, action_idx, rnn_state, rnn_output, chosen_relations
