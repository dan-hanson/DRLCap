# -*- coding: utf-8 -*-
# Modified 2025-03-27 for GTX 1650 compatibility
# Power capping agent using DDQN + Prioritized Replay
# GPU: GTX 1650
# Power limit range: 35W to 75W
# Note: GTX 1650 does NOT support frequency locking

import math
import os
import tensorflow as tf
import numpy as np

# Enable persistent mode (keeps settings between runs)
os.system("sudo /usr/bin/nvidia-smi -pm 1")

# Set default power cap (GTX 1650 max = 75W)
gpu_limit = 75
os.system("sudo /usr/bin/nvidia-smi -pl %s" % gpu_limit)

# -----------------------------
# RL State Variables:
# (1) GPU Utilization
# (2) Memory Utilization
# (3) Power Draw
# (4) Temperature
# (5) Power Cap (optional toggle)
# -----------------------------
GPU_LABELS = ('UTIL_GPU', 'UTIL_MEM', 'POWER', 'TEMP')

# Normalize input state values
MINS = { 'UTIL_GPU': 0, 'UTIL_MEM': 0, 'POWER': 30, 'TEMP': 30 }
MAXS = { 'UTIL_GPU': 100, 'UTIL_MEM': 100, 'POWER': 75, 'TEMP': 90 }

# Discretization buckets per feature
BUCKETS = { 'UTIL_GPU': 20, 'UTIL_MEM': 20, 'POWER': 20, 'TEMP': 30 }
gpu_num_buckets = np.array([BUCKETS[k] for k in GPU_LABELS], dtype=np.double)

# -----------------------------
# GPU Clock is not controllable on 1650,
# but original code uses it in state space.
# We'll disable it but leave the structure.
# -----------------------------
CLOCKS_GPU = []
clock_gpu_bucket = {}
POWER_IN_STATE = 0  # Whether to include power cap as an RL state (disabled)

# -----------------------------
# Action Space: Power limits between 35W and 75W
# -----------------------------
GPU = range(35, 76, 1)
gpu_to_bucket = {GPU[i]: i for i in range(len(GPU))}

# -----------------------------
# RL Configurations
# -----------------------------
N_FEATURES = 4  # Only using 4 features now (excluding clock/power cap)
INITIAL_EPSILON = 0.3
FINAL_EPSILON = 0.1
LEARNING_RATE = 0.0001
REWARD_DECAY = 0.95
REPLACE_TARGET_ITER = 10
MEMORY_SIZE = 1000
BATCH_SIZE = 32
STEPS = 2000



# %% RL STATE FUNCTION — Collect GPU Telemetry and Convert to Discrete RL Input

def state():
    # Query GPU telemetry and write to CSV
    os.system(
        'nvidia-smi --format=csv,noheader,nounits --filename=state.csv '
        '--query-gpu=power.draw,clocks.current.graphics,utilization.gpu,utilization.memory,temperature.gpu')
    
    # Sleep to allow telemetry query to finish
    os.system('sleep 0.3')
    
    # Read the single-line CSV output
    with open('state.csv', 'r') as fo:
        states_lines = fo.readlines()
        for states in states_lines:
            states = states.replace(',', '')  # remove CSV commas
            power_gpu      = float(states.split()[0])
            clock_gpu      = float(states.split()[1])
            util_gpu       = float(states.split()[2])
            util_memory    = float(states.split()[3])
            temp           = float(states.split()[4])

    # Compose telemetry dictionary
    stats = {
        'GPUL': gpu_limit,               # Power cap
        'CLOCKS_GPU': clock_gpu,
        'UTIL_GPU': util_gpu,
        'UTIL_MEM': util_memory,
        'POWER': power_gpu,
        'TEMP': temp
    }

    print("Raw stats:", stats)

    # ---------------------------------------
    # Normalize + Bucketize the 4 main features
    # ---------------------------------------
    gpu_all_mins     = np.array([MINS[k] for k in GPU_LABELS], dtype=np.double)
    gpu_all_maxs     = np.array([MAXS[k] for k in GPU_LABELS], dtype=np.double)
    gpu_num_buckets  = np.array([BUCKETS[k] for k in GPU_LABELS], dtype=np.double)

    # Bucket width = range / number of buckets
    gpu_widths = np.divide(gpu_all_maxs - gpu_all_mins, gpu_num_buckets)

    # Collect current raw values
    gpu_raw_values = [stats[k] for k in GPU_LABELS]
    gpu_clipped    = np.clip(gpu_raw_values, gpu_all_mins, gpu_all_maxs)
    
    # Convert to relative position within bucket range
    gpu_floored    = gpu_clipped - gpu_all_mins
    gpu_state      = np.divide(gpu_floored, gpu_widths)
    gpu_state      = np.clip(gpu_state, 0, gpu_num_buckets - 1)

    # Optional: Add clock frequency bucket index (currently unused on GTX 1650)
    if CLOCKS_GPU:
        gpu_state = np.append(gpu_state, [clock_gpu_bucket.get(stats['CLOCKS_GPU'], 0)])

    # Optional: Add power cap index to state vector
    if POWER_IN_STATE:
        gpu_state = np.append(gpu_state, [gpu_to_bucket[stats['GPUL']]])

    # Convert all floats to integers (bucket IDs)
    gpu_state = [int(x) for x in gpu_state]

    print("Discrete gpu_state:", gpu_state)

    return gpu_state, stats



# %% DDQN Prioritized Replay - SumTree Implementation
np.random.seed(1)
tf.set_random_seed(1)
print(tf.__version__)


# Deep Q Network off-policy
class SumTree(object):
    """
    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:  # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:  # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1  # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


# %% DDQN Prioritized Experience Replay Buffer
class Memory(object):
    """
    Stores transitions as (s, a, r, s') in a SumTree.
    Provides sampling based on TD error priority.
    """

    # Priority hyperparameters
    epsilon = 0.01       # Small constant to avoid zero priority
    alpha = 0.6          # [0 ~ 1]: How much TD-error affects priority (0 = uniform, 1 = full prioritization)
    beta = 0.4           # Importance-sampling bias correction (increases over time)
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.0  # Clip max TD-error for stability

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        """
        Store a new experience with the current max priority.
        """
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)

    def sample(self, n):
        """
        Sample a batch of experiences, using priority-based stratified sampling.
        Returns indices, memory batch, and importance-sampling (IS) weights.
        """
        b_idx = np.empty((n,), dtype=np.int32)
        b_memory = np.empty((n, self.tree.data[0].size))
        ISWeights = np.empty((n, 1))

        # Segment total priority into n regions
        pri_seg = self.tree.total_p / n

        # Gradually increase bias correction over time
        self.beta = np.min([1.0, self.beta + self.beta_increment_per_sampling])

        # Find smallest probability (for normalization)
        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p

        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)

            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)

            b_idx[i] = idx
            b_memory[i, :] = data

        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        """
        Update priorities in the tree using new TD-errors.
        """
        abs_errors += self.epsilon  # Avoid 0 probability
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


# %% DDQN Agent with Prioritized Replay
class DQNPrioritizedReplay:
    def __init__(
            self,
            n_actions,
            n_features=N_FEATURES,
            learning_rate=LEARNING_RATE,
            reward_decay=REWARD_DECAY,
            e_greedy=INITIAL_EPSILON,
            replace_target_iter=REPLACE_TARGET_ITER,
            memory_size=MEMORY_SIZE,
            batch_size=BATCH_SIZE,
            # e_greedy_increment=E_GREEDY_INCREMENT,
            output_graph=False,
            prioritized=True,
            sess=tf.Session(),
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon = e_greedy

        self.prioritized = prioritized  # decide to use double q or not

        self.learn_step_counter = 0

        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        if self.prioritized:
            self.memory = Memory(capacity=memory_size)
        else:
            self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        # start 12/7/21
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        # end 12/7/21
        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.cost_his = []

    def _build_net(self):
        # Shared layer-building helper
        def build_layers(s, c_names, n_l1, w_initializer, b_initializer, trainable):
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names,
                                     trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names,
                                     trainable=trainable)
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1)  # hidden layers
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names,
                                     trainable=trainable)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names,
                                     trainable=trainable)
                out = tf.matmul(l1, w2) + b2  # Q Value layer
            return out

        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        if self.prioritized:
            self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')
        # Evaluation network
        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 20, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers
            self.q_eval = build_layers(self.s, c_names, n_l1, w_initializer, b_initializer, True)

        # Loss and optimizer
        with tf.variable_scope('loss'):
            if self.prioritized:
                self.abs_errors = tf.reduce_sum(tf.abs(self.q_target - self.q_eval), axis=1)  # for updating Sumtree
                self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.q_target, self.q_eval))
            else:
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = build_layers(self.s_, c_names, n_l1, w_initializer, b_initializer, False)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

    def store_transition(self, s, a, r, s_):
        if self.prioritized:  # prioritized replay
            transition = np.hstack((s, [a, r], s_))
            self.memory.store(transition)  # have high priority for newly arrived transition
        else:  # random replay
            if not hasattr(self, 'memory_counter'):
                self.memory_counter = 0
            transition = np.hstack((s, [a, r], s_))
            index = self.memory_counter % self.memory_size
            self.memory[index, :] = transition
            self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        print("epsilon:", self.epsilon)
        if np.random.uniform() > self.epsilon:
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)

            print("max!!!!!!!!!!!!!!!!!!!!!")
        else:
            action = np.random.randint(0, self.n_actions)
            print("random!!!!!!!!!!!!!!!!!!!!!")
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        if self.prioritized:
            tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
            batch_memory = self.memory[sample_index, :]

        #  DQN
        # q_next, q_eval = self.sess.run(
        #         [self.q_next, self.q_eval],
        #         feed_dict={self.s_: batch_memory[:, -self.n_features:],
        #                    self.s: batch_memory[:, :self.n_features]})

        # DDQN wym
        q_next, q_eval4next = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={self.s_: batch_memory[:, -self.n_features:],  # next observation
                       self.s: batch_memory[:, -self.n_features:]})  # next observation
        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_features]})

        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        # DQN
        # q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        # DDQN wym
        max_act4next = np.argmax(q_eval4next, axis=1)  # the action that brings the highest value is evaluated by q_eval
        selected_q_next = q_next[batch_index, max_act4next]
        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        if self.prioritized:
            _, abs_errors, self.cost = self.sess.run([self._train_op, self.abs_errors, self.loss],
                                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                                self.q_target: q_target,
                                                                self.ISWeights: ISWeights})
            self.memory.batch_update(tree_idx, abs_errors)  # update priority
        else:
            _, self.cost = self.sess.run([self._train_op, self.loss],
                                         feed_dict={self.s: batch_memory[:, :self.n_features],
                                                    self.q_target: q_target})

        self.cost_his.append(self.cost)

        # self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / (STEPS - MEMORY_SIZE)
        self.learn_step_counter += 1


sess = tf.Session()

with tf.variable_scope('DQN_with_prioritized_replay'):
    RL = DQNPrioritizedReplay(
        n_actions=len(GPU)
    )
# Get initial observation and state
obeservation, states = state()

for step in range(STEPS):
    action = RL.choose_action(np.array(obeservation))
    gpu_limit = GPU[action]
    print('power cap:', gpu_limit)

    os.system(f"sudo nvidia-smi -pl {gpu_limit}")  # apply power limit

    power_g = math.floor(states['POWER'])
    fre_g = states['CLOCKS_GPU']

    # Get next observation and calculate reward
    obeservation_, states_ = state()
    power_g_ = math.floor(states_['POWER'])
    fre_g_ = states_['CLOCKS_GPU']

    power = power_g_ - power_g
    fre = fre_g_ - fre_g
    print('power:', power, 'freq delta:', fre)

    # Reward logic (unchanged)
    if power <= 0:
        if -45 <= fre:
            reward = 5
        elif -90 <= fre < -45:
            reward = -1
        elif -135 <= fre < -90:
            reward = -2
        else:
            reward = -3
    else:
        if fre <= 45:
            reward = -1
        elif 45 < fre <= 90:
            reward = -2
        elif 90 < fre <= 135:
            reward = -3
        else:
            reward = -4

    print('reward:', reward)

    # Store and learn
    RL.store_transition(obeservation, action, reward, obeservation_)

    if step > MEMORY_SIZE:
        RL.learn()

    # Update current state
    states = states_
    obeservation = obeservation_

    print('step-----------------:', step)
    print()

# Save model
print('Uninitialized variables:')
print(sess.run(tf.report_uninitialized_variables()))

saver = tf.train.Saver()
save_path = saver.save(sess, './results/checkpoints/save_net.ckpt')
print('Saved to:', save_path)

model_path = saver.save(sess, './results/models/final_model')
print('Saved model to:', model_path)

# Reset power cap to default
os.system("sudo nvidia-smi -pl 75")  # default for 1650
print('end')
