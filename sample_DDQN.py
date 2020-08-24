''' 
    Before run into source code, we recommend using provided Docker image  
    Provided and document: TODO: INSERT LINK TO THE DOCUMENT HERE 
''' 

# Import TF and Keras 
from keras.layers import Dense, Activation 
from keras.models import Sequential, load_model, model_from_json
from keras.optimizers import Adam 
from keras import backend as K
import tensorflow as tf
from random import random, randrange
import numpy as np 

class ReplayBuffer(object): 
    def __init__(
        self,
        max_size, # max size of buffer  
        input_shape, 
        n_actions, # The number of action 
        discrete = False
    ):
        self.mem_size = max_size
        self.mem_counter = 0 # Contain index of last element 
        self.discrete = discrete 
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))

        dtype = np.int8 if self.discrete else np.float32 # type setting for memory saving
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype = dtype) 
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype = np.float32)

    def store_transition(self, state, action, reward, state_next, done): 
        index = self.mem_counter % self.mem_size # TODO: not clear this line
        self.new_state_memory[index] = state_next
        if self.discrete: 
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0 
            self.action_memory[index] = actions  
        else: 
            self.action_memory[index] = action 
        self.reward_memory[index] = reward 
        self.terminal_memory[index] = 1 - int(done) # flag 
        self.mem_counter = self.mem_counter + 1

    def sample_buffer(self, batch_size): 
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]
        rewards = self.reward_memory[batch]

        return states, actions, rewards, new_states, terminal

def build_dqn(learning_rate, n_actions, input_dims, fc1_dims, fc2_dims): 
    u''' 
        fc1_dims : fully connected layers dimension
        fc2_dims : fully connected layers dimension
    '''
    model = Sequential([
        Dense(fc1_dims, input_shape(input_dims,)), 
        Activation('relu'), 
        Dense(fc2_dims), 
        Activation('relu'), 
        Dense(n_actions), 
    ])
    model.compile(optimizer=Adam(lr=learning_rate), loss='mse')
    return model


# Double Deep Q Network off-policy 
class DDQNAgent(object): 
    def __init__(
        self, 
        input_dims, # input dimension   
        n_actions, # the total numbere of actions 
        batch_size, # batch size for tranining experience replay  
        alpha = 0.00025, # learning rate
        gamma = 0.99, # discount factor
        epsilon = 1, # exploration factor for epsilon greedy policy  
        epsilon_decay = 0.999, 
        epsilon_min = 0.01, 
        fname = 'ddqn_model.h5', 
        mem_size = 10000, 
        replace_target = 100, # replace target network with online network 
    ): 
        self.n_actions = n_actions
        self.action_space = [i for i in range(self.n_actions)]
        self.batch_size = batch_size
        self.replace_target = replace_target
        self.model = fname 
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions, discrete=True)

        # Hyperparameters 
        self.gamma = gamma 
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Creating network for evaluation and target
        self.q_eval = build_dqn(alpha, n_actions, input_dims, 256, 256) 
        self.q_target = build_dqn(alpha, n_actions, input_dims, 256, 256) 

    def remember(self, state, action, reward, new_state, done): 
        self.memory.store_transition(state, action, reward, new_state, done) 

    # Action for each state with epsilon greedy policy
    def choose_act(self,state):
        state = state[np.newaxis,:]
        rand = np.random()
        if rand < self.epsilon: 
            action = np.random.choice(self.action_space)
        else: 
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)
        return action

    # TODO: NEED FURTHER UNDERSTANDING
    def learn(self): 
        if self.memory.mem_counter > self.batch_size: 
            state, action, reward, new_state, done = self.memory.sample_buffer(
                                                                self.batch_size) 
            action_values = np.array(self.action_space, dtype = np.int8)
            action_indices = np.dot(action, action_values)

            q_next = self.q_target.predict(new_state)
            q_eval = self.q_eval.predict(new_state)

            q_pred = self.q_eval.predict(state) 

            max_actions = np.argmax(q_eval, axis = 1)

            q_target = q_pred 
            batch_index = np.arange(self.batch_size, dtype = np.int32) 

            q_target[batch_index, action_indices] = reward + self.gamma*q_next[batch_index, max_actions.astype(int)]*done

            _ = self.q_evaluate.fit(state, q_target, verbose = 0)
            
            # Decay epsilon after iterations 
            self.epsilon = self.epsilon*self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon_min

            if self.memory.mem_counter % self.replace_target == 0: 
                self.update_network_parameters()


    def update_network_parameters(self): 
        self.q_target.model.set_weights(self.q_evaluate.model.get_weights())

    def save_model(self):
        self.q_evaluate.save_model(self.model_file)

    def load_model(self): 
        self.q_evaluate.load_model(self.model_file)

        if self.epsilon <= self.epsilon_min: 
            self.update_network_parameters()










