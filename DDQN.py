''' 
    Before run into source code, we recommend using provided Docker image  
    Provided and document: 
''' 

# Import provided source code RLComp2020 environment
# import sys
# import DQNModel
# import Memory

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
        action_space, # The number of action 
        discrete = False
    ):
        self.mem_size = max_size
        self.mem_counter = 0 # Contain index of last element 
        self.discrete = discrete 
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))

        dtype = np.int8 if self.discrete else np.float32 # type setting for memory saving
        self.action_memory = np.zeros((self.mem_size, action_space), dtype = dtype) 
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
        states = self.action_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]
        rewards = self.reward_memory[batch]

        return states, actions, rewards, new_states, terminal


# Build DQN model 
# build DQN function was taken and rename from DQNModel.py -> DQN.create_model()
def build_dqn(learning_rate, input_dim, action_space): 
    model = Sequential()
    model.add(Dense(300, input_dim = input_dim))
    model.add(Activation('relu'))
    model.add(Dense(300))
    model.add(Activation('relu'))
    model.add(Dense(action_space)) 
    model.add(Activation('linear')) 
    adam = optimizers.adam(lr=learning_rate)
    model.compile(optimizer = adam, loss = 'mse')
    return model


# Double Deep Q Network off-policy 
class DDQNAction(object): 
    def __init__(
        self,
        batch_size, # Batch size for experience replay 
        input_dim, #The number of inputs for the DQN network
        action_space, #The number of actions for the DQN network
        fname = 'ddqn_model.h5', 
        # DDQN model hyper parameter
        epsilon = 1, # Epsilon - the exploration factor
        epsilon_min = 0.01, # The minimum epsilon 
        epsilon_decay = 0.999,# The decay epislon for each update_epsilon time
        learning_rate = 0.00025, # The learning rate for the DQN network
        gamma = 0.99, # The discount factor
        replace_target = 100, # Replace the target network with current network for every 100 steps

        # target_model = None, #The DQN target model 
        # sess=None,
    ):
        self.input_dim = input_dim 
        self.action_space = action_space 
        self.batch_size = batch_size
        self.gamma = gamma 
        self.epsilon = epsilon 
        self.epsilon_min = epsilon_min 
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.replace_target = replace_target
        self.model_file = model_file

        # Creating networks for evaluate and target
        self.q_target = build_model(self.learning_rate, self.input_dim, self.action_space)
        self.q_evaluate = build_model(self.learning_rate, self.input_dim, self.action_space)

        #Tensorflow GPU optimization
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config)
        K.set_session(sess)
        self.sess.run( tf.compat.v1.global_variables_initializer()) 
 

    # Action for each state with epsilon greedy policy
    def act(self,state):
        # Get the index of the maximum Q values      
        a_max = np.argmax(self.q_evaluate.predict(state.reshape(1,len(state))))   
        if (random() < self.epsilon):
            a_chosen = randrange(self.action_space)
        else:
            a_chosen = a_max      
        return a_chosen

    # # TODO: modified to fit DDQN
    # def replay(self, samples, batch_size): 
    #     inputs = np.zeros((batch_size, self.input_dim))
    #     targets = np.zeros((batch_size, self.input_dim))

    #     for i in range(0,batch_size):
    #         state = samples[0][i,:]
    #         action = samples[1][i]
    #         reward = samples[2][i]
    #         new_state = samples[3][i,:]
    #         done= samples[4][i]
            
    #         inputs[i,:] = state
    #         targets[i,:] = self.target_model.predict(state.reshape(1,len(state)))        
    #         # if done:
    #         #     targets[i,action] = reward # if terminated, only equals reward
    #         # else:
    #         #     Q_future = np.max(self.target_model.predict(new_state.reshape(1,len(new_state))))
    #         #     targets[i,action] = reward + Q_future * self.gamma
    #     #Training
    #     loss = self.model.train_on_batch(inputs, targets)  

    # TODO: working on learn function 
    # def learn(self): 

    def update_network_parameters(self): 
        self.q_target.model.set_weights(self.q_evaluate.model.get_weights())

    def save_model(self):
        self.q_evaluate.save_model(self.model_file)

    def load_model(self): 
        self.q_evaluate.load_model(self.model_file)

        if self.epsilon <= self.epsilon_min: 
            self.update_network_parameters










