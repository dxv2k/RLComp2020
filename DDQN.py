''' 
    Before run into source code, we recommend using provided Docker image  
    Provided and document: 
''' 

# Import provided source code environment
import sys
import DQNModel
import Memory

from keras.layers import Dense, Activation 
from keras.models import Sequential, load_model
from keras.optimizers import Adam 
import numpy as np 

print('Import libraries successfully')

# Build DQN
# build DQN function was taken from DQNModel.py -> DQN.create_model()
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


# Double DQN 
class DDQN: 
    def __init__(
        self,
        input_dim, #The number of inputs for the DQN network
        action_space, #The number of actions for the DQN network
        gamma = 0.99, #The discount factor
        epsilon = 1, #Epsilon - the exploration factor
        epsilon_min = 0.01, #The minimum epsilon 
        epsilon_decay = 0.999,#The decay epislon for each update_epsilon time
        learning_rate = 0.00025, #The learning rate for the DQN network
        replace_target = 100, # Replace the target network with current network for every 100 steps
        # model = None, #The DQN model
        # target_model = None, #The DQN target model 
        # sess=None,
    ):
        self.input_dim = input_dim 
        self.action_space = action_space 
        self.gamma = gamma 
        self.epsilon = epsilon 
        self.epsilon_min = epsilon_min 
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.replace_target = replace_target

    # Creating networks for evaluate and target
    self.q_target = build_model(self.learning_rate, self.input_dim, self.action_space)
    self.q_evaluate = build_model(self.learning_rate, self.input_dim, self.action_space)

    # Tensorflow GPU optimization 


    # Action for each state with epsilon greedy policy
    def act(self,state):
      # Get the index of the maximum Q values      
      a_max = np.argmax(self.q_evaluate.predict(state.reshape(1,len(state))))   
      if (random() < self.epsilon):
        a_chosen = randrange(self.action_space)
      else:
        a_chosen = a_max      
      return a_chosen


    # TODO: working on learn function 
    # def learn(self): 







