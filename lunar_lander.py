from keras import backend as K
from keras.layers import Activation, Dense, Input
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
from gym import wrappers
import gym
import tensorflow.compat.v1 as tf

sess = tf.Session(config=tf.ConfigProto(
  intra_op_parallelism_threads=24))

class Agent(object):
    def __init__(self, alpha, beta, gamma=0.99, n_actions=4, layer1_size=1024, layer2_size=512, input_dims=8):
        
        self.gamma=gamma
        self.alpha=alpha
        self.beta=beta
        self.input_dims = input_dims
        self.fc1_dims = layer1_size
        self.fc2_dims = layer2_size
        self.n_actions = n_actions
        self.actor, self.critic, self.policy = self.build_actor_critic_network()
        self.action_space = [i for i in range(self.n_actions)]
        
    def build_actor_critic_network(self):
        input = Input(shape=(self.input_dims,))
        delta = Input(shape=[1])
        dense1 = Dense(self.fc1_dims, activation='relu')(input)
        dense2 = Dense(self.fc2_dims, activation='relu')(dense1)
        probs = Dense(self.n_actions, activation='softmax')(dense2)
        values = Dense(1, activation='linear')(dense2)
        
        def custom_loss(y_true, y_pred):
            out = K.clip(y_pred, 1e-8, 1-1e-8)
            log_lik = y_true * K.log(out)
            
            return K.sum(-log_lik*delta)
        
            
        actor = Model([input, delta], [probs])
        actor.compile(optimizer=Adam(lr=self.alpha), loss=custom_loss)
        critic = Model([input], [values])
        critic.compile(optimizer=Adam(lr=self.beta), loss='mean_squared_error')
        policy = Model([input], [probs])
        return actor, critic, policy
    
    def choose_action(self, observation):
        state = observation[np.newaxis, :]
        probabilities = self.policy.predict(state)[0]
        action = np.random.choice(self.action_space, p = probabilities)
        return action
    
    def learn(self, state, action, reward, state_, done):
        state = state[np.newaxis, :]
        state_ = state_[np.newaxis, :]
        
        critic_value_ = self.critic.predict(state_)
        critic_value = self.critic.predict(state)
        
        target = reward + self.gamma * critic_value_*(1 - int(done))
        delta = target - critic_value
        actions = np.zeros([1, self.n_actions])
        actions[np.arrange(1), action] = 1.0
        
        self.actor.fit([state, delta], actions, verbose=0)
        self.critic.fit(state, target, verbose=0)
	
if __name__ == '__main__':
    agent = Agent(alpha=0.00001, beta = 0.00005)
    
    env = gym.make('LunarLander-v2')
    
    score_history = []
    num_episodes = 40000
    
    for i in range(num_episodes):
        done = False
        score = 0
        observation = env.reset()
        step = 0
        
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            observation = observation_
            score += reward
            env.render()
        
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        print('episode', i, 'score %.2f average score %.2f' % (score, avg_score))
    
    print('done')
	
    env.close()