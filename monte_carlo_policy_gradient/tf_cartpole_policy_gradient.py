from __future__ import print_function

import gym
import tensorflow as tf
import numpy as np
import matplotlib as mlp
mlp.use('Agg')
import matplotlib.pyplot as plt

gamma = 0.99

class ActorNetwork(object):
    
    def __init__(self, sess, state_dim, action_dim, learning_rate):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        
        # Actor Network
        self.inputs, self.out = self.create_actor_network()
        
        self.network_params = tf.trainable_variables()
        
        # This Advantage will be provided by the Discount Reward
        self.returns = tf.placeholder("float", [None,1])
        self.actions = tf.placeholder("float", [None,self.a_dim])
        
        action_probabilities = tf.reduce_sum(tf.mul(self.out, self.actions), reduction_indices=[1])
        eligibility = tf.log(action_probabilities) * self.returns        # todo add gradient, apply_gradient
        self.policy_loss = -tf.reduce_sum(eligibility)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.policy_loss)        
        
    def create_actor_network(self):
        inputs = tf.placeholder("float", [None, self.s_dim])        
        w = tf.get_variable("w1", [self.s_dim, self.a_dim])
        out = tf.nn.softmax(tf.matmul(inputs, w))
        
        return inputs, out
        
    def train(self, inputs, actions, q_value):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.actions: actions,
            self.returns: q_value
        })
        
    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })
   
    @property
    def loss():
        return self.sess.run(self.policy_loss, feed_dict={
            self.inputs: inputs,
            self.returns: q_value
        })
        
        
def get_discount_rewards(transitions):
    discounted_r = [np.zeros([1]) for _ in range(len(transitions))]
    running_add = 0
    #print(len(transitions))
    for t in reversed(range(0, len(transitions))):
        running_add = running_add * gamma + transitions[t][2]
        discounted_r[t][0] = running_add
        
    return discounted_r    
        
def train(sess, env, actor) :
    sess.run(tf.global_variables_initializer())    
    epoch = 10000
    plt.ioff()
    fig = plt.figure()
    reward_list = []
    for episode_num in range(epoch):
    
        observation = env.reset()
        states = []
        actions = []
        transitions = []
        reward_sum = 0        
        for _ in range(1000):
            obs_vector = np.expand_dims(observation, axis=0)
            act_prob = actor.predict(obs_vector)
            action = 0 if np.random.uniform() < act_prob[0][0] else 1
            
            states.append(observation)
            actionblank = np.zeros(2)
            actionblank[action] = 1
            actions.append(actionblank)
            
            # take the acion in the environment
            old_observation = observation
            observation, reward, done, info = env.step(action)
            transitions.append((old_observation, action, reward))
            reward_sum += reward
            
            if done:
                print("episode : " + str(episode_num) + ", reward : " + str(reward_sum))
                reward_list.append(reward_sum)
                
                break
                
        future_reward = get_discount_rewards(transitions)
        
        advantages = future_reward - np.mean(future_reward)
        advantages /= np.std(future_reward)        
        
        actor.train(states, actions, advantages)
        
    plt.plot(reward_list)
    plt.savefig('myfig.png')
    plt.close()

         
            
        
    
def main(_):
    with tf.Session() as sess:
    
        env = gym.make('CartPole-v0')
        # todo add random seed
        
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        print(action_dim)
        # todo add action bound
        
        actor = ActorNetwork(sess, state_dim, action_dim, 0.01)
        
        train(sess, env, actor)
    
if __name__ == '__main__':
    tf.app.run()
    
    