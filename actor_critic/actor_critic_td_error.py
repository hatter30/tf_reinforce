from __future__ import print_function

import gym
import tensorflow as tf
import numpy as np
import matplotlib as mlp
mlp.use('Agg')
import matplotlib.pyplot as plt

import utils

gamma = 0.99

class ActorNetwork(object):
    
    def __init__(self, sess, state_dim, action_dim, learning_rate):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        
        # Actor Network
        self.inputs, self.out = self.create_actor_network()
        network_params = tf.trainable_variables()
        
        # This returns will be provided by the Discount Reward
        self._target = tf.placeholder("float", [None,1], name='returns')
        self.actions = tf.placeholder("float", [None,self.a_dim], name='actions')

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.action_prob = tf.reduce_sum(self.actions * self.out, reduction_indices=1)
        self.loss = -tf.log(self.action_prob) * self._target
        #self.optimize = optimizer.minimize(self.loss)
        grads_and_vars = optimizer.compute_gradients(self.loss, network_params)
        self.optimize = optimizer.apply_gradients(grads_and_vars)
        
    def create_actor_network(self):
        h_dim = 10
        inputs = tf.placeholder("float", [None, self.s_dim])        
        w1 = tf.get_variable("p_w1", [self.s_dim, h_dim], initializer=tf.contrib.layers.xavier_initializer())
        h1 = tf.nn.relu(tf.matmul(inputs, w1))
        
        w2 = tf.get_variable("p_w2", [h_dim, self.a_dim], initializer=tf.contrib.layers.xavier_initializer())
        out = tf.nn.softmax(tf.matmul(h1, w2))
        
        return inputs, out
        
    def train(self, inputs, actions, target):
        feed = {self.inputs: np.vstack(inputs), self.actions: np.vstack(actions), self._target: np.vstack(target)}
        _, output2 = self.sess.run([self.optimize, self.action_prob], feed)
        
    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })
        
class CriticNetwork(object):
    
    def __init__(self, sess, state_dim, learning_rate):
        self.sess = sess
        self.s_dim = state_dim
        self.learning_rate = learning_rate
        
        # Critic Network
        self.inputs, self._out = self.create_actor_network()
        
        self._target = tf.placeholder("float", [None,1], name='targets')

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self._loss = tf.nn.l2_loss(self._out - self._target)
        self.optimize = optimizer.minimize(self._loss)
        
    def create_actor_network(self):
        inputs = tf.placeholder("float", [None, self.s_dim])
        w1 = tf.get_variable("c_w1", [4, 10])
        b1 = tf.get_variable("c_b1", [10])
        h1 = tf.nn.relu(tf.matmul(inputs, w1) + b1)
        w2 = tf.get_variable("c_w2", [10, 1])
        b2 = tf.get_variable("c_b2", [1])
        calculated = tf.matmul(h1, w2) + b2
        return inputs, calculated
        
    def train(self, inputs, target):
        feed = {self.inputs: np.vstack(inputs), self._target: np.vstack(target)}
        _, loss = self.sess.run([self.optimize, self._loss], feed)
        return loss
        
    def predict(self, inputs):
        return self.sess.run(self._out, feed_dict={
            self.inputs: inputs
        })
        
def train(sess, env, actor, critic) :
    sess.run(tf.global_variables_initializer())    
    epoch = 3000
    plt.ioff()
    fig = plt.figure()
    reward_list = []
    running_reward = None
    for episode_num in range(epoch):
    
        observation = env.reset()
        states = []
        next_states = []
        actions = []
        rewards = []
        reward_sum = 0        
        for _ in range(1000):
            obs_vector = np.expand_dims(observation, axis=0)
            act_prob = actor.predict(obs_vector)
            act_prob = act_prob[0, :]
            action = np.random.choice(env.action_space.n, p=act_prob)          
            actionblank = np.zeros_like(act_prob)
            actionblank[action] = 1
            
            # take the acion in the environment
            old_observation = observation
            observation, reward, done, info = env.step(action)
            reward_sum += reward
            
            states.append(old_observation); actions.append(actionblank); rewards.append(reward)
            next_states.append(observation)
            
            if done:
                # update running reward
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                
                print("episode : %4d" % episode_num + ", reward %4d: " % reward_sum + ", reward mean: %4.3f" % running_reward)
                reward_list.append(reward_sum)                
                break       

        if running_reward >= 180:
            print("CartPole solved!!!!!")
            break
            
        value_next = critic.predict(next_states)
        td_target = np.vstack(rewards) + 0.99 * value_next
        td_error = td_target - critic.predict(states)
            
        loss = critic.train(states, td_target)
        actor.train(states, actions, td_error)
        
    plt.plot(reward_list)
    plt.savefig('myfig.png')
    plt.close()

def main(_):
    with tf.Session() as sess:
    
        env = gym.make('CartPole-v0')
        
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        actor = ActorNetwork(sess, state_dim, action_dim, 0.01)
        critic = CriticNetwork(sess, state_dim, 0.01)
        
        train(sess, env, actor, critic)
    
if __name__ == '__main__':
    tf.app.run()
    
    