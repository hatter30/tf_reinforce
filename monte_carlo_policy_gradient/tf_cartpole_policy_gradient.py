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
        
        # This returns will be provided by the Discount Reward
        self.returns = tf.placeholder("float", [None,1], name='returns')
        self.actions = tf.placeholder("float", [None,self.a_dim], name='actions')
        
        # tf reward processing
        self.tf_discounted_epr = self.tf_discount_rewards(self.returns)
        tf_mean, tf_variance= tf.nn.moments(self.tf_discounted_epr, [0], shift=None, name="reward_moments")
        self.tf_discounted_epr -= tf_mean
        self.tf_discounted_epr /= tf.sqrt(tf_variance + 1e-6)
        
        self.loss = tf.nn.l2_loss(self.actions-self.out)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        grads = optimizer.compute_gradients(self.loss, var_list=tf.trainable_variables(), grad_loss=self.tf_discounted_epr)
        self.optimize = optimizer.apply_gradients(grads)      
        
    def create_actor_network(self):
        h_dim = 10
        inputs = tf.placeholder("float", [None, self.s_dim])        
        w1 = tf.get_variable("w1", [self.s_dim, h_dim], initializer=tf.contrib.layers.xavier_initializer())
        h1 = tf.nn.relu(tf.matmul(inputs, w1))
        
        w2 = tf.get_variable("w2", [h_dim, self.a_dim], initializer=tf.contrib.layers.xavier_initializer())
        out = tf.nn.softmax(tf.matmul(h1, w2))
        
        return inputs, out
        
    def train(self, inputs, actions, returns):
        feed = {self.inputs: np.vstack(inputs), self.actions: np.vstack(actions), self.returns: np.vstack(returns)}        
        _, output2 = self.sess.run([self.optimize, self.tf_discounted_epr], feed)
        
    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })
        
    def tf_discount_rewards(self, tf_r):
        discount_f = lambda a, v: a*gamma + v;
        tf_r_reverse = tf.scan(discount_f, tf.reverse(tf_r,[True, False]))
        tf_discounted_r = tf.reverse(tf_r_reverse,[True, False])
        return tf_discounted_r
        
        
def get_discount_rewards(transitions):
    discounted_r = [np.zeros([1]) for _ in range(len(transitions))]
    running_add = 0
    for t in reversed(range(0, len(transitions))):
        running_add = running_add * gamma + transitions[t][2]
        discounted_r[t][0] = running_add
        
    return discounted_r    
        
def train(sess, env, actor) :
    sess.run(tf.global_variables_initializer())    
    epoch = 3000
    plt.ioff()
    fig = plt.figure()
    reward_list = []
    for episode_num in range(epoch):
    
        observation = env.reset()
        states = []
        actions = []
        rewards = []
        reward_sum = 0        
        for _ in range(1000):
            obs_vector = np.expand_dims(observation, axis=0)
            act_prob = actor.predict(obs_vector); act_prob = act_prob[0,:]
            action = np.random.choice(env.action_space.n, p=act_prob)          
            actionblank = np.zeros_like(act_prob);  actionblank[action] = 1
            
            # take the acion in the environment
            old_observation = observation
            observation, reward, done, info = env.step(action)
            reward_sum += reward
            
            states.append(old_observation); actions.append(actionblank); rewards.append(reward)
            
            if done:
                print("episode : " + str(episode_num) + ", reward : " + str(reward_sum))
                reward_list.append(reward_sum)                
                break
       
        
        actor.train(states, actions, rewards)
        
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
    
    