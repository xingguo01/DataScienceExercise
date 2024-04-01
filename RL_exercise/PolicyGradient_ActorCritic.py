import tensorflow as tf
import numpy as np
import gymnasium as gym
from tensorflow.keras import Model, layers, initializers, optimizers
import collections
import sys
import itertools
import matplotlib
from tensorflow.python.framework.ops import enable_eager_execution

if "../" not in sys.path:
  sys.path.append("../") 
from lib.envs.cliff_walking import CliffWalkingEnv
from lib import plotting

matplotlib.style.use('ggplot')
env = CliffWalkingEnv()
print("Eager execution:", tf.executing_eagerly())
enable_eager_execution()
class PolicyEstimator(Model):
    def __init__(self, num_actions, learning_rate=0.01):
        super(PolicyEstimator, self).__init__()
        self.dense = layers.Dense(num_actions, activation='softmax',
                                  kernel_initializer=initializers.Zeros())
        self.optimizer = optimizers.Adam(learning_rate=learning_rate)

    def call(self, inputs):
    #    state_one_hot = tf.one_hot(state, depth=env.observation_space.n)
    #    inputs=tf.expand_dims(state_one_hot, 0) 
        return self.dense(inputs)

    def action_probs(self, state):
        state_one_hot = tf.one_hot(state, depth=env.observation_space.n)
        return self(tf.expand_dims(state_one_hot, 0))

    def update(self, state, target, action):
        with tf.GradientTape() as tape:
            probs = self.action_probs(state)
            picked_action_prob = probs[0, action]
            #print('policy debug:',probs,picked_action_prob)
            loss = -tf.math.log(picked_action_prob) * target
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

class ValueEstimator(Model):
    def __init__(self, learning_rate=0.1):
        super(ValueEstimator, self).__init__()
        self.dense = layers.Dense(1, kernel_initializer=initializers.Zeros())
        self.optimizer = optimizers.Adam(learning_rate=learning_rate)

    def call(self, inputs):
        return self.dense(inputs)

    def value(self, state):
        state_one_hot = tf.one_hot(state, depth=env.observation_space.n)
        return tf.squeeze(self(tf.expand_dims(state_one_hot, 0)), axis=0)

    def update(self, state, target):
        with tf.GradientTape() as tape:
            value_pred = self.value(state)
            loss = tf.math.reduce_mean(tf.square(target - value_pred))
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

def actor_critic(env, estimator_policy, estimator_value, num_episodes, discount_factor=1.0):
    """
    Actor Critic Algorithm. Optimizes the policy 
    function approximator using policy gradient.
    
    Args:
        env: OpenAI environment.
        estimator_policy: Policy Function to be optimized 
        estimator_value: Value function approximator, used as a critic
        num_episodes: Number of episodes to run for
        discount_factor: Time-discount factor
    
    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))    
    
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    
    for i_episode in range(num_episodes):
        # Reset the environment and pick the fisrst action
        state = env.reset()
        
        episode = []
        
        # One step in the environment
        for t in itertools.count():
            
            # Take a step
            action_probs_tensor = estimator_policy.action_probs(state)
            #print('action probs tensor', action_probs_tensor)
            action_probs=action_probs_tensor.numpy()
            #print('action probs', action_probs)
            #tf.print(action_probs)
            
            # Use TensorFlow to sample an action based on the predicted probabilities
            #logits = tf.math.log(action_probs_tensor + 1e-10)  # Prevent log(0)
            #action_tensor = tf.random.categorical(logits, 1)[0, 0]
            #action_tensor = tf.random.categorical(logits, num_samples=1, seed=42)
            # Convert the action tensor to a numpy scalar for use with the Gym environment
            #action_tensor = np.random.choice(np.arange(len(action_probs)), p=action_probs)

            #print('Action tensor:', action_tensor)
            # Should work with eager execution
            
            
            #print("Eager execution:", tf.executing_eagerly())
            # After action_tensor is defined
            
            action = np.random.choice(np.arange(action_probs[0].shape[0]), p=action_probs[0])
            #action = tf.random.categorical(tf.math.log(action_probs_tensor), 1)[0, 0]
            #print('Action:', action)
            
            
            next_state, reward, done, _ = env.step(action)
            
            # Keep track of the transition
            episode.append(Transition(
              state=state, action=action, reward=reward, next_state=next_state, done=done))
            
            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            
            # Calculate TD Target
            value_next = estimator_value.value(next_state).numpy()[0]
            td_target = reward + discount_factor * value_next
            td_error = td_target - estimator_value.value(state).numpy()[0]
            
            # Update the value estimator
            estimator_value.update(state, td_target)
            # Update the policy estimator
            # using the td error as our advantage estimate
            estimator_policy.update(state, td_error, action)
            
            # Print out which step we're on, useful for debugging.
            print("\rStep {} @ Episode {}/{} ({})".format(
                    t, i_episode + 1, num_episodes, stats.episode_rewards[i_episode - 1]), end="")

            if done:
                break
                
            state = next_state
    
    return stats
# Create a new graph

#graph = tf.Graph()
#with graph.as_default():
    # Your TensorFlow operations here
    # For example:
policy_estimator = PolicyEstimator(env.action_space.n)
value_estimator = ValueEstimator()
    
    # If you need to use a Session, do it within this block
   
stats = actor_critic(env, policy_estimator, value_estimator, 300)
plotting.plot_episode_stats(stats, smoothing_window=10)
