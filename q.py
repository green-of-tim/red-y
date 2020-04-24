# -*- coding: utf-8 -*-
import numpy as np

class QLearningAgent:

    def make_table(obs_num, action_nim):
      q_table = np.zeros((obs_num, action_num))
      return q_table

    def greedy(eps):
      pivot = random.uniform(0, 1)
      return pivot > eps

    def train(Q_table, episodes_num, env):

      rewards = []
      steps = []
      epsilon = epsilon_max

      for episode in range(episodes_num):
        state = env.reset()
        reward = 0
        done = False
        cumul_rewards = 0

        for step in range(max_steps):

          # Choose an action
          if greedy(epsilon):
            action = np.argmax(q_table[state,:])
          else:
            action = env.action_space.sample()

          # Perform it
          new_state, reward, done, _ = env.step(action)

          # Update q-table
          Q_table[state, action] = Q_table[state, action] + learning_rate * (reward + 
                        discount * np.max(Q_table[new_state, :]) - Q_table[state, action])
          state = new_state

          cumul_rewards += reward

          if done == True:
            steps.append(step + 1)
            break

        if done == False:
          steps.append(max_steps + 1)
        rewards.append(cumul_rewards)
        epsilon = epsilon_min + (epsilon_max - epsilon_min)*np.exp(-decay_rate * episode) 

      env.close()
      return Q_table, rewards, steps

    def evaluate(Q_table, eval_num, env):
      rewards_test = []
      steps_test = []

      for episode in range(eval_num):
          state = env.reset()
          total_reward = 0
          step = 0
          done = False

          for step in range(max_steps):

              action = np.argmax(q_table[state,:])

              new_state, reward, done, info = env.step(action)

              total_reward += reward

              if done == True:
                steps.append(step + 1)
                break
              state = new_state
          if done == False:
            steps_test.append(max_steps + 1)
          rewards_test.append(total_reward)

      env.close()
      return rewards_test, steps_test
