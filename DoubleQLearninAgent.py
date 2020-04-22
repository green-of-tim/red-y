def make_tables(obs_num, action_num):
  q1_table = np.zeros((obs_num, action_num))
  q2_table = np.zeros((obs_num, action_num))
  return q1_table, q2_table

def greedy(eps):
  pivot = random.uniform(0, 1)
  return pivot > eps

def train(Q1_table, Q2_table, episodes_num, env)
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
        action = np.argmax(Q1_table[state,:] + Q2_table[state,:])
      else:
        action = env.action_space.sample()

      # Perform it
      new_state, reward, done, _ = env.step(action)

      # Update q-table
      if np.random.random() > 0.5:
        a1_max = np.argmax(Q1_table[new_state, :])
        Q1_table[state, action] = Q1_table[state, action] + learning_rate * (reward + 
                    discount * Q2_table[new_state, a1_max] - Q1_table[state, action])
      else:
        a2_max = np.argmax(Q2_table[new_state, :])
        Q2_table[state, action] = Q2_table[state, action] + learning_rate * (reward + 
                    discount * Q1_table[new_state, a2_max] - Q2_table[state, action])
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
  return Q1_table, Q2_table, rewards, steps

def evaluate(Q1_table, Q2_table, eval_ep, env):
  Q_table = Q1_table + Q2_table
  reward_eval = []
  steps_eval = []

  for episode in range(1000):
      total_reward = 0
      state = env.reset()
      done = False

      for step in range(max_steps):
          
          action = np.argmax(Q_table[state,:])
          
          new_state, reward, done, _ = env.step(action)

          total_reward += reward
          
          if done == True:
            steps_eval.append(step + 1)
            break
          state = new_state
        
      if done == False:
        steps_eval.append(max_steps + 1)  
      reward_eval.append(total_reward)
  env.close()
  return rewards_eval, steps_eval
