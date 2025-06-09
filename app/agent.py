import random

class QLearningAgent:
    def __init__(self, epsilon=0.1, alpha=0.9, gamma=0.99):
        self.q_table = {}
        self.epsilon = epsilon  # Taxa de exploração
        self.alpha = alpha      # Taxa de aprendizado
        self.gamma = gamma      # Fator de desconto
        
    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)
    
    def set_q(self, state, action, value):
        self.q_table[(state, action)] = value
        
    def choose_action(self, state, valid_actions, training=True):
        if not valid_actions:
            return None
            
        if random.random() < self.epsilon and training:
            return random.choice(valid_actions)
        else:
            q_values = [(self.get_q(state, a), a) for a in valid_actions]
            max_q = max(q_values, key=lambda x: x[0])[0]
            best_actions = [a for q, a in q_values if q == max_q]
            return best_actions[0]
    
    def learn(self, state, action, reward, next_state, next_valid_actions):
        max_next_q = max([self.get_q(next_state, a) for a in next_valid_actions]) if next_valid_actions else 0
        current_q = self.get_q(state, action)
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.set_q(state, action, new_q)
    
    def get_policy(self, state, valid_actions):
        if not valid_actions:
            return None
        q_values = [(self.get_q(state, a), a) for a in valid_actions]
        max_q = max(q_values, key=lambda x: x[0])[0]
        best_actions = [a for q, a in q_values if q == max_q]
        return best_actions[0]