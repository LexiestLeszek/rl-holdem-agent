import numpy as np
import random

class TexasHoldemEnvironment:
    def __init__(self):
        self.deck = None
        self.player_hand = []
        self.community_cards = []
        self.pot = 0
        self.player_chips = 1000
        self.opponent_chips = 1000
    
    def reset(self):
        self.deck = [i for i in range(52)]
        random.shuffle(self.deck)
        self.player_hand = [self.deck.pop(), self.deck.pop()]
        self.community_cards = []
        self.pot = 0
        self.player_chips = 1000
        self.opponent_chips = 1000
        return self.get_state()
    
    def deal_community_card(self):
        if len(self.deck) > 0:
            self.community_cards.append(self.deck.pop())
    
    def get_state(self):
        # Simplified state representation
        return tuple(sorted([card % 13 for card in self.player_hand + self.community_cards]))
    
    def step(self, action):
        reward = 0
        done = False
        
        if action == "call":
            bet_amount = min(100, self.player_chips // 10)
            self.pot += bet_amount * 2
            self.player_chips -= bet_amount
            self.opponent_chips -= bet_amount
            
            # Simulate opponent's turn
            if random.random() < 0.5:
                self.pot *= 2
                self.player_chips += self.pot
                self.opponent_chips -= self.pot
                reward = self.pot / 100
            else:
                self.opponent_chips += self.pot
                reward = -self.pot / 100
            
            done = True
        
        elif action == "raise":
            bet_amount = min(200, self.player_chips // 5)
            self.pot += bet_amount * 2
            self.player_chips -= bet_amount
            self.opponent_chips -= bet_amount
            
            # Simulate opponent's turn
            if random.random() < 0.3:
                self.pot *= 3
                self.player_chips += self.pot
                self.opponent_chips -= self.pot
                reward = self.pot / 100
            else:
                self.opponent_chips += self.pot
                reward = -self.pot / 100
            
            done = True
        
        elif action == "fold":
            self.opponent_chips += self.pot
            reward = -self.pot / 100
            done = True
        
        elif action == "check":
            # Check is valid after pre-flop, flop, turn, and river
            if len(self.community_cards) >= 3:
                # Opponent decides whether to call, raise, or fold
                if random.random() < 0.7:  # 70% chance of calling
                    bet_amount = min(100, self.player_chips // 10)
                    self.pot += bet_amount * 2
                    self.player_chips -= bet_amount
                    self.opponent_chips -= bet_amount
                    reward = -bet_amount / 100
                elif random.random() < 0.9:  # 90% chance of folding
                    self.opponent_chips += self.pot
                    reward = -self.pot / 100
                    done = True
                else:  # 30% chance of raising
                    bet_amount = min(200, self.player_chips // 5)
                    self.pot += bet_amount * 2
                    self.player_chips -= bet_amount
                    self.opponent_chips -= bet_amount
                    reward = -bet_amount / 100
            else:
                # Before community cards, check is invalid
                reward = -10  # Penalty for unnecessary check
                done = True
        
        return self.get_state(), reward, done, {}

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
    
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(["call", "raise", "fold", "check"])
        
        q_values = self.get_q_value(state)
        return max(q_values, key=q_values.get)
    
    def get_q_value(self, state):
        if state not in self.q_table:
            self.q_table[state] = {"call": 0, "raise": 0, "fold": 0, "check": 0}
        return self.q_table[state]
    
    def update(self, state, action, next_state, reward):
        q_value = self.get_q_value(state)[action]
        next_q_values = self.get_q_value(next_state)
        next_q_value = max(next_q_values.values())
        
        new_q_value = (1 - self.alpha) * q_value + self.alpha * (reward + self.gamma * next_q_value)
        self.q_table[state][action] = new_q_value

def train_agent(env, agent, num_episodes=10000):
    return ""

def test_agent(env, agent, num_games=1000):
    wins = 0
    for _ in range(num_games):
        state = env.reset()
        done = False
        
        # Simulate pre-flop
        action = agent.choose_action(state)
        _, reward, done, _ = env.step(action)
        
        if not done:
            # Simulate flop
            env.deal_community_card()
            env.deal_community_card()
            env.deal_community_card()
            
            action = agent.choose_action(env.get_state())
            _, reward, done, _ = env.step(action)
            
            if not done:
                # Simulate turn
                env.deal_community_card()
                
                action = agent.choose_action(env.get_state())
                _, reward, done, _ = env.step(action)
                
                if not done:
                    # Simulate river
                    env.deal_community_card()
                    
                    action = agent.choose_action(env.get_state())
                    _, reward, done, _ = env.step(action)
        
        if env.player_chips > env.opponent_chips:
            wins += 1
    
    win_rate = wins / num_games
    print(f"Win rate after training: {win_rate:.2%}")

# Main execution
env = TexasHoldemEnvironment()
agent = QLearningAgent(alpha=0.05, gamma=0.95, epsilon=0.05)

# Decay epsilon over time
epsilon_decay = 0.99
epsilon_min = 0.01
epsilon = agent.epsilon

for episode in range(5000000):
    state = env.reset()
    done = False
    
    # Simulate pre-flop
    action = agent.choose_action(state)
    next_state, reward, done, _ = env.step(action)
    agent.update(state, action, next_state, reward)
    state = next_state
    
    if not done:
        # Simulate flop
        env.deal_community_card()
        env.deal_community_card()
        env.deal_community_card()
        
        action = agent.choose_action(env.get_state())
        next_state, reward, done, _ = env.step(action)
        agent.update(env.get_state(), action, next_state, reward)
        
        if not done:
            # Simulate turn
            env.deal_community_card()
            
            action = agent.choose_action(env.get_state())
            next_state, reward, done, _ = env.step(action)
            agent.update(env.get_state(), action, next_state, reward)
            
            if not done:
                # Simulate river
                env.deal_community_card()
                
                action = agent.choose_action(env.get_state())
                next_state, reward, done, _ = env.step(action)
                agent.update(env.get_state(), action, next_state, reward)
    
    if episode % 1000 == 0:
        print(f"Episode {episode}, Player chips: {env.player_chips}")
    
    # Decay epsilon
    epsilon *= epsilon_decay
    epsilon = max(epsilon, epsilon_min)
    agent.epsilon = epsilon

test_agent(env, agent)
