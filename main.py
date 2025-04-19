import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import time
from tqdm import tqdm

class SimpleGridWorld:
    def __init__(self, width=5, height=5, adversary_count=1, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.width = width
        self.height = height
        self.adversary_count = adversary_count
        self.reset()
        self.goal_pos = (width-1, height-1)
        self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.action_names = ["up", "right", "down", "left"]
        self.step_reward = -0.1
        self.goal_reward = 10.0
        self.collision_penalty = -10.0
        self.tokens = {}
        for x in range(width):
            for y in range(height):
                self.tokens[(x, y)] = 1

    def reset(self):
        self.avatar_pos = (0, 0)
        self.adversary_positions = []
        while len(self.adversary_positions) < self.adversary_count:
            pos = (random.randint(0, self.width-1), random.randint(0, self.height-1))
            if pos != self.avatar_pos and pos != (self.width-1, self.height-1):
                self.adversary_positions.append(pos)
        return self._get_state()

    def _get_state(self):
        return (self.avatar_pos, tuple(self.adversary_positions))

    def is_valid_pos(self, pos):
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def move_avatar(self, action):
        dx, dy = self.actions[action]
        new_pos = (self.avatar_pos[0] + dx, self.avatar_pos[1] + dy)
        if self.is_valid_pos(new_pos):
            self.avatar_pos = new_pos

    def move_adversaries(self):
        for i in range(len(self.adversary_positions)):
            move_probs = self._get_adversary_behavior(i)
            action = np.random.choice(len(self.actions), p=move_probs)
            dx, dy = self.actions[action]
            new_pos = (self.adversary_positions[i][0] + dx, self.adversary_positions[i][1] + dy)
            if self.is_valid_pos(new_pos):
                self.adversary_positions[i] = new_pos

    def _get_adversary_behavior(self, adversary_idx):
        adv_pos = self.adversary_positions[adversary_idx]
        x_diff = self.avatar_pos[0] - adv_pos[0]
        y_diff = self.avatar_pos[1] - adv_pos[1]
        probs = np.ones(4) * 0.25
        if abs(x_diff) > abs(y_diff):
            if x_diff < 0:
                probs[0] += 0.2
                probs[2] -= 0.1
            elif x_diff > 0:
                probs[2] += 0.2
                probs[0] -= 0.1
        else:
            if y_diff < 0:
                probs[3] += 0.2
                probs[1] -= 0.1
            elif y_diff > 0:
                probs[1] += 0.2
                probs[3] -= 0.1
        return probs / np.sum(probs)

    def step(self, action):
        self.move_avatar(action)
        reached_goal = self.avatar_pos == self.goal_pos
        collision = self.avatar_pos in self.adversary_positions
        reward = self.step_reward
        if self.tokens.get(self.avatar_pos, 0) == 1:
            reward += 1.0
            self.tokens[self.avatar_pos] = 0
        if reached_goal:
            reward += self.goal_reward
        if collision:
            reward += self.collision_penalty
        self.move_adversaries()
        done = reached_goal or collision
        return self._get_state(), reward, done, {"reached_goal": reached_goal, "collision": collision}

    def get_available_actions(self):
        valid_actions = []
        for i, (dx, dy) in enumerate(self.actions):
            new_pos = (self.avatar_pos[0] + dx, self.avatar_pos[1] + dy)
            if self.is_valid_pos(new_pos):
                valid_actions.append(i)
        return valid_actions

    def render(self):
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        grid[self.goal_pos[0]][self.goal_pos[1]] = 'G'
        for pos in self.adversary_positions:
            grid[pos[0]][pos[1]] = 'X'
        grid[self.avatar_pos[0]][self.avatar_pos[1]] = 'A'
        for i in range(self.height):
            print('|' + '|'.join(grid[i]) + '|')
        print()

class ModelChecker:
    def __init__(self, environment, horizon=3):
        self.env = environment
        self.horizon = horizon
        self.collision_probs = {}

    def compute_collision_probability(self, state, action, depth=0):
        avatar_pos, adversary_positions = state
        cache_key = (avatar_pos, tuple(adversary_positions), action, depth)
        if cache_key in self.collision_probs:
            return self.collision_probs[cache_key]
        if depth == self.horizon:
            return 0.0
        env_copy = SimpleGridWorld(self.env.width, self.env.height, len(adversary_positions))
        env_copy.avatar_pos = avatar_pos
        env_copy.adversary_positions = list(adversary_positions)
        new_state, _, done, info = env_copy.step(action)
        if info["collision"]:
            self.collision_probs[cache_key] = 1.0
            return 1.0
        if done:
            self.collision_probs[cache_key] = 0.0
            return 0.0
        valid_actions = env_copy.get_available_actions()
        if not valid_actions:
            self.collision_probs[cache_key] = 0.0
            return 0.0
        min_prob = 1.0
        for next_action in valid_actions:
            prob = self.compute_collision_probability(new_state, next_action, depth + 1)
            min_prob = min(min_prob, prob)
        self.collision_probs[cache_key] = min_prob
        return min_prob

class ProbabilisticShield:
    def __init__(self, model_checker, delta=0.7):
        self.model_checker = model_checker
        self.delta = delta

    def get_action_values(self, state):
        env = self.model_checker.env
        valid_actions = env.get_available_actions()
        action_values = {}
        for action in valid_actions:
            prob = self.model_checker.compute_collision_probability(state, action)
            action_values[action] = prob
        return action_values

    def get_shielded_actions(self, state):
        action_values = self.get_action_values(state)
        if not action_values:
            return []
        opt_value = min(action_values.values())
        allowed_actions = []
        for action, value in action_values.items():
            if self.delta * value <= opt_value:
                allowed_actions.append(action)
        return allowed_actions

class QAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.95, exploration_rate=0.1):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = defaultdict(lambda: np.zeros(4))

    def choose_action(self, state, available_actions):
        if not available_actions:
            return None
        if random.random() < self.exploration_rate:
            return random.choice(available_actions)
        else:
            state_key = (state[0], tuple(state[1]))
            q_values = self.q_table[state_key]
            best_value = float('-inf')
            best_actions = []
            for action in available_actions:
                if q_values[action] > best_value:
                    best_value = q_values[action]
                    best_actions = [action]
                elif q_values[action] == best_value:
                    best_actions.append(action)
            return random.choice(best_actions)

    def update_q_value(self, state, action, reward, next_state, done):
        state_key = (state[0], tuple(state[1]))
        next_state_key = (next_state[0], tuple(next_state[1]))
        current_q = self.q_table[state_key][action]
        max_next_q = 0 if done else np.max(self.q_table[next_state_key])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state_key][action] = new_q

def run_experiment(shielded=True, num_episodes=500, grid_size=5, adversary_count=1, delta=0.7, horizon=3, seed=42):
    env = SimpleGridWorld(width=grid_size, height=grid_size, adversary_count=adversary_count, seed=seed)
    model_checker = ModelChecker(env, horizon=horizon)
    shield = ProbabilisticShield(model_checker, delta=delta)
    agent = QAgent(env)
    episode_rewards = []
    episode_lengths = []
    collisions = 0
    goals_reached = 0
    for episode in tqdm(range(num_episodes), desc="Training"):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        while not done and episode_length < 100:
            available_actions = env.get_available_actions()
            if shielded:
                shielded_actions = shield.get_shielded_actions(state)
                if not shielded_actions and available_actions:
                    action_values = shield.get_action_values(state)
                    safest_action = min(action_values, key=action_values.get)
                    shielded_actions = [safest_action]
                available_actions = shielded_actions
            action = agent.choose_action(state, available_actions)
            if action is None:
                break
            next_state, reward, done, info = env.step(action)
            agent.update_q_value(state, action, reward, next_state, done)
            episode_reward += reward
            episode_length += 1
            state = next_state
            if info["collision"]:
                collisions += 1
            if info["reached_goal"]:
                goals_reached += 1
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    return {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "collisions": collisions,
        "goals_reached": goals_reached
    }

def plot_results(shielded_results, unshielded_results, window_size=20):
    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    shielded_rewards_ma = moving_average(shielded_results["episode_rewards"], window_size)
    unshielded_rewards_ma = moving_average(unshielded_results["episode_rewards"], window_size)
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(range(window_size-1, len(shielded_results["episode_rewards"])), shielded_rewards_ma, label='Shielded')
    plt.plot(range(window_size-1, len(unshielded_results["episode_rewards"])), unshielded_rewards_ma, label='Unshielded')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Moving Average Reward (window size = {})'.format(window_size))
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 1, 2)
    labels = ['Goals Reached', 'Collisions']
    shielded_stats = [shielded_results["goals_reached"], shielded_results["collisions"]]
    unshielded_stats = [unshielded_results["goals_reached"], unshielded_results["collisions"]]
    x = np.arange(len(labels))
    width = 0.35
    plt.bar(x - width/2, shielded_stats, width, label='Shielded')
    plt.bar(x + width/2, unshielded_stats, width, label='Unshielded')
    plt.xlabel('Metric')
    plt.ylabel('Count')
    plt.title('Performance Metrics')
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    print("Running experiments... This may take a few minutes.")
    seed = 42
    shielded_results = run_experiment(shielded=True, num_episodes=500, grid_size=5, adversary_count=1, delta=0.7, horizon=3, seed=seed)
    unshielded_results = run_experiment(shielded=False, num_episodes=500, grid_size=5, adversary_count=1, delta=0.7, horizon=3, seed=seed)
    print("\nResults:")
    print("Shielded RL:")
    print(f"  Goals reached: {shielded_results['goals_reached']}")
    print(f"  Collisions: {shielded_results['collisions']}")
    print(f"  Average reward: {np.mean(shielded_results['episode_rewards']):.2f}")
    print("\nUnshielded RL:")
    print(f"  Goals reached: {unshielded_results['goals_reached']}")
    print(f"  Collisions: {unshielded_results['collisions']}")
    print(f"  Average reward: {np.mean(unshielded_results['episode_rewards']):.2f}")
    plot_results(shielded_results, unshielded_results)

if __name__ == "__main__":
    main()