import numpy as np
import matplotlib.pyplot as plt

# Introduction to the Multi-Armed Bandit Problem
# ------------------------------------------------
# Imagine you're in a casino with multiple slot machines (bandits).
# Each machine has a different, unknown probability of paying out.
# Your goal is to maximize your total winnings by deciding which machines to play,
# and how often to play each machine.

# This problem is known as the "multi-armed bandit problem."
# It poses a fundamental dilemma: exploration vs. exploitation.
# Exploration involves trying out different machines to learn about their payouts,
# while exploitation involves using the knowledge you've gained to maximize your rewards.

# Now, let's dive into some common strategies to solve this problem.

class Bandit:
    """
    Represents a single slot machine (bandit).
    """
    def __init__(self, true_mean):
        """
        Initialize the bandit with a true mean, which is the actual payout probability.
        """
        self.true_mean = true_mean  # The true mean payout of the bandit (unknown to the agent).
        self.estimated_mean = 0     # The estimated mean payout of the bandit (agent's belief).
        self.n_pulls = 0            # Number of times this bandit has been pulled.
    
    def pull(self):
        """
        Simulate pulling the bandit's lever.
        Returns a reward based on the bandit's true mean.
        """
        return np.random.randn() + self.true_mean  # Random reward based on normal distribution
    
    def update(self, reward):
        """
        Update the estimated mean payout of the bandit using the new reward.
        """
        self.n_pulls += 1
        self.estimated_mean = (self.estimated_mean * (self.n_pulls - 1) + reward) / self.n_pulls

def run_experiment(bandit_probs, num_pulls, strategy, seed=1):
    """
    Runs the multi-armed bandit experiment with a given strategy.
    
    Parameters:
        bandit_probs: List of true mean probabilities for each bandit.
        num_pulls: Number of times to pull the levers.
        strategy: Function that determines the next bandit to pull.
    
    Returns:
        rewards: List of rewards obtained over time.
        cumulative_average: List of cumulative average rewards.
    """
    if seed is not None:
        np.random.seed(seed)  # Fixing the random seed for reproducibility
    bandits = [Bandit(p) for p in bandit_probs]
    rewards = np.zeros(num_pulls)
    for i in range(num_pulls):
        j = strategy(bandits)
        reward = bandits[j].pull()
        bandits[j].update(reward)
        rewards[i] = reward
    
    cumulative_average = np.cumsum(rewards) / (np.arange(num_pulls) + 1)
    return rewards, cumulative_average

# Strategy 1: Epsilon-Greedy
# ----------------------------
# The epsilon-greedy strategy is one of the most commonly used techniques.
# It balances exploration and exploitation by introducing a probability epsilon (ε) of exploring.
# With probability (1 - ε), the agent exploits by choosing the bandit with the highest estimated mean.
# With probability ε, the agent explores by choosing a bandit at random.

def epsilon_greedy_strategy(bandits, epsilon=0.01):
    """
    Epsilon-greedy strategy: choose a random bandit with probability epsilon, 
    otherwise choose the bandit with the highest estimated mean.
    """
    if np.random.random() < epsilon:
        return np.random.choice(len(bandits))  # Explore
    else:
        estimated_means = [bandit.estimated_mean for bandit in bandits]
        return np.argmax(estimated_means)  # Exploit

# Strategy 2: Optimistic Initial Values
# --------------------------------------
# The optimistic initial values strategy is a variant of epsilon-greedy.
# Instead of initializing the estimated means to 0, we initialize them to a high value.
# This encourages the agent to explore more initially, as the high initial estimates 
# make all bandits seem promising.

def optimistic_initial_values_strategy(bandits, initial_value=10):
    """
    Optimistic initial values strategy: initialize estimated means to a high value,
    encouraging exploration.
    """
    for bandit in bandits:
        bandit.estimated_mean = initial_value
    
    return epsilon_greedy_strategy(bandits, epsilon=0.0)  # No need for epsilon, as exploration is implicit

# Strategy 3: UCB1 (Upper Confidence Bound)
# -----------------------------------------
# The UCB1 algorithm is another popular strategy that selects the bandit with the highest 
# upper confidence bound on its reward estimate.
# This approach balances exploration and exploitation by favoring bandits that either
# have a high estimated mean or haven't been pulled much, thus allowing for exploration.

def ucb1_strategy(bandits):
    """
    UCB1 strategy: choose the bandit with the highest upper confidence bound.
    """
    total_pulls = sum(bandit.n_pulls for bandit in bandits) + 1e-5  # Add a small value to avoid log(0)
    
    ucb_values = []
    for bandit in bandits:
        if bandit.n_pulls == 0:
            # Handle case where bandit has never been pulled
            ucb_values.append(float('inf'))  # Infinite UCB value to ensure exploration
        else:
            bonus = np.sqrt(2 * np.log(total_pulls) / (bandit.n_pulls + 1e-5))
            ucb_values.append(bandit.estimated_mean + bonus)
    
    return np.argmax(ucb_values)

# Visualization and Analysis
# ---------------------------
# Now, let's run the experiment with these strategies and visualize the results.

def plot_results(bandit_probs, num_pulls, strategies, strategy_names):
    """
    Plot the results of different strategies.
    
    Parameters:
        bandit_probs: List of true mean probabilities for each bandit.
        num_pulls: Number of times to pull the levers.
        strategies: List of strategy functions to compare.
        strategy_names: List of names for each strategy.
    """
    for strategy, name in zip(strategies, strategy_names):
        _, cumulative_average = run_experiment(bandit_probs, num_pulls, strategy)
        plt.plot(cumulative_average, label=name)
    
    plt.xlabel('Number of Pulls')
    plt.ylabel('Cumulative Average Reward')
    plt.legend()
    plt.title('Multi-Armed Bandit Strategies Comparison')
    plt.show()

# Let's define our bandits with different probabilities of payout.
bandit_probs = [0.2, 0.5, 0.75]
num_pulls = 1000

# List of strategies to compare.
strategies = [lambda bandits: epsilon_greedy_strategy(bandits, epsilon=0.1),
              lambda bandits: optimistic_initial_values_strategy(bandits, initial_value=10),
              ucb1_strategy]

# Corresponding strategy names.
strategy_names = ['Epsilon-Greedy', 'Optimistic Initial Values', 'UCB1']

# Plotting the results.
plot_results(bandit_probs, num_pulls, strategies, strategy_names)
