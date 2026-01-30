#!/usr/bin/env python3
"""Reinforcement learning training example for distillation column control.

This script demonstrates how to:
1. Set up the Gymnasium environment for RL
2. Train an agent using stable-baselines3 (if available)
3. Evaluate the trained policy
4. Compare against a baseline controller

Requirements:
    pip install stable-baselines3

Run with: python examples/rl_training.py
"""

import numpy as np
import time
from typing import Callable

from jax_distillation.env import (
    DistillationColumnEnv,
    wrap_env,
    RewardConfig,
)


def evaluate_policy(
    env: DistillationColumnEnv,
    policy: Callable,
    n_episodes: int = 5,
    max_steps: int = 200,
) -> dict:
    """Evaluate a policy over multiple episodes.

    Args:
        env: Environment to evaluate in.
        policy: Function mapping observation to action.
        n_episodes: Number of evaluation episodes.
        max_steps: Maximum steps per episode.

    Returns:
        Dictionary with evaluation metrics.
    """
    episode_rewards = []
    episode_lengths = []
    final_x_D = []
    final_x_B = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0

        for step in range(max_steps):
            action = policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        if "outputs" in info:
            final_x_D.append(info["outputs"]["x_D"])
            final_x_B.append(info["outputs"]["x_B"])

    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "mean_x_D": np.mean(final_x_D) if final_x_D else None,
        "mean_x_B": np.mean(final_x_B) if final_x_B else None,
    }


def random_policy(obs: np.ndarray, env: DistillationColumnEnv) -> np.ndarray:
    """Random policy baseline."""
    return env.action_space.sample()


def constant_policy(obs: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Constant action policy."""
    return action


def simple_heuristic_policy(obs: np.ndarray, env: DistillationColumnEnv) -> np.ndarray:
    """Simple heuristic policy based on observation.

    Uses a basic rule: if distillate purity is low, increase reboiler duty.
    """
    # Observation structure (with include_flows=True):
    # [0:n_trays]: tray temperatures (normalized)
    # [n_trays:2*n_trays]: tray compositions
    # ... more features

    # Get action space bounds
    low = env.action_space.low
    high = env.action_space.high

    # Start with middle values
    action = (low + high) / 2

    # Adjust Q_R based on assumed distillate composition
    # (In real scenario, would need to know observation structure)
    # Higher Q_R for more separation
    action[0] = high[0] * 0.6  # 60% of max reboiler duty

    # Moderate reflux ratio
    action[1] = low[1] + (high[1] - low[1]) * 0.4  # 40% of range

    return action.astype(np.float32)


def train_with_sb3(env, total_timesteps: int = 10000):
    """Train using stable-baselines3 PPO.

    Args:
        env: Gymnasium environment.
        total_timesteps: Total training timesteps.

    Returns:
        Trained model or None if sb3 not available.
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env

        print("\nTraining with stable-baselines3 PPO...")

        # Wrap environment for better training
        wrapped_env = wrap_env(
            env,
            normalize_action=True,
            record_stats=True,
        )

        # Create and train model
        model = PPO(
            "MlpPolicy",
            wrapped_env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
        )

        model.learn(total_timesteps=total_timesteps)

        return model, wrapped_env

    except ImportError:
        print("\nstable-baselines3 not installed.")
        print("Install with: pip install stable-baselines3")
        return None, None


def main():
    print("=" * 60)
    print("RL Training for Distillation Column Control")
    print("=" * 60)

    # Create environment with custom reward config
    reward_config = RewardConfig(
        x_D_target=0.90,  # Target 90% purity in distillate
        x_B_target=0.10,  # Target 10% in bottoms
        purity_weight=1.0,
        energy_weight=0.05,  # Small penalty for energy use
        stability_weight=0.1,
        constraint_weight=10.0,
        tolerance_D=0.05,
        tolerance_B=0.05,
    )

    env = DistillationColumnEnv(
        reward_config=reward_config,
        max_episode_steps=200,
        use_reduced_action_space=False,
    )

    print(f"\nEnvironment created:")
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.shape}")
    print(f"  Action bounds: {env.action_space.low} to {env.action_space.high}")

    # =========================================================================
    # Baseline Evaluation
    # =========================================================================
    print("\n" + "-" * 50)
    print("Evaluating Baseline Policies")
    print("-" * 50)

    # Random policy
    print("\n1. Random Policy:")
    random_results = evaluate_policy(
        env,
        lambda obs: random_policy(obs, env),
        n_episodes=5,
    )
    print(f"   Mean reward: {random_results['mean_reward']:.2f} ± {random_results['std_reward']:.2f}")
    if random_results['mean_x_D']:
        print(f"   Final x_D: {random_results['mean_x_D']:.3f}")
        print(f"   Final x_B: {random_results['mean_x_B']:.3f}")

    # Constant policy (moderate settings)
    print("\n2. Constant Policy (moderate settings):")
    moderate_action = np.array([5000.0, 3.0, 0.03, 0.02], dtype=np.float32)
    constant_results = evaluate_policy(
        env,
        lambda obs: constant_policy(obs, moderate_action),
        n_episodes=5,
    )
    print(f"   Mean reward: {constant_results['mean_reward']:.2f} ± {constant_results['std_reward']:.2f}")
    if constant_results['mean_x_D']:
        print(f"   Final x_D: {constant_results['mean_x_D']:.3f}")
        print(f"   Final x_B: {constant_results['mean_x_B']:.3f}")

    # Heuristic policy
    print("\n3. Heuristic Policy:")
    heuristic_results = evaluate_policy(
        env,
        lambda obs: simple_heuristic_policy(obs, env),
        n_episodes=5,
    )
    print(f"   Mean reward: {heuristic_results['mean_reward']:.2f} ± {heuristic_results['std_reward']:.2f}")
    if heuristic_results['mean_x_D']:
        print(f"   Final x_D: {heuristic_results['mean_x_D']:.3f}")
        print(f"   Final x_B: {heuristic_results['mean_x_B']:.3f}")

    # =========================================================================
    # RL Training (if stable-baselines3 available)
    # =========================================================================
    print("\n" + "-" * 50)
    print("RL Training")
    print("-" * 50)

    model, wrapped_env = train_with_sb3(env, total_timesteps=10000)

    if model is not None:
        print("\n4. Trained PPO Policy:")

        def ppo_policy(obs):
            action, _ = model.predict(obs, deterministic=True)
            return action

        # Evaluate trained model
        ppo_results = evaluate_policy(
            wrapped_env,
            ppo_policy,
            n_episodes=5,
        )
        print(f"   Mean reward: {ppo_results['mean_reward']:.2f} ± {ppo_results['std_reward']:.2f}")
        if ppo_results['mean_x_D']:
            print(f"   Final x_D: {ppo_results['mean_x_D']:.3f}")
            print(f"   Final x_B: {ppo_results['mean_x_B']:.3f}")

        # Compare all results
        print("\n" + "=" * 50)
        print("Summary Comparison")
        print("=" * 50)
        print(f"{'Policy':<20} {'Mean Reward':>15} {'x_D':>10} {'x_B':>10}")
        print("-" * 55)
        print(f"{'Random':<20} {random_results['mean_reward']:>15.2f} {random_results['mean_x_D'] or 0:>10.3f} {random_results['mean_x_B'] or 0:>10.3f}")
        print(f"{'Constant':<20} {constant_results['mean_reward']:>15.2f} {constant_results['mean_x_D'] or 0:>10.3f} {constant_results['mean_x_B'] or 0:>10.3f}")
        print(f"{'Heuristic':<20} {heuristic_results['mean_reward']:>15.2f} {heuristic_results['mean_x_D'] or 0:>10.3f} {heuristic_results['mean_x_B'] or 0:>10.3f}")
        print(f"{'PPO (trained)':<20} {ppo_results['mean_reward']:>15.2f} {ppo_results['mean_x_D'] or 0:>10.3f} {ppo_results['mean_x_B'] or 0:>10.3f}")

    else:
        print("\nTo train an RL agent, install stable-baselines3:")
        print("  pip install stable-baselines3")
        print("\nAlternatively, you can use other RL libraries like:")
        print("  - CleanRL (https://github.com/vwxyzjn/cleanrl)")
        print("  - PureJaxRL (https://github.com/luchris429/purejaxrl)")
        print("  - RLlib (https://docs.ray.io/en/latest/rllib/)")

    # =========================================================================
    # Example: Manual Episode Rollout
    # =========================================================================
    print("\n" + "-" * 50)
    print("Example Episode Rollout")
    print("-" * 50)

    obs, info = env.reset()
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Initial info: step={info['step']}, time={info['time']:.1f}s")

    total_reward = 0
    for step in range(10):
        # Use heuristic policy
        action = simple_heuristic_policy(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 2 == 0:
            print(f"  Step {step+1}: reward={reward:.3f}, "
                  f"x_D={info['outputs']['x_D']:.3f}, "
                  f"x_B={info['outputs']['x_B']:.3f}")

    print(f"\nTotal reward over 10 steps: {total_reward:.2f}")

    print("\n" + "=" * 60)
    print("RL training example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
