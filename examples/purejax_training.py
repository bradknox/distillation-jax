#!/usr/bin/env python3
"""Pure JAX RL training example for distillation column control.

This script demonstrates high-throughput RL training using:
1. DistillationEnvJax (Gymnax-style JAX-native environment)
2. Vectorized environments via jax.vmap
3. A simple PPO implementation in pure JAX

This achieves 10-100x higher throughput than NumPy-based RL libraries
like stable-baselines3.

Run with: python examples/purejax_training.py
"""

import time
from functools import partial
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jax_distillation.env.jax_env import DistillationEnvJax, EnvState, EnvParams


# =============================================================================
# PPO Components (Pure JAX)
# =============================================================================


class ActorCritic(NamedTuple):
    """Simple MLP actor-critic parameters."""
    actor_w1: jnp.ndarray
    actor_w2: jnp.ndarray
    actor_mean: jnp.ndarray
    actor_logstd: jnp.ndarray
    critic_w1: jnp.ndarray
    critic_w2: jnp.ndarray
    critic_out: jnp.ndarray


def init_actor_critic(key: jnp.ndarray, obs_dim: int, action_dim: int, hidden_dim: int = 64) -> ActorCritic:
    """Initialize actor-critic network parameters."""
    keys = jax.random.split(key, 7)

    # Xavier initialization
    def xavier_init(key, shape):
        std = jnp.sqrt(2.0 / (shape[0] + shape[1]))
        return jax.random.normal(key, shape) * std

    return ActorCritic(
        actor_w1=xavier_init(keys[0], (obs_dim, hidden_dim)),
        actor_w2=xavier_init(keys[1], (hidden_dim, hidden_dim)),
        actor_mean=xavier_init(keys[2], (hidden_dim, action_dim)) * 0.01,
        actor_logstd=jnp.zeros((action_dim,)),
        critic_w1=xavier_init(keys[3], (obs_dim, hidden_dim)),
        critic_w2=xavier_init(keys[4], (hidden_dim, hidden_dim)),
        critic_out=xavier_init(keys[5], (hidden_dim, 1)) * 0.01,
    )


def forward_actor(params: ActorCritic, obs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Forward pass through actor network."""
    x = jnp.tanh(obs @ params.actor_w1)
    x = jnp.tanh(x @ params.actor_w2)
    mean = x @ params.actor_mean
    std = jnp.exp(params.actor_logstd)
    return mean, std


def forward_critic(params: ActorCritic, obs: jnp.ndarray) -> jnp.ndarray:
    """Forward pass through critic network."""
    x = jnp.tanh(obs @ params.critic_w1)
    x = jnp.tanh(x @ params.critic_w2)
    value = x @ params.critic_out
    return value.squeeze(-1)


def sample_action(
    key: jnp.ndarray,
    params: ActorCritic,
    obs: jnp.ndarray,
    action_low: jnp.ndarray,
    action_high: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Sample action from policy and compute log probability."""
    mean, std = forward_actor(params, obs)

    # Sample from Gaussian
    noise = jax.random.normal(key, mean.shape)
    action_unbounded = mean + std * noise

    # Squash to action bounds using tanh
    action = action_low + (action_high - action_low) * (jnp.tanh(action_unbounded) + 1) / 2

    # Log probability (simplified, ignoring tanh correction for demo)
    log_prob = -0.5 * jnp.sum((action_unbounded - mean) ** 2 / (std ** 2) + 2 * jnp.log(std) + jnp.log(2 * jnp.pi))

    return action, log_prob


class Transition(NamedTuple):
    """Single transition for PPO training."""
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    log_prob: jnp.ndarray
    value: jnp.ndarray


class RolloutBuffer(NamedTuple):
    """Buffer of transitions for PPO update."""
    transitions: Transition
    last_value: jnp.ndarray


def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    last_value: jnp.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute Generalized Advantage Estimation."""
    # Append last value for bootstrapping
    values_ext = jnp.concatenate([values, last_value[None]])

    def scan_fn(carry, t):
        gae = carry
        delta = rewards[t] + gamma * values_ext[t + 1] * (1 - dones[t]) - values_ext[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        return gae, gae

    # Scan backwards through time
    _, advantages = jax.lax.scan(
        scan_fn,
        jnp.zeros_like(last_value),
        jnp.arange(len(rewards) - 1, -1, -1),
    )
    advantages = advantages[::-1]  # Reverse to get correct order

    returns = advantages + values
    return advantages, returns


# =============================================================================
# Training Loop
# =============================================================================


def create_train_state(key: jnp.ndarray, env: DistillationEnvJax):
    """Create initial training state."""
    obs_dim = env.observation_space_shape[0]
    action_dim = env.action_space_shape[0]

    params = init_actor_critic(key, obs_dim, action_dim)
    return params


def collect_rollout(
    key: jnp.ndarray,
    params: ActorCritic,
    env: DistillationEnvJax,
    env_state: EnvState,
    env_params: EnvParams,
    n_steps: int,
) -> Tuple[RolloutBuffer, EnvState, jnp.ndarray]:
    """Collect rollout data using vectorized environments."""
    action_low = env.action_space_low
    action_high = env.action_space_high

    def step_fn(carry, _):
        key, state = carry
        key, action_key, step_key = jax.random.split(key, 3)

        # Get observation
        obs = env._get_obs(state)

        # Sample action
        action, log_prob = sample_action(action_key, params, obs, action_low, action_high)

        # Get value estimate
        value = forward_critic(params, obs)

        # Step environment
        next_obs, next_state, reward, done, info = env.step(step_key, state, action, env_params)

        # Create transition
        transition = Transition(
            obs=obs,
            action=action,
            reward=reward,
            done=done,
            log_prob=log_prob,
            value=value,
        )

        # Reset environment if done
        reset_key, key = jax.random.split(key)
        next_state = jax.lax.cond(
            done,
            lambda: env.reset(reset_key, env_params)[1],
            lambda: next_state,
        )

        return (key, next_state), transition

    # Run rollout
    (key, final_state), transitions = jax.lax.scan(
        step_fn,
        (key, env_state),
        None,
        length=n_steps,
    )

    # Get final value for bootstrapping
    final_obs = env._get_obs(final_state)
    last_value = forward_critic(params, final_obs)

    buffer = RolloutBuffer(transitions=transitions, last_value=last_value)
    return buffer, final_state, key


def ppo_update(
    params: ActorCritic,
    buffer: RolloutBuffer,
    action_low: jnp.ndarray,
    action_high: jnp.ndarray,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    lr: float = 3e-4,
) -> Tuple[ActorCritic, dict]:
    """Perform PPO update on collected rollout."""
    transitions = buffer.transitions

    # Compute advantages and returns
    advantages, returns = compute_gae(
        transitions.reward,
        transitions.value,
        transitions.done,
        buffer.last_value,
    )

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def loss_fn(params):
        # Recompute policy outputs
        mean, std = forward_actor(params, transitions.obs)
        values = forward_critic(params, transitions.obs)

        # Compute current log probs
        action_unbounded = 2 * (transitions.action - action_low) / (action_high - action_low) - 1
        action_unbounded = jnp.arctanh(jnp.clip(action_unbounded, -0.999, 0.999))
        log_probs = -0.5 * jnp.sum(
            (action_unbounded - mean) ** 2 / (std ** 2) + 2 * jnp.log(std) + jnp.log(2 * jnp.pi),
            axis=-1,
        )

        # PPO clipped objective
        ratio = jnp.exp(log_probs - transitions.log_prob)
        clipped_ratio = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
        policy_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped_ratio * advantages))

        # Value loss
        value_loss = jnp.mean((values - returns) ** 2)

        # Entropy bonus
        entropy = jnp.mean(0.5 * jnp.log(2 * jnp.pi * jnp.e * std ** 2).sum(axis=-1))

        total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

        return total_loss, {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
        }

    # Compute gradients
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

    # Simple SGD update (could use Adam for better results)
    new_params = jax.tree.map(lambda p, g: p - lr * g, params, grads)

    metrics["total_loss"] = loss
    return new_params, metrics


def train_ppo(
    key: jnp.ndarray,
    env: DistillationEnvJax,
    n_envs: int = 64,
    n_steps: int = 128,
    n_updates: int = 100,
    lr: float = 3e-4,
) -> Tuple[ActorCritic, list]:
    """Train PPO agent with vectorized environments."""
    env_params = env.default_params

    # Initialize parameters
    key, init_key = jax.random.split(key)
    params = create_train_state(init_key, env)

    # Initialize vectorized environments
    key, reset_key = jax.random.split(key)
    reset_keys = jax.random.split(reset_key, n_envs)

    # Vectorized reset
    reset_fn = jax.vmap(env.reset, in_axes=(0, None))
    _, env_states = reset_fn(reset_keys, env_params)

    # Vectorized rollout collection
    def collect_single(key, params, env_state):
        return collect_rollout(key, params, env, env_state, env_params, n_steps)

    collect_fn = jax.vmap(collect_single, in_axes=(0, None, 0))

    # JIT compile training step
    @jax.jit
    def train_step(key, params, env_states):
        # Collect rollouts from all environments
        keys = jax.random.split(key, n_envs + 1)
        key, rollout_keys = keys[0], keys[1:]

        buffers, new_env_states, _ = collect_fn(rollout_keys, params, env_states)

        # Flatten batch dimensions for update
        flat_transitions = jax.tree.map(
            lambda x: x.reshape(-1, *x.shape[2:]) if x.ndim > 2 else x.reshape(-1),
            buffers.transitions,
        )
        flat_buffer = RolloutBuffer(
            transitions=flat_transitions,
            last_value=buffers.last_value.mean(),  # Average last values
        )

        # PPO update
        new_params, metrics = ppo_update(
            params,
            flat_buffer,
            env.action_space_low,
            env.action_space_high,
            lr=lr,
        )

        # Compute mean reward
        mean_reward = buffers.transitions.reward.mean()
        metrics["mean_reward"] = mean_reward

        return key, new_params, new_env_states, metrics

    # Training loop
    metrics_history = []

    print(f"\nTraining PPO with {n_envs} parallel environments...")
    print(f"Steps per update: {n_steps * n_envs:,}")
    print("-" * 50)

    start_time = time.time()
    total_steps = 0

    for update in range(n_updates):
        key, params, env_states, metrics = train_step(key, params, env_states)
        total_steps += n_steps * n_envs
        metrics_history.append(metrics)

        if (update + 1) % 10 == 0:
            elapsed = time.time() - start_time
            steps_per_sec = total_steps / elapsed
            print(
                f"Update {update + 1:3d} | "
                f"Reward: {float(metrics['mean_reward']):7.3f} | "
                f"Loss: {float(metrics['total_loss']):7.3f} | "
                f"Steps/sec: {steps_per_sec:,.0f}"
            )

    elapsed = time.time() - start_time
    final_steps_per_sec = total_steps / elapsed

    print("-" * 50)
    print(f"\nTraining complete!")
    print(f"Total steps: {total_steps:,}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Throughput: {final_steps_per_sec:,.0f} steps/sec")

    return params, metrics_history


def evaluate_policy(
    key: jnp.ndarray,
    params: ActorCritic,
    env: DistillationEnvJax,
    n_episodes: int = 10,
    max_steps: int = 200,
) -> dict:
    """Evaluate trained policy."""
    env_params = env.default_params
    action_low = env.action_space_low
    action_high = env.action_space_high

    episode_rewards = []
    final_x_D = []
    final_x_B = []

    for ep in range(n_episodes):
        key, reset_key = jax.random.split(key)
        obs, state = env.reset(reset_key, env_params)

        total_reward = 0.0
        for step in range(max_steps):
            key, action_key, step_key = jax.random.split(key, 3)

            # Get action from policy (deterministic for evaluation)
            mean, _ = forward_actor(params, obs)
            action = action_low + (action_high - action_low) * (jnp.tanh(mean) + 1) / 2

            obs, state, reward, done, info = env.step(step_key, state, action, env_params)
            total_reward += float(reward)

            if done:
                break

        episode_rewards.append(total_reward)
        final_x_D.append(float(info.x_D))
        final_x_B.append(float(info.x_B))

    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_x_D": np.mean(final_x_D),
        "mean_x_B": np.mean(final_x_B),
    }


def benchmark_throughput(env: DistillationEnvJax, n_envs_list: list = [1, 8, 32, 64, 128]):
    """Benchmark environment throughput for different batch sizes."""
    print("\nThroughput Benchmark")
    print("=" * 50)

    env_params = env.default_params
    action = jnp.array([5000.0, 3.0, 0.03, 0.03])  # Fixed action
    n_steps = 1000

    for n_envs in n_envs_list:
        # Reset environments
        key = jax.random.PRNGKey(0)
        reset_keys = jax.random.split(key, n_envs)
        reset_fn = jax.vmap(env.reset, in_axes=(0, None))
        _, states = reset_fn(reset_keys, env_params)

        # Batch actions
        actions = jnp.tile(action, (n_envs, 1))

        # Vectorized step
        step_fn = jax.vmap(env.step, in_axes=(0, 0, 0, None))

        @jax.jit
        def run_steps(states, base_key):
            def body(carry, _):
                states, key = carry
                # Split key for this step and next iteration
                key, subkey = jax.random.split(key)
                step_keys = jax.random.split(subkey, n_envs)
                _, new_states, _, _, _ = step_fn(step_keys, states, actions, env_params)
                return (new_states, key), None

            (final_states, _), _ = jax.lax.scan(body, (states, base_key), None, length=n_steps)
            return final_states

        # Warmup
        base_key = jax.random.PRNGKey(1)
        _ = run_steps(states, base_key)

        # Benchmark (block_until_ready ensures we measure actual compute time)
        start = time.time()
        final_states = run_steps(states, base_key)
        final_states.column_state.tray_T.block_until_ready()
        elapsed = time.time() - start

        total_steps = n_envs * n_steps
        steps_per_sec = total_steps / elapsed

        print(f"  {n_envs:3d} envs: {steps_per_sec:>12,.0f} steps/sec")

    print("=" * 50)


def main():
    print("=" * 60)
    print("Pure JAX RL Training for Distillation Column Control")
    print("=" * 60)

    # Create JAX-native environment
    env = DistillationEnvJax()

    print(f"\nEnvironment: DistillationEnvJax")
    print(f"  Observation shape: {env.observation_space_shape}")
    print(f"  Action shape: {env.action_space_shape}")

    # Benchmark throughput
    benchmark_throughput(env)

    # Train PPO
    key = jax.random.PRNGKey(42)
    trained_params, metrics = train_ppo(
        key,
        env,
        n_envs=64,
        n_steps=128,
        n_updates=20,  # Reduced for demo; increase for better training
        lr=3e-4,
    )

    # Evaluate trained policy
    print("\nEvaluating trained policy...")
    key = jax.random.PRNGKey(0)
    results = evaluate_policy(key, trained_params, env)

    print(f"\nEvaluation Results:")
    print(f"  Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Mean x_D: {results['mean_x_D']:.3f}")
    print(f"  Mean x_B: {results['mean_x_B']:.3f}")

    print("\n" + "=" * 60)
    print("PureJAX training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
