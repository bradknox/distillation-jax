"""Parameter estimation using JAX autodiff optimization.

This module provides tools for fitting simulator parameters to
plant data using gradient-based optimization with JAX.

Fittable parameters:
- Tray efficiency (global or sectional)
- Hydraulic time constants
- Heat loss coefficients
- Sensor bias/scale factors
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

try:
    from jaxopt import GradientDescent, LBFGS
    JAXOPT_AVAILABLE = True
except ImportError:
    JAXOPT_AVAILABLE = False

from scipy.optimize import minimize


@dataclass
class FittableParameter:
    """Specification for a fittable parameter.

    Attributes:
        name: Parameter name
        initial_value: Starting value for optimization
        lower_bound: Minimum allowed value
        upper_bound: Maximum allowed value
        scale: Scaling factor for optimization
        description: Human-readable description
    """

    name: str
    initial_value: float
    lower_bound: float = -np.inf
    upper_bound: float = np.inf
    scale: float = 1.0
    description: str = ""


@dataclass
class ParameterEstimationResult:
    """Result of parameter estimation.

    Attributes:
        initial_params: Starting parameter values
        fitted_params: Optimized parameter values
        initial_loss: Loss before optimization
        final_loss: Loss after optimization
        loss_reduction: Relative improvement
        n_iterations: Number of optimization iterations
        converged: True if optimization converged
        fit_metrics: Dict of fit quality metrics
        cross_validation: Cross-validation results if computed
    """

    initial_params: Dict[str, float]
    fitted_params: Dict[str, float]
    initial_loss: float
    final_loss: float
    loss_reduction: float
    n_iterations: int
    converged: bool
    fit_metrics: Dict[str, float]
    cross_validation: Optional[Dict] = None


def _create_loss_function(
    simulator: Callable,
    measurements: np.ndarray,
    inputs: np.ndarray,
    times: np.ndarray,
    measurement_weights: Optional[np.ndarray] = None,
) -> Callable:
    """Create a loss function for parameter fitting.

    Args:
        simulator: Function (params, inputs, times) -> outputs
        measurements: Measured outputs [n_times, n_outputs]
        inputs: Input trajectory [n_times, n_inputs]
        times: Time points [n_times]
        measurement_weights: Optional weights [n_outputs]

    Returns:
        Loss function that takes parameters and returns scalar loss.
    """
    if measurement_weights is None:
        measurement_weights = np.ones(measurements.shape[1])

    def loss_fn(params: Dict[str, float]) -> float:
        """Compute weighted MSE loss."""
        # Run simulator
        simulated = simulator(params, inputs, times)

        # Compute weighted MSE
        errors = simulated - measurements
        weighted_errors = errors * measurement_weights
        mse = np.mean(weighted_errors ** 2)

        return mse

    return loss_fn


def fit_parameters(
    parameters: List[FittableParameter],
    simulator: Callable,
    measurements: np.ndarray,
    inputs: np.ndarray,
    times: np.ndarray,
    method: str = "scipy",
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    measurement_weights: Optional[np.ndarray] = None,
    cross_validate: bool = False,
    cv_folds: int = 5,
) -> ParameterEstimationResult:
    """Fit parameters to match measurements.

    Args:
        parameters: List of FittableParameter specifications
        simulator: Simulation function (params, inputs, times) -> outputs
        measurements: Measured data [n_times, n_outputs]
        inputs: Input data [n_times, n_inputs]
        times: Time points [n_times]
        method: Optimization method ("scipy", "jaxopt_gd", "jaxopt_lbfgs")
        max_iterations: Maximum optimization iterations
        tolerance: Convergence tolerance
        measurement_weights: Optional measurement weights
        cross_validate: Whether to perform cross-validation
        cv_folds: Number of cross-validation folds

    Returns:
        ParameterEstimationResult with fitted parameters.
    """
    # Build initial parameter dict
    initial_params = {p.name: p.initial_value for p in parameters}

    # Build bounds
    bounds = [(p.lower_bound, p.upper_bound) for p in parameters]
    param_names = [p.name for p in parameters]

    # Create loss function
    loss_fn = _create_loss_function(
        simulator, measurements, inputs, times, measurement_weights
    )

    # Initial loss
    initial_loss = loss_fn(initial_params)

    # Convert dict-based loss to array-based for optimization
    def loss_array(x: np.ndarray) -> float:
        params = {name: x[i] for i, name in enumerate(param_names)}
        return loss_fn(params)

    x0 = np.array([initial_params[name] for name in param_names])

    # Optimize
    if method == "scipy":
        result = minimize(
            loss_array,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iterations, "ftol": tolerance},
        )
        x_opt = result.x
        n_iterations = result.nit
        converged = result.success

    elif method == "jaxopt_lbfgs" and JAXOPT_AVAILABLE:
        # Use JAXopt for gradient-based optimization
        def jax_loss(x):
            params = {name: x[i] for i, name in enumerate(param_names)}
            return loss_fn(params)

        solver = LBFGS(fun=jax_loss, maxiter=max_iterations, tol=tolerance)
        x_opt, state = solver.run(jnp.array(x0))
        x_opt = np.array(x_opt)
        n_iterations = state.iter_num
        converged = state.iter_num < max_iterations

    else:
        # Fallback to scipy
        result = minimize(
            loss_array,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iterations},
        )
        x_opt = result.x
        n_iterations = result.nit
        converged = result.success

    # Build fitted params dict
    fitted_params = {name: float(x_opt[i]) for i, name in enumerate(param_names)}

    # Final loss
    final_loss = loss_fn(fitted_params)

    # Loss reduction
    loss_reduction = (initial_loss - final_loss) / initial_loss if initial_loss > 0 else 0

    # Fit metrics
    simulated = simulator(fitted_params, inputs, times)
    residuals = measurements - simulated
    fit_metrics = {
        "mse": float(np.mean(residuals ** 2)),
        "rmse": float(np.sqrt(np.mean(residuals ** 2))),
        "mae": float(np.mean(np.abs(residuals))),
        "max_error": float(np.max(np.abs(residuals))),
        "r_squared": float(1 - np.var(residuals) / np.var(measurements)),
    }

    # Cross-validation
    cv_results = None
    if cross_validate and cv_folds > 1:
        cv_results = _cross_validate(
            parameters,
            simulator,
            measurements,
            inputs,
            times,
            cv_folds,
            method,
            max_iterations,
            tolerance,
        )

    return ParameterEstimationResult(
        initial_params=initial_params,
        fitted_params=fitted_params,
        initial_loss=float(initial_loss),
        final_loss=float(final_loss),
        loss_reduction=float(loss_reduction),
        n_iterations=int(n_iterations),
        converged=converged,
        fit_metrics=fit_metrics,
        cross_validation=cv_results,
    )


def _cross_validate(
    parameters: List[FittableParameter],
    simulator: Callable,
    measurements: np.ndarray,
    inputs: np.ndarray,
    times: np.ndarray,
    n_folds: int,
    method: str,
    max_iterations: int,
    tolerance: float,
) -> Dict:
    """Perform k-fold cross-validation.

    Args:
        parameters: Parameter specifications
        simulator: Simulation function
        measurements: Full measurement data
        inputs: Full input data
        times: Full time array
        n_folds: Number of CV folds
        method: Optimization method
        max_iterations: Max iterations per fold
        tolerance: Convergence tolerance

    Returns:
        Dict with CV metrics.
    """
    n_samples = len(times)
    fold_size = n_samples // n_folds
    indices = np.arange(n_samples)

    train_losses = []
    val_losses = []

    for fold in range(n_folds):
        # Split data
        val_start = fold * fold_size
        val_end = val_start + fold_size
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

        train_meas = measurements[train_idx]
        train_inputs = inputs[train_idx]
        train_times = times[train_idx] - times[train_idx[0]]

        val_meas = measurements[val_idx]
        val_inputs = inputs[val_idx]
        val_times = times[val_idx] - times[val_idx[0]]

        # Fit on training set
        result = fit_parameters(
            parameters,
            simulator,
            train_meas,
            train_inputs,
            train_times,
            method=method,
            max_iterations=max_iterations,
            tolerance=tolerance,
            cross_validate=False,
        )

        train_losses.append(result.final_loss)

        # Evaluate on validation set
        val_sim = simulator(result.fitted_params, val_inputs, val_times)
        val_loss = float(np.mean((val_sim - val_meas) ** 2))
        val_losses.append(val_loss)

    return {
        "n_folds": n_folds,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "mean_train_loss": np.mean(train_losses),
        "mean_val_loss": np.mean(val_losses),
        "std_val_loss": np.std(val_losses),
    }


def print_parameter_estimation_report(result: ParameterEstimationResult) -> None:
    """Print a formatted parameter estimation report.

    Args:
        result: ParameterEstimationResult from fit_parameters.
    """
    print("=" * 60)
    print("PARAMETER ESTIMATION REPORT")
    print("=" * 60)

    print(f"\nConverged: {'Yes' if result.converged else 'No'}")
    print(f"Iterations: {result.n_iterations}")

    print("\nParameter Values:")
    print(f"{'Parameter':<20} {'Initial':>12} {'Fitted':>12} {'Change':>12}")
    print("-" * 56)
    for name in result.initial_params:
        init = result.initial_params[name]
        fitted = result.fitted_params[name]
        change = (fitted - init) / init * 100 if init != 0 else 0
        print(f"{name:<20} {init:>12.4f} {fitted:>12.4f} {change:>+11.1f}%")

    print("\nLoss:")
    print(f"  Initial: {result.initial_loss:.6f}")
    print(f"  Final:   {result.final_loss:.6f}")
    print(f"  Reduction: {result.loss_reduction * 100:.1f}%")

    print("\nFit Metrics:")
    for metric, value in result.fit_metrics.items():
        print(f"  {metric}: {value:.6f}")

    if result.cross_validation:
        cv = result.cross_validation
        print(f"\nCross-Validation ({cv['n_folds']} folds):")
        print(f"  Mean train loss: {cv['mean_train_loss']:.6f}")
        print(f"  Mean val loss:   {cv['mean_val_loss']:.6f}")
        print(f"  Std val loss:    {cv['std_val_loss']:.6f}")

    print("=" * 60)
