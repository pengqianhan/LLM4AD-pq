"""
Test the best equation (sample_order 6) on test_id and test_ood datasets
"""
import numpy as np
from scipy.optimize import minimize
import json

# Import test datasets directly to avoid dependency issues
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'llm4ad', 'task', 'science_discovery', 'oscillator1'))
import test_id
import test_odd


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """ 
    Best equation from sample_order 6
    Mathematical function for acceleration in a damped nonlinear oscillator
    
    Args:
        x: A numpy array representing observations of current position.
        v: A numpy array representing observations of velocity.
        params: Array of numeric constants or parameters to be optimized

    Return:
        A numpy array representing acceleration as the result of applying 
        the mathematical function to the inputs.
    """
    return (params[0] * np.sin(params[1] * x + params[2]) + 
            params[3] * np.sin(params[4] * v + params[5]) + 
            params[6] * x * v + 
            params[7] * np.exp(-params[8] * x**2) + 
            params[9])


def evaluate_on_dataset(data_dict: dict, equation_func: callable, dataset_name: str):
    """
    Evaluate the equation on a given dataset
    
    Args:
        data_dict: Dictionary containing 'x', 'v', 'a' keys with data values
        equation_func: The equation function to evaluate
        dataset_name: Name of the dataset for display
        
    Returns:
        Dictionary with optimization results
    """
    print(f"\n{'='*60}")
    print(f"Testing on {dataset_name}")
    print(f"{'='*60}")
    
    # Convert data to numpy arrays
    data = []
    for point in data_dict:
        data.append([point['x'], point['v'], point['a']])
    data = np.array(data)
    
    x = data[:, 0]
    v = data[:, 1]
    a_true = data[:, 2]
    
    print(f"Dataset size: {len(data)} points")
    print(f"x range: [{x.min():.4f}, {x.max():.4f}]")
    print(f"v range: [{v.min():.4f}, {v.max():.4f}]")
    print(f"a range: [{a_true.min():.4f}, {a_true.max():.4f}]")
    
    # Define loss function (MSE)
    def loss(params):
        try:
            a_pred = equation_func(x, v, params)
            mse = np.mean((a_pred - a_true) ** 2)
            return mse
        except Exception as e:
            print(f"Error in loss calculation: {e}")
            return 1e10
    
    # Optimize parameters
    print("\nOptimizing parameters...")
    MAX_NPARAMS = 10
    initial_params = [1.0] * MAX_NPARAMS
    
    result = minimize(loss, initial_params, method='BFGS', options={'maxiter': 1000})
    
    # Calculate metrics
    optimized_params = result.x
    final_mse = result.fun
    final_score = -final_mse  # Score is negative MSE
    
    # Calculate additional metrics
    a_pred_optimized = equation_func(x, v, optimized_params)
    mae = np.mean(np.abs(a_pred_optimized - a_true))
    rmse = np.sqrt(final_mse)
    
    # Calculate R² score
    ss_res = np.sum((a_true - a_pred_optimized) ** 2)
    ss_tot = np.sum((a_true - np.mean(a_true)) ** 2)
    r2_score = 1 - (ss_res / ss_tot)
    
    # Print results
    print(f"\nOptimization completed: {result.message}")
    print(f"Success: {result.success}")
    print(f"Iterations: {result.nit}")
    
    print(f"\n{'-'*60}")
    print("PERFORMANCE METRICS")
    print(f"{'-'*60}")
    print(f"Score (negative MSE): {final_score:.10f}")
    print(f"MSE:                  {final_mse:.10f}")
    print(f"RMSE:                 {rmse:.10f}")
    print(f"MAE:                  {mae:.10f}")
    print(f"R² Score:             {r2_score:.6f}")
    
    print(f"\n{'-'*60}")
    print("OPTIMIZED PARAMETERS")
    print(f"{'-'*60}")
    for i, param in enumerate(optimized_params):
        print(f"params[{i}] = {param:.6f}")
    
    return {
        'dataset_name': dataset_name,
        'dataset_size': len(data),
        'x_range': [float(x.min()), float(x.max())],
        'v_range': [float(v.min()), float(v.max())],
        'score': float(final_score),
        'mse': float(final_mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2_score': float(r2_score),
        'optimized_params': optimized_params.tolist(),
        'optimization_success': result.success,
        'optimization_iterations': result.nit
    }


def main():
    print("="*60)
    print("Testing Best Equation (Sample Order 6)")
    print("="*60)
    print("\nEquation:")
    print("a = params[0] * sin(params[1] * x + params[2])")
    print("  + params[3] * sin(params[4] * v + params[5])")
    print("  + params[6] * x * v")
    print("  + params[7] * exp(-params[8] * x²)")
    print("  + params[9]")
    
    # Test on test_id (In-Distribution)
    result_id = evaluate_on_dataset(test_id.data, equation, "test_id.py (In-Distribution)")
    
    # Test on test_odd (Out-of-Distribution)
    result_ood = evaluate_on_dataset(test_odd.data, equation, "test_odd.py (Out-of-Distribution)")
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}")
    print(f"\n{'Metric':<25} {'test_id (ID)':<20} {'test_odd (OOD)':<20}")
    print(f"{'-'*65}")
    print(f"{'Score (neg MSE)':<25} {result_id['score']:<20.10f} {result_ood['score']:<20.10f}")
    print(f"{'MSE':<25} {result_id['mse']:<20.10f} {result_ood['mse']:<20.10f}")
    print(f"{'RMSE':<25} {result_id['rmse']:<20.10f} {result_ood['rmse']:<20.10f}")
    print(f"{'MAE':<25} {result_id['mae']:<20.10f} {result_ood['mae']:<20.10f}")
    print(f"{'R² Score':<25} {result_id['r2_score']:<20.6f} {result_ood['r2_score']:<20.6f}")
    
    # Calculate performance degradation
    if result_id['mse'] > 0:
        mse_degradation = ((result_ood['mse'] - result_id['mse']) / result_id['mse']) * 100
        print(f"\nMSE degradation (ID -> OOD): {mse_degradation:+.2f}%")
    
    # Save results to JSON
    output_file = 'test_results_sample6.json'
    results = {
        'equation_description': 'Sample order 6 from samples_best.json',
        'test_id': result_id,
        'test_odd': result_ood
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

