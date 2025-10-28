"""
Test the best equation from the last sample_order in samples_best.json on test_id and test_ood datasets
"""
import numpy as np
from scipy.optimize import minimize
import json
import re

# Import test datasets directly to avoid dependency issues
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'llm4ad', 'task', 'science_discovery', 'oscillator1'))
import test_id
import test_odd


def load_best_equation_from_json(json_path: str):
    """
    Load the function from the last sample_order in samples_best.json
    
    Args:
        json_path: Path to the samples_best.json file
        
    Returns:
        tuple: (sample_order, function_code, score, equation_function)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    if not samples:
        raise ValueError("No samples found in JSON file")
    
    # Get the last sample
    last_sample = samples[-1]
    sample_order = last_sample['sample_order']
    function_code = last_sample['function']
    score = last_sample['score']
    
    # Extract the function body and create a callable function
    # The function code includes the def statement, so we need to execute it
    local_namespace = {'np': np}
    exec(function_code, local_namespace)
    equation_func = local_namespace['equation']
    
    return sample_order, function_code, score, equation_func


# Global variables to store equation info (will be loaded in main)
equation = None
equation_info = {}


def evaluate_on_dataset(data_dict: dict, equation_func: callable, dataset_name: str, max_params: int = 10):
    """
    Evaluate the equation on a given dataset
    
    Args:
        data_dict: Dictionary containing 'x', 'v', 'a' keys with data values
        equation_func: The equation function to evaluate
        dataset_name: Name of the dataset for display
        max_params: Maximum number of parameters for optimization
        
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
    print(f"\nOptimizing parameters (max_params={max_params})...")
    initial_params = [1.0] * max_params
    
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
    # Path to the samples_best.json file
    json_path = os.path.join('logs', 'funsearch', '20251028_072122', 'samples', 'samples_best.json')
    
    # Load the best equation from JSON
    print("="*60)
    print("Loading Best Equation from samples_best.json")
    print("="*60)
    
    sample_order, function_code, original_score, equation_func = load_best_equation_from_json(json_path)
    
    print(f"\nLoaded equation from sample_order: {sample_order}")
    print(f"Original score from FunSearch: {original_score:.10f}")
    print(f"\nFunction code:")
    print("-" * 60)
    print(function_code)
    # save the function code to a file
    # with open('best_equation.py', 'w') as f:
    #     f.write(function_code)
    print("-" * 60)
    
    # Determine the number of parameters needed
    # Extract param count from function code (look for params[N])
    import re
    param_indices = re.findall(r'params\[(\d+)\]', function_code)
    if param_indices:
        max_param_index = max(int(idx) for idx in param_indices)
        max_params = max_param_index + 1
    else:
        # Check for P0, P1, ... style parameters
        param_indices = re.findall(r'P(\d+)', function_code)
        if param_indices:
            max_param_index = max(int(idx) for idx in param_indices)
            max_params = max_param_index + 1
        else:
            max_params = 10  # Default
    
    print(f"\nDetected number of parameters: {max_params}")
    
    # Test on test_id (In-Distribution)
    result_id = evaluate_on_dataset(test_id.data, equation_func, 
                                     "test_id.py (In-Distribution)", 
                                     max_params=max_params)
    
    # Test on test_odd (Out-of-Distribution)
    result_ood = evaluate_on_dataset(test_odd.data, equation_func, 
                                      "test_odd.py (Out-of-Distribution)", 
                                      max_params=max_params)
    
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
    output_file = f'test_results_sample{sample_order}.json'
    results = {
        'sample_order': sample_order,
        'original_funsearch_score': original_score,
        'function_code': function_code,
        'max_params': max_params,
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

