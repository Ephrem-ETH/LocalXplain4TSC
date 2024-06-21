import sklearn.metrics as skm
import numpy as np

def fidelity(lstm_preds, tree_preds):

  fidelity = skm.accuracy_score(lstm_preds, tree_preds)
  print(f"Fidelity: {fidelity:.3f}")
  return fidelity

def tree_node_depth(decision_tree):
  # Compute the interpretability metric
  depth = decision_tree.tree_.max_depth
  n_nodes = decision_tree.tree_.node_count
  print(f"Depth: {depth}")
  print(f"Number of nodes: {n_nodes}")
  
  return depth, n_nodes
  
  
def objective_evaluation(tree_model, lstm_preds, tree_preds):
  # Compute fidelity and tree node depth
  fidelity_score = fidelity(lstm_preds, tree_preds)
  depth, n_nodes = tree_node_depth(tree_model)
  
  return fidelity_score, depth, n_nodes

def compute_fidelity_accuracy(black_box_model, surrogate_model, true_labels, lime_explanations):
    # Obtain predictions from the black-box model and the surrogate model
    black_box_predictions = black_box_model.predict(lime_explanations)
    surrogate_predictions = surrogate_model.predict(lime_explanations)

    # Compute accuracy of the black-box model
    accuracy_black_box = skm.accuracy_score(true_labels, black_box_predictions)

    # Compute accuracy of the surrogate model
    accuracy_surrogate = skm.accuracy_score(true_labels, surrogate_predictions)

    # Compute fidelity of the surrogate model
    fidelity = skm.accuracy_score(black_box_predictions, surrogate_predictions)

    return accuracy_black_box, accuracy_surrogate, fidelity
  

def performance_decrease(model, test_data, replacement_method, important_timesteps):
    """
    Compute the decrease in model performance after perturbing important time steps.

    Parameters:
        model: Trained model for time series classification.
        test_data: Test data for evaluation (3D array).
        replacement_method: Method to replace important time steps ('zero', 'random', 'swap', 'mean').
        important_timesteps: Indices of important time steps for each instance.

    Returns:
        decrease_dict: Dictionary containing the decrease in accuracy for each instance in test data.
    """
    decrease_dict = {}

    # Original accuracy on test set
    original_accuracy = model.evaluate(test_data)[1]

    # Perturb important time steps based on replacement method
    perturbed_data = test_data.copy()  # Using test_data.copy() instead of np.copy(test_data)
    if replacement_method == 'zero':
        perturbed_data[:, :, important_timesteps] = 0
    elif replacement_method == 'random':
        for i, instance_important_timesteps in enumerate(important_timesteps):
            perturbed_data[i, :, instance_important_timesteps] = np.random.normal(
                np.mean(test_data[i, :, instance_important_timesteps]),
                np.std(test_data[i, :, instance_important_timesteps]),
                len(instance_important_timesteps)
            )
    elif replacement_method == 'swap':
        swap_indices = np.random.choice(
            [idx for idx in range(test_data.shape[1]) if idx not in important_timesteps[0]],
            len(important_timesteps[0]),
            replace=False
        )
        for i, instance_important_timesteps in enumerate(important_timesteps):
            perturbed_data[i, :, instance_important_timesteps] = test_data[i, :, swap_indices]
    elif replacement_method == 'mean':
        for i, instance_important_timesteps in enumerate(important_timesteps):
            for timestep_idx in instance_important_timesteps:
                perturbed_data[i, :, timestep_idx] = np.mean(test_data[i, :, timestep_idx])

    # Evaluate model performance on perturbed data
    perturbed_accuracies = model.evaluate(perturbed_data)[:, 1]

    # Compute decrease in accuracy
    decrease_dict = {i: original_accuracy - acc for i, acc in enumerate(perturbed_accuracies)}

    return decrease_dict
  
  