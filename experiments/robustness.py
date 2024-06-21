import numpy as np
import sklearn.metrics as skm
from utils.test_dataloader import test_dataloader
from globXplain import GlobXplain4TSC

class Robustness_Evaluation:
  def __init__(self, learner, base_dir):
    self.base_dir = base_dir
    self.globxplain = GlobXplain4TSC(base_dir=base_dir)
    self.learner=learner
  def evaluate_XAI_robustness( self, X_test, y_test, class_names, perturbation_method=np.random.normal, perturbation_strength=0.1, random_seed=42):

    # Perturb the data by adding random noise
    X_test_perturbed = X_test + np.random.normal(0, perturbation_strength, X_test.shape)

    # Dataloader for test and perturbed data
    test_dl = test_dataloader(self.learner, X_test, y_test)
    test_dl_perturbed  =  test_dataloader(self.learner, X_test_perturbed, y_test)



    # Evaluate model perfromance on the original data
    test_probas, test_targets, test_preds = self.learner.get_preds(dl=test_dl, with_decoded=True, save_preds=None, save_targs=None)

    # Evaluate model performance on the perturbed data
    test_probas_perturbed, test_targets_perturbed, test_preds_perturbed = self.learner.get_preds(dl=test_dl_perturbed, with_decoded=True, save_preds=None, save_targs=None)

    # Compute the accuracy
    print(f'Accuracy on original test set: {skm.accuracy_score(test_targets, test_preds):10.6f}')
    print(f'Accuracy after perturbation: {skm.accuracy_score(test_targets_perturbed, test_preds_perturbed):10.6f}')

    # Apply the XAI method
    final_data, kmeans_dict, scaler_dict, cluster_centroids_origi, master_dict_origi = self.globxplain.combine_data(X_test)
    final_data_perturbed, kmeans_dict_perturbed, scaler_dict_perturbed, cluster_centroids_perturb, master_dict_perturb = self.globxplain.combine_data(X_test_perturbed)

    # Train the surrogate (linear) models

    print(f" Linear models performance on  perturbed test set")
    important_features_perturb, perturb_preds, _, _, _=self.globxplain.main(full_data=final_data_perturbed, lstm_preds=test_preds_perturbed, class_names=class_names, X_test=X_test, cluster_centroids=cluster_centroids_perturb, master_dict=master_dict_perturb, only_dt=True, ccp_alpha=None)

    print(f" Linear models performance on original test set")
    important_features_origi, origi_preds, _, _, _=self.globxplain.main(full_data=final_data, lstm_preds=test_preds, class_names=class_names, X_test=X_test, cluster_centroids=cluster_centroids_origi, master_dict= master_dict_origi, only_dt=True, ccp_alpha=None)

    # Robustness
    # print(f" {origi_test_acc} - {perturb_test_acc} = {(origi_test_acc - perturb_test_acc)}")round((origi_test_acc - perturb_test_acc) /len_dt_test , 3)
    robustness_score = sum(origi_preds==perturb_preds)/ len(origi_preds)
    print(f"Robustness of the method: {robustness_score:.2f}")
    return important_features_origi, cluster_centroids_origi, robustness_score