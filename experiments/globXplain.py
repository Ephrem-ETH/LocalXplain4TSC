# import all necessary libraries

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from experiments.utils.helper_class import *
from experiments.utils.helper_class import *
from experiments.metrics import *
from experiments.perturbation import *
from experiments.utils.test_dataloader import *

# For Sever
# from utils.helper_class import *
# from metrics import *
# from perturbation import *
# from utils.test_dataloader import *



class GlobXplain4TSC:
  def __init__(self, base_dir):
    self.base_dir= base_dir
    self.helper_instance = HelperClass(base_dir=base_dir) 

  # Turn the data of 2D shape into 3D
  def preprocessing_data(self, X):
    print(f'Shape of data : {X.shape}')

    # Create sample input data
    # data = np.random.rand(90, 24, 51)

    # Reshape input data into a 2D array
    reshaped_data = np.empty((X.shape[0], X.shape[1]), dtype=np.ndarray)
    for i in range(X.shape[0]):
      for j in range(X.shape[1]):
          reshaped_data[i, j] = X[i, j, :]

    # Create a list of column names for the DataFrame
    col_names = [f"ch{i+1}" for i in range(X.shape[1])]


    # Create the DataFrame
    df = pd.DataFrame(reshaped_data, columns=col_names)

    # Print the resulting DataFrame
    # print(df)
    return df


  def global_feature_extraction(self, df):
    # Compute the mean of each cell
    # mean_df = df.applymap(lambda x: np.mean(x))
    # mean_df = mean_df.round(7)
    reshaped_data = df.reshape((df.shape[0], df.shape[2]))

  # Compute the mean across all time series for each time step
    mean_df = np.mean(reshaped_data, axis=0)
    # Create a DataFrame with dynamic column names for each channel
    num_channels = df.shape[1]
    channel_columns = [f'ch{i+1}' for i in range(num_channels)]
    mean_df = pd.DataFrame(mean_df, columns=channel_columns)
    # Define the dynamic column name pattern
    col_name = 'global_feature'

    # Create a dictionary to map the original column names to the new column names
    column_mapping = {col: f'{col_name}_{col}' for col in mean_df.columns}

    # Rename the columns using the dictionary
    mean_df = mean_df.rename(columns=column_mapping)

    return mean_df



  def extract_inc_dec_events(self, data):
    # Reshape input data into a 2D array
    reshaped_data = np.empty((data.shape[0], data.shape[1]), dtype=np.ndarray)
    for i in range(data.shape[0]):
      for j in range(data.shape[1]):
          reshaped_data[i, j] = data[i, j, :]

    # Create a list of column names for the DataFrame
    col_names = [f"ch{i+1}" for i in range(data.shape[1])]


    # Create the DataFrame
    df = pd.DataFrame(reshaped_data, columns=col_names)

    # Initialize the result DataFrame
    result_df = pd.DataFrame(index=df.index)
    events_dict = {}
    # Extract increasing and decreasing events for each column of each instance
    for col_name in df.columns:
      increasing_events = []
      decreasing_events = []

      for instance_idx in range(len(df)):
          col_values = df.loc[instance_idx, col_name]
          inc_start_time = 0
          dec_start_time = 0
          inc_duration = 0
          dec_duration = 0
          inc_events = []
          dec_events = []
          inc_sum_values = 0
          dec_sum_values = 0
          for i in range(1, len(col_values)):

              if col_values[i] > col_values[i-1]:
                  if dec_duration > 0:
                      dec_avg_value = dec_sum_values / dec_duration
                      dec_events.append([dec_start_time, dec_duration, dec_avg_value])
                      dec_duration = 0
                      dec_sum_values = col_values[i]
                  if inc_duration == 0:
                      inc_start_time = i
                  inc_duration += 1
                  inc_sum_values += col_values[i]
              elif col_values[i] < col_values[i-1]:
                  if inc_duration > 0:
                      inc_avg_value = inc_sum_values / inc_duration
                      inc_events.append([ inc_start_time, inc_duration, inc_avg_value])
                      inc_duration = 0
                      inc_sum_values = col_values[i]
                  if dec_duration == 0:
                      dec_start_time = i
                  dec_duration += 1
                  dec_sum_values += col_values[i]
          if inc_duration > 0:
              inc_avg_value = inc_sum_values / inc_duration
              inc_events.append([inc_start_time, inc_duration, inc_avg_value])
          if dec_duration > 0:
              dec_avg_value = dec_sum_values / dec_duration
              dec_events.append([dec_start_time, dec_duration, dec_avg_value])
          increasing_events.append(inc_events)
          decreasing_events.append(dec_events)
      # print(pd.Series(inc_events))
      result_df[f"Increasing_{col_name}"] = increasing_events
      result_df[f"Decreasing_{col_name}"] = decreasing_events

    # Display the resulting DataFrame
    # print(type(result_df))
    return result_df


  def extract_local_max_min_events(self, data):
    # Reshape input data into a 2D array
    reshaped_data = np.empty((data.shape[0], data.shape[1]), dtype=np.ndarray)
    for i in range(data.shape[0]):
      for j in range(data.shape[1]):
          reshaped_data[i, j] = data[i, j, :]

    # Create a list of column names for the DataFrame
    col_names = [f"ch{i+1}" for i in range(data.shape[1])]


    # Create the DataFrame
    df = pd.DataFrame(reshaped_data, columns=col_names)

    # Initialize the result DataFrame
    result_df = pd.DataFrame(index=df.index)
    events_dict = {}
    # Extract local max and local min events for each column of each instance
    for col_name in df.columns:
      local_max_events = []
      local_min_events = []
      for instance_idx in range(len(df)):
          col_values = df.loc[instance_idx, col_name]
          max_events = []
          min_events = []
          for i in range(1, len(col_values)-1):
              if col_values[i] > col_values[i-1] and col_values[i] > col_values[i+1]:
                  max_events.append([i, col_values[i]])
              elif col_values[i] < col_values[i-1] and col_values[i] < col_values[i+1]:
                  min_events.append([i, col_values[i]])

          local_max_events.append(max_events)
          local_min_events.append(min_events)

      result_df[f"LocalMax_{col_name}"] = local_max_events
      result_df[f"LocalMin_{col_name}"] = local_min_events

    # Display the resulting DataFrame
    # print(result_df)
    return result_df

  def flatten_nested_events(self, events_list):
    inner_values = [inner for row in events_list for inner in row]
    inner_values_2d = np.array(inner_values, dtype=object).tolist()
    return inner_values_2d

  def cluster_events(self, all_events, k=30, col_name=""):

      silhouette_scores = []
      sse = []
      
      # Preprocess the data
      data = np.array(all_events)
      scaler= StandardScaler()
      data_transformed = scaler.fit_transform(data)
      
      
      max_stable_iterations = 3  # Maximum number of consecutive iterations with stable silhouette score
      stable_count = 0

      for n_clusters in range(2,k):


        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data_transformed)
        labels = kmeans.labels_
        silhouette_avg = silhouette_score(data_transformed, labels)
        silhouette_scores.append(silhouette_avg)
        optimal_k = np.argmax(silhouette_scores) + 2

        # print(optimal_k)
        sse.append(kmeans.inertia_)
        # print(labels)
         # Check for stability in silhouette scores
        if len(silhouette_scores) >= max_stable_iterations:
            if np.all(np.diff(silhouette_scores[-max_stable_iterations:]) == 0):
                stable_count += 1
            else:
                stable_count = 0
        
        # Break the loop if silhouette scores have been stable for max_stable_iterations
        if stable_count >= max_stable_iterations:
            break

      # Fit the kmeans with optimal K value
      kmeans = KMeans(n_clusters=optimal_k, random_state=12).fit(data_transformed)
      labels = kmeans.labels_
      centroids = kmeans.cluster_centers_

      plt.plot(range(2, k), silhouette_scores, marker='o')
      plt.xlabel('Number of clusters', fontsize=12)
      plt.ylabel('Silhouette score', fontsize=12)
      plt.title('Silhouette Method', fontsize=12)
      # Add grid lines
      plt.grid(True)

      # Customize tick labels
      plt.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)


      plt.savefig(f'{self.base_dir}/{col_name}_silhouette_plot.png', dpi=300)
      # Save the plot to a PDF file (vector format)
      plt.savefig(f'{self.base_dir}/{col_name}_silhouette_plot.pdf', format='pdf', dpi=300, bbox_inches='tight')
      plt.show()

      # Plot the sum of squared distances against k
      plt.plot(range(2, k), sse)
      plt.xlabel('Number of clusters', fontsize=12)
      plt.ylabel('Sum of squared distances', fontsize=12)
      plt.title('Elbow Method', fontsize=12)
      # Add grid lines
      plt.grid(True)

      # Customize tick labels
      plt.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
      plt.savefig(f'{self.base_dir}/{col_name}_elbow_plot.png')
      plt.savefig(f'{self.base_dir}/{col_name}_elbow_plot.pdf', format='pdf', dpi=300, bbox_inches='tight')

      plt.show()
      return kmeans, scaler

  # Function to perform event attribution and mapping the extracted events to the clusters.
  def event_attribution(self, kmeans, parametrized_events, scaler):
      # Initialize  the clusters
    n_clusters = len(kmeans.cluster_centers_)

    # clusters = {i: [] for i in range(n_clusters)}
    rows = {}
    for i,row in enumerate(parametrized_events):
        # clusters.clear()
        # key = "cls_{}".format(i)
        clusters = {i: [] for i in range(n_clusters)}
        for event in row:

            event_scaled = scaler.transform([event])
            # print(f"{event}  -> {event_scaled}")
            label = kmeans.predict(event_scaled)[0]
            # print(label)
            for cluster_label, cluster_list in clusters.items():
                if cluster_label == label:
                    cluster_list.append('yes')
                    # print(clusters)

                else:
                    cluster_list.append('no')
                    # print(clusters)
            rows[i] = clusters




    cluster_event_counts = {}
    for key, values in rows.items():
      yes_prob = []
      for k, value in values.items():
        counter_list  = Counter(value)
        count_no = counter_list['no']
        count_yes = counter_list['yes']
        # print((count_yes/count_no))
        # print(counter_list)
        # yes_prob.append(round(count_yes/(count_yes + count_no),5))
        yes_prob.append(count_yes)
      cluster_event_counts[key] = yes_prob
    return cluster_event_counts

  def merge_event_df(self, df_inc_dec, df_max_min):

    # Merge the DataFrames by index
    merged_df = df_inc_dec.merge(df_max_min, left_index=True, right_index=True)
    merged_df.head()
    # merged_df = df_max_min.copy()
    return merged_df

  def prepare_data4DT(self, merged_df, k=20, kmeans_dict=None, scaler_dict=None, for_eval=False):
  #   helper_instance = HelperClass(base_dir="results")

    appended_df = pd.DataFrame()
    count = 0
    master_dict = {}
    cluster_centroids = {}
    if kmeans_dict is None:
      kmeans_dict = {}
      scaler_dict = {}
    for col_name in merged_df.columns:
      parametrized_events = merged_df[col_name]
      # print(type(col_name))
      flatten_data =  self.flatten_nested_events(parametrized_events)
      # Extract the part of the column name before the second underscore
      col_prefix = "_".join(col_name.split("_")[:2])
      if for_eval:
        # print(kmeans_dict[col_name])
        # print(kmeans_dict[f'{col_name}'])
        kmeans = kmeans_dict[col_name]
        scaler = scaler_dict[col_name]

        attributed_data = self.event_attribution(kmeans, parametrized_events, scaler)
      else:
        kmeans, scaler = self.cluster_events(flatten_data, k, col_name=col_name)
        kmeans_dict[col_name] = kmeans
        scaler_dict[col_name] = scaler
        attributed_data = self.event_attribution(kmeans, parametrized_events, scaler)
      # Determine the maximum length of the values
      max_length = max(len(values) for values in attributed_data.values())

      # Generate dynamic column names
      column_names = [f"{col_name}_c{i+1}" for i in range(max_length)]

      # Convert the dictionary to DataFrame with dynamic column names
      df = pd.DataFrame(attributed_data.values(), columns=column_names)
      appended_df = pd.concat([appended_df, df], axis=1)
      post_processed_col_name, cluster_centroid = self.helper_instance.post_processed(kmeans, scaler, col_name)
      master_dict.update(post_processed_col_name)
      cluster_centroids.update(cluster_centroid)
      if 'increasing' in col_name.lower() or 'decreasing' in col_name.lower():
        self.helper_instance.plot_3D(kmeans=kmeans, flatten_events=flatten_data, scaler=scaler, col_name=col_name)
      elif 'localmax' in col_name.lower() or 'localmin' in col_name.lower():
        self.helper_instance.plot_2D(kmeans=kmeans, flatten_events=flatten_data, scaler=scaler, col_name=col_name)
    # Save the dictionaries containing all models
    joblib.dump(kmeans_dict, f'{self.base_dir}/kmeans_models_dict.pkl')
    joblib.dump(scaler_dict, f'{self.base_dir}/scaler_models_dict.pkl')

    return appended_df, master_dict, cluster_centroids, kmeans_dict, scaler_dict



  def combine_data(self, X_test, kmeans_dict=None, scaler_dict=None, for_eval=False):
    df = self.preprocessing_data(X_test)
    df_inc_dec = self.extract_inc_dec_events(X_test)
    df_max_min = self.extract_local_max_min_events(X_test)
    merged_df = self.merge_event_df(df_inc_dec=df_inc_dec, df_max_min=df_max_min)
    if for_eval:
      # print(kmeans_dict)
      kmeans_dict = kmeans_dict
      scaler_dict = scaler_dict
      appended_df, master_dict, cluster_centroids, kmeans_dict, scaler_dict = self.prepare_data4DT(merged_df, kmeans_dict=kmeans_dict, scaler_dict=scaler_dict, for_eval=True)
      # print(kmeans_dict)
    else:
      appended_df, master_dict, cluster_centroids, kmeans_dict, scaler_dict = self.prepare_data4DT(merged_df)
    # master_dict = self.helper_instance.update_master_dict(master_dict)
    full_data = appended_df.copy()

    # Convert 0 to False and non-zero to True
    # full_data = full_data.astype(bool)

    # Convert False to 0 and True to 1
    # full_data = full_data.astype(int)
    # Global feature calculation
    # mean_df = global_feature_extraction(X_test)

    # full_data = pd.concat([full_data, mean_df], axis=1)
    # full_data['y'] = lstm_preds
    # tree_model = apply_dt(full_data)

    # # objective evaluation
    # data = full_data.drop('y', axis=1)
    # tree_preds = tree_model.predict(data)
    # objective_evaluation(tree_model, lstm_preds, tree_preds)
    return full_data, kmeans_dict, scaler_dict, cluster_centroids, master_dict
    
    
  def apply_lr(self, perturbed_instances, target, weights, class_names, master_dict, model_regressor):
      star = 5 * '*'
      print(f' {star}LR Model Accuracy {star}')
      
      
      # Train a linear regression model
      # model_regressor = Ridge(alpha=1, fit_intercept=True, random_state=12)
      if model_regressor is None:
        model_regressor = Ridge()
      elif model_regressor == 'LinearRegression':
        model_regressor = LinearRegression()
      elif model_regressor == 'Lasso':
        model_regressor = Lasso()
        
      easy_model= model_regressor
      easy_model.fit(perturbed_instances, target, sample_weight= weights)
      
      prediction_score = easy_model.score(perturbed_instances, target, sample_weight= weights)
      
      local_pred = easy_model.predict(perturbed_instances.iloc[0].values.reshape(1, -1))

      # Get feature importance scores
      feature_importance = easy_model.coef_
      feature_names = [master_dict.get(col) for col in perturbed_instances.columns]

      # Plot the feature importance
      plt.figure(figsize=(8, 12), dpi=300)
      plt.barh(feature_names, feature_importance, color='skyblue')
      plt.xlabel('Feature Importance', fontsize=12)
      plt.ylabel('Features', fontsize=12)
      plt.title('Linear Regression Feature Importance', fontsize=14)
      plt.xticks(fontsize=10)  # Adjust font size of x-axis labels
      plt.yticks(fontsize=10)  # Adjust font size of y-axis labels
      plt.gca().invert_yaxis()  # Invert y-axis for better readability
      # Add grid lines for clarity
      plt.grid(axis='x', linestyle='--', alpha=0.7)

      # Save the plot in high quality
      plt.savefig(f'{self.base_dir}/lr_fi_plot.png', dpi=300, bbox_inches='tight')
      # Save the plot in PDF format
      plt.savefig(f'{self.base_dir}/lr_fi_plot.pdf', format='pdf', dpi=300, bbox_inches='tight')
      plt.show()
      
      # Evaluate local fidelity
      print(f"R-Score (Local Fidelity): {np.round(prediction_score,2)}")
      print(f" Local Prediction: {np.round(local_pred,2)}")



      return easy_model, feature_importance, prediction_score, local_pred


  def main_lr(self, full_data, lstm_preds, weights, class_names, X, cluster_centroids, master_dict, model_regressor=None):
    # full_data['y'] = lstm_preds
    
    model_lr, important_features_lr, prediction_score, local_pred = self.apply_lr(full_data, lstm_preds, weights, class_names, master_dict, model_regressor)
    # objective evaluation
    # data = full_data.drop('y', axis=1)
    # tree_preds = tree_model.predict(full_data)
    # fidelity_score, depth, n_nodes = objective_evaluation(tree_model, lstm_preds, tree_preds)
    # self.helper_instance.plot_events_on_timeseries(X, lstm_preds, important_features_lr, cluster_centroids, class_names)
    # self.helper_instance.plot_events_as_line_on_timeseries_1(X, lstm_preds, important_features_lr, cluster_centroids, class_names)
    # self.helper_instance.plot_events_as_line_on_timeseries1(X_test, lstm_preds, important_features_lr, cluster_centroids, class_names)
    # print(important_features_lr)
    return important_features_lr, prediction_score, local_pred
  
    
  def ts_local_explanation(self, origi_instance, learner, num_perturbations=1000, n_clusters=20, kernel_width=None, top_n=10, class_names=[0,1], model_regressor=None, replacement_method='mean'):
      print(f'{origi_instance.shape[1]}')

      perturbed_list, distances = generate_perturbations(origi_instance[0], num_perturbations, replacement_method=replacement_method)
      perturbed_instances = np.array(perturbed_list).reshape((-1, 1, origi_instance.shape[1]))
      
      perturb_dl = test_dataloader(learner, perturbed_instances)
      perturb_probas, _, perturb_preds = learner.get_preds(dl=perturb_dl, with_decoded=True, save_preds=None)
      instances_probs, _ = torch.max(perturb_probas, dim=1)
      
      unique_predictions, counts = np.unique(perturb_preds, return_counts=True)

      # Print unique predictions and their counts
      for pred, count in zip(unique_predictions, counts):
          print(f"Class {pred}: {count} instances")
      
      print(perturb_probas[0])
      df_inc_dec = self.extract_inc_dec_events(perturbed_instances)
      df_max_min = self.extract_local_max_min_events(perturbed_instances)
      merged_df = self.merge_event_df(df_inc_dec=df_inc_dec, df_max_min=df_max_min)
      
      final_data, master_dict, cluster_centroids, kmeans_dict, scaler_dict = self.prepare_data4DT(merged_df, k=n_clusters)
      if kernel_width is None:
        kernel_width = 2 * np.sqrt(len(final_data.columns))
      weights = kernel(np.array(distances), kernel_width)
      important_features_lr, prediction_score, local_pred = self.main_lr(full_data=final_data, lstm_preds=instances_probs, weights=weights,
                                                          class_names=class_names, X=origi_instance, cluster_centroids=cluster_centroids, 
                                                          master_dict=master_dict, model_regressor=model_regressor)
      
      # print(important_features_lr)
      top_n = min(top_n, len(final_data.columns))
      # Check if top_n is greater than the number of features
      if top_n > len(final_data.columns):
          print("Warning: top_n exceeds the number of available features. Adjusting top_n to", len(final_data.columns))
      # Filter out features with zero importance
      non_zero_features = [(index, importance) for index, importance in enumerate(important_features_lr) if importance > 0]

      # Sort the features by importance in descending order
      sorted_features = sorted(non_zero_features, key=lambda x: x[1], reverse=True)
      # sorted_features = sorted(enumerate(important_features_lr), key=lambda x: x[1], reverse=True)

      # Select the top N features
      selected_features_indices = [index for index, _ in sorted_features[:top_n]]
      selected_features = {final_data.columns[index]: importance for index, importance in enumerate(important_features_lr) if index in selected_features_indices}
      
      origi_instance_pred = perturb_preds[0].item()
      
      print(f'Predection probability : {perturb_probas[0]}')
      probabilities = perturb_probas[0].squeeze().tolist()
    
      # Print colored filled boxes for each class
      for i, prob in enumerate(probabilities):
          self.helper_instance.print_colored_filled_boxes(class_names[i], i , prob)
        
      origi_instance_events = merged_df.iloc[0]
      important_motifs = self.helper_instance.events_in_topK_clusters(selected_features, origi_instance_events, kmeans_dict=kmeans_dict, scaler_dict=scaler_dict)
      
      self.helper_instance.plot_events_as_line_on_timeseries_1(origi_instance, origi_instance_pred, selected_features, cluster_centroids, class_names)
      self.helper_instance.plot_events_on_time_series(origi_instance, origi_instance_pred, important_motifs, class_names)
      return important_motifs, prediction_score, local_pred
    
    
    
    
               

          
          
    
    
    
    
  



