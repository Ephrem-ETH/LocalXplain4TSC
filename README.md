# A Global Model-Agnostic Rule-Based XAI Method based on Parameterised Event Primitives for Time Series Classifiers


Introducing a new global model agnostic rule-based XAI method tailored for deep learning based time series classifiers. While there is a plethora of eXplainable AI (XAI) methods designed to elucidate the functioning of models trained on image and tabular data, adapting these methods for explaining deep learning-based time series classifiers may not be straightforward due to the temporal nature of time series data. The temporal nature of time series data adds complexity, necessitating a specialized approach.

Our project addresses this challenge with a novel methodology tailored for deep learning time series classifiers. The primary objective of this project is to offer a clear and interpretable understanding of deep learning-based time series classifiers while maintaining the temporal relationship in the sequence. The proposed solution aims to generate decision trees as explanations, providing insights that are comprehensible for human interpretation.

<!-- Introducing a novel global model-agnostic XAI method for deep learning-based time series classifiers. Adapting existing methods for this context poses challenges due to the temporal nature of time series data. Our project addresses this with a specialized approach, aiming to offer clear, interpretable insights. The primary objective is to generate decision trees as explanations, enhancing transparency for human interpretation. -->

## File description

 - experiments: This directory contains code files and results.

    - results: Includes experiment results for each dataset.

        - output.log: This file contains the objective evaluation of the method for each dataset and model type.

        * Example: Evaluation result for FCN model architecture trained on the ECG dataset can be found at `experiments\results\simulation\ecg200\fcn--2024-01-08_16-01-07\output.log.`


    



## Method Design 

<img src="design\globXplain_V2.1-1.png" alt="Method Design Diagram" width="70%" />


## Objective Evaluation Metrics

#### Table 1. Objective Evaluation Metrics for Rule-Based Explanation

| Metric     | Definition                                            | Formula                                     |
|------------|-------------------------------------------------------|---------------------------------------------|
| Accuracy   | Proportion of correctly predicted instances (c) out of the total instances (N) | A = c / N                                   |
| Fidelity   | Ratio of input instances where the surrogate model agrees (a) with the actual model, divided by the total number of instances (N) | F = a / N                                   |
| Complexity | The complexity or simplicity of the generated explanation is measured by the number of nodes and depth | C = #Depth, #Nodes                          |
| Robustness  | The persistence of methods to withstand small perturbations (δ) of the input that does not change the prediction of the model | R = Σ[g(x_n)=g(x_n+δ)] / N                  |


## Result

#### Table 4. Mean and standard deviation of the objective evaluation of the rule-based explanation for LSTM-FCN model



| Dataset   | Acc      | Fidelity | #Depth | #Node | Rob. |
|---------|----------|----------|--------|-------|------|
| ECG200  | 0.80±0.12 | 0.89±0.06 | 4±2 | 10±5 | 0.76±0.14 |
| GunPoint| 0.73±0.11 | 0.88±0.07 | 4±2 | 12±5 | 0.64±0.17 |
| FordA   | 0.79±0.04 | 0.84±0.05 | 8±4 | 41±38| 0.77±0.05 |
| FordB   | 0.81±0.04 | 0.86±0.05 | 7±4 | 37±33| 0.79±0.08 |


#### Table 4. Mean and standard deviation of the objective evaluation of the rule-based explanation for FCN model



| Dataset        | Acc | Fidelity | #Depth | #Node | Rob. |
|---------|-----|----------|--------|-------|------|
| ECG200  | 0.79±0.10 | 0.89±0.06 | 3±2 | 10±6 | 0.78±0.12 |
| GunPoint| 0.74±0.12 | 0.88±0.11 | 4±2 | 12±5 | 0.64±0.18 |
| FordA   | 0.78±0.03 | 0.84±0.04 | 8±3 | 42±34| 0.76±0.04 |
| FordB   | 0.81±0.04 | 0.87±0.05 | 8±4 | 42±34| 0.77±0.06 |


## Usage

To run the the simulation of the experiment, use the following command:



* For FCN model
```
python fcn_simulation --dataset [dataset-name] --num_runs [100 ]  --class_labels [list of the class names]
```


* For LSTM-FCN model
```
python lstm_fcn_simulation --dataset [dataset-name] --num_runs [100 ] --class_labels [list of the class names]
```



## Requirments 



