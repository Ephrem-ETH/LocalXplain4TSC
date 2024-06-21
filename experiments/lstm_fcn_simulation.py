import random
import time
import os
import argparse
import logging
import random
import time
import os
import sys
from tsai.all import *
from fastai.vision.all import *
from sklearn.model_selection import train_test_split
from fastai.callback.tracker import EarlyStoppingCallback
from fastai.vision.all import *
from torch.utils.data.dataset import ConcatDataset
from tsai.utils import set_seed
from sklearn.preprocessing import MinMaxScaler
from  globXplain import GlobXplain4TSC
from robustness import *
from joblib import Parallel, delayed


set_seed(1024)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(f"device: {device}")
my_setup()

def setup_logging(logfile):
    logging.basicConfig(filename=logfile, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    parser = argparse.ArgumentParser(description="Rule based global explanation for time series classifier")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--model", required=False, default=LSTM_FCN, help="Model name")
    parser.add_argument("--num_runs", type=int, default=100, help="Number of Monte Carlo runs")
    parser.add_argument("--class_labels", nargs='+', required=True, help="List of class labels")
    return parser.parse_args()

def data_preparation(X, y, test_size=0.4, random_state=12):
    # X_train, y_train = load_UCR_UEA_dataset(name=dataset, split='train', return_type='numpyflat')
    # X_test, y_test = load_UCR_UEA_dataset(name=dataset, split='test', return_type='numpyflat')
    # # Reshape the train and test data
    # X_train = X_train.reshape(-1, 1, X_train.shape[1])
    # X_test = X_test.reshape(-1, 1, X_test.shape[1])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    #Reserve 10% for validation, validation set is required in fastai
    X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.4, stratify=y_test, random_state=random_state)


    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=12)
    # X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=12)
    print(f'Shape of X_train :{X_train.shape}')
    print(f'Shape of y_train :{y_train.shape}')
    # X_valid, X_test, y_valid, y_test = train_test_split(X_test,y_test, test_size=0.65, random_state=1024)
    print(f'Shape of X_valid :{X_valid.shape}')
    print(f'Shape of X_test :{X_test.shape}')

    tfms  = [None, [Categorize()]]
    train_ds = TSDatasets(X_train, y_train, tfms=tfms)
    # valid_ds = TSDatasets(X_valid, y_valid, tfms=tfms)
    valid_ds = TSDatasets(X_valid, y_valid, tfms=tfms)


    combined_ds = ConcatDataset([train_ds, valid_ds])
    # print(combined_ds[0][0].shape)
    tfms = [None, [Categorize()]]
    # dls = get_ts_dls(combined_ds, tfms=tfms, bs=64)
    dls = TSDataLoaders.from_dsets(train_ds, valid_ds, bs=[64, 128], batch_tfms=[TSStandardize()], num_workers=0,  device=device, shuffle=True )
    print(f' Number of classes : {dls.c}')
    return dls, X_test, y_test

def trainer (base_dir, model, dls, epochs=150, learning_rate=1e-3, patience=15):
    kwargs = {}
    metrics=[accuracy]
    path = f'{base_dir}/export'
    os.makedirs(path, exist_ok=True)
    # Create model
    model = create_model(model, dls=dls, **kwargs)
    # Define early stopping criteria
    early_stopping = EarlyStoppingCallback(monitor='valid_loss', min_delta=0.001, patience=patience)

    # Define a call back to save the best model
    save_callback = SaveModelCallback(monitor='accuracy')
    # Train and evaluate model with early stopping
    # set_seed(42)
    cbs = [early_stopping] #ShowGraph(), save_callback
    learn = Learner(dls=dls, model=model, opt_func=Adam, metrics=metrics,cbs=cbs) #
    learn.fit_one_cycle(epochs, learning_rate)
    # learn.save_all(path=path, dls_fname='dls', model_fname='model', learner_fname='learner')
    return learn

def validator(learn, X_test, y_test):
    dls = learn.dls
    valid_dl = dls.valid
    train_dl = dls.train
    # Labelled test data
    test_ds = valid_dl.dataset.add_test(X_test, y_test)# In this case I'll use X and y, but this would be your test data
    test_dl = valid_dl.new(test_ds)
    test_probas, test_targets, test_preds = learn.get_preds(dl=test_dl, with_decoded=True, save_preds=None, save_targs=None)
    # print(f'Test Accuracy: {skm.accuracy_score(test_targets, test_preds):10.6f}')
    valid_probas, valid_targets, valid_preds = learn.get_preds(dl=valid_dl, with_decoded=True)
    train_probas, train_targets, train_preds = learn.get_preds(dl=train_dl, with_decoded=True)
    
    valid_accuracy = (valid_targets == valid_preds).float().mean()
    test_accuracy = (test_targets == test_preds).float().mean()
    train_accuracy = (train_targets == train_preds).float().mean()

    print(f"Validation Accuracy :{valid_accuracy:.2f}")
    print(f"Test Accuracy :{test_accuracy:.2f}")
    print(f"Train Accuracy :{train_accuracy:.2f}")
#     valid_dl.show_batch(sharey=True)
#     test_dl.show_batch(sharey=True)
#     learn.show_results(max_n=6)

    return train_accuracy, valid_accuracy, test_accuracy, test_preds, valid_preds

def run_single_iteration(run, X, y, model_name, class_names, base_dir):

    try:
        print(f"Starting iteration {run}")
        
        # Code for a single iteration
        dls, X_test, y_test = data_preparation(X, y, random_state=run)

        if 'cuda' in str(device):
            learn_new = trainer(base_dir, model_name, dls)
        else:
            print('GPU is not available!')

        train_acc, valid_acc, test_acc, test_preds, valid_preds = validator(learn_new, X_test, y_test)
        
        # Randomly pick one index along the first axis (axis=0)
        randomly_picked_index = np.random.choice(X_test.shape[0], size=1, replace=False)

        # Use the index to get the corresponding instance
        randomly_picked_instance = X_test[randomly_picked_index]

        # Squeeze the singleton dimension
        randomly_picked_instance = np.squeeze(randomly_picked_instance, axis=0)

        # The shape of randomly_picked_instance will be (1, 96)
        print(randomly_picked_instance.shape)

        # Print the index of the randomly picked instance
        print("Index of the randomly picked instance:", randomly_picked_index)
        # dt_acc, fidelity, depth, n_nodes = globxplain_instance.ts_global_rule_explanation(X_test, test_preds, class_names)
        important_features_lr, prediction_score, local_pred = globxplain_instance.ts_local_explanation(randomly_picked_instance, learn_new, num_perturbations=1000, class_names=class_names)
        
        # Robustness 
        # robustness_evaluater = Robustness_Evaluation(learner=learn_new, base_dir=base_dir)
        # _, _, robustness_score = robustness_evaluater.evaluate_XAI_robustness(X_test, y_test, class_names, perturbation_strength=0.1)

        print(f"Iteration {run} completed successfully")

        return train_acc, valid_acc, test_acc, prediction_score, local_pred
    except Exception as e:
        print(f"Exception occurred in iteration {run}: {e}")
        raise

def monte_carlo_cross_val_parallel(num_runs, model_name, class_names, base_dir):
    start_time = time.time()

    results = Parallel(n_jobs=1, backend='loky')(
        delayed(run_single_iteration)(run, X, y, model_name, class_names, base_dir) for run in range(num_runs)
    )

    train_accs, valid_accs, test_accs, prediction_scores,_ = zip(*results)
    # Save the results as a text file with a header
    header = "train_accs, valid_accs, test_accs, dt_accs, fidelities, depths, nodes, robust_score"
    np.savetxt(f'{base_dir}/results.txt', np.array([train_accs, valid_accs, test_accs, prediction_scores]).T, header=header)

    # Calculate means and std deviations
    train_mean, train_std = round(np.mean(train_accs), 2), round(np.std(train_accs), 2)
    valid_mean, valid_std = round(np.mean(valid_accs), 2), round(np.std(valid_accs), 2)
    test_mean, test_std = round(np.mean(test_accs), 2), round(np.std(test_accs), 2)
    fidelity_mean, fidelity_std = round(np.mean(prediction_scores), 2),  round(np.std(prediction_scores), 2)
    

    result_dict = {
        'v_acc_mean': valid_mean,
        'v_acc_std': valid_std,
        't_acc_mean': test_mean,
        't_acc_std': test_std,
        'fidelity_mean': fidelity_mean,
        'fidelity_std': fidelity_std,
        
    }

    end_time = time.time()
    total_time = round(end_time - start_time, 2)
    print(f"Total time taken: {total_time} seconds")

    return result_dict

if __name__ == "__main__":
    args = parse_arguments()
    dsid = args.dataset
    dataset_name = dsid.lower()
    model = args.model.lower()
    num_runs = args.num_runs
    class_labels = args.class_labels
    
    print(f'{dsid}, {model}, {num_runs}')

    cur_time = time.strftime('%Y-%m-%d_%H-%M-%S')
    base_dir = f'results/simulation/{dataset_name}/{model}--{cur_time}' if 'cuda' in str(device) else f'results/{dataset_name}--{cur_time}'
    os.makedirs(base_dir, exist_ok=True)

    logfile = f'{base_dir}/logfile.log'
    setup_logging(logfile)

    # Redirect standard output to a file
    sys.stdout = open(f'{base_dir}/output.log', 'w')

    logging.info("Starting the script.")

    try:
        X, y, splits = get_UCR_data(dsid, on_disk=True, return_split=False)
        deviation = 0.02
        globxplain_instance = GlobXplain4TSC(base_dir=base_dir)
        dls, X_test, y_test = data_preparation(X, y)
        # class_labels = ['No Symptom Exist', 'Symptom Exist']

        result_dict = monte_carlo_cross_val_parallel(num_runs, LSTM_FCN, class_labels, base_dir)

        from tabulate import tabulate

        lstm_fcn_result = [
            ["Test Accuracy", f"{result_dict['t_acc_mean']:.2f} \u00B1 {result_dict['t_acc_std']:.2f}"],
            ["Valid Accuracy", f"{result_dict['v_acc_mean']:.2f} \u00B1 {result_dict['v_acc_std']:.2f}"],
            ["Fidelity", f"{result_dict['fidelity_mean']:.2f} \u00B1 {result_dict['fidelity_std']:.2f}"],
        ]

        print(f"Info:\nDataset: {args.dataset}\nModel: {args.model}\nNum runs: {args.num_runs}")
        table_headers = ["Metric", f" Mean \u00B1 Std"]
        print(tabulate(lstm_fcn_result, headers=table_headers, tablefmt="grid"))

        logging.info("Script completed successfully.")
    except Exception as e:
        logging.error(f"An exception occurred: {e}", exc_info=True)

    # Restore standard output
    sys.stdout = sys.__stdout__

