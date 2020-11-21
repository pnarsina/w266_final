import os, json
from types import SimpleNamespace
from experiment import run_model
from eval import calculate_stats
import pickle
from datetime import datetime 
import torch
from transformers import BertTokenizer
from sklearn.metrics import classification_report
from util.tools import load_config, configEncoder


def save_missed_cases_to_file(config, file_start_name, dev_preds, dev_label_ids, train_inputs):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
    missed_cases = []
    for i in range(0,50):
        if dev_label_ids[i] !=  dev_preds[i]:
             missed_cases.append([ dev_preds[i],  dev_label_ids[i] , " ". join (tokenizer.convert_ids_to_tokens(train_inputs[i])) ])

    #Save into a file
    missed_cases_file = config.programsettings.REPORTS_DIR +file_start_name + str(datetime.now()).replace(":", "_").replace(".", "_") + ".pkl"
    with open(missed_cases_file, "wb") as f:
        pickle.dump(missed_cases, f)  

def run_save_results(config,device,all_experiment_results):
#   Run the model and capture the details
    train_inputs, train_label_ids, train_preds, train_loss, dev_inputs, dev_label_ids, dev_loss, dev_preds = run_model(config, device)
    
#   save missed cases
    save_missed_cases_to_file(config, "BIOBERT_fc_missedcases_" , dev_preds, dev_label_ids, train_inputs)
    
#   save results stats
    train_mcc, train_f1_score, train_df_results, train_label_matches_df = calculate_stats(train_label_ids,train_preds )
    dev_mcc, dev_f1_score, dev_df_results, dev_label_matches_df = calculate_stats(dev_label_ids,dev_preds )

    config_json     = configEncoder().encode(config)
    
    all_experiment_results.append([config_json, train_loss, dev_loss, train_mcc, train_f1_score,dev_mcc,dev_f1_score, 
                               dev_label_ids, dev_preds,train_label_ids,train_preds  ])    

    


def run_all_experiments_save():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    config_folder = "config/"
    config = load_config(config_folder)

    all_experiment_results = []
    # Run with default configuration
    
    run_save_results (config, device, all_experiment_results)

    config.hyperparams.LEARNING_RATE = 0.75e-5
    run_save_results (config, device, all_experiment_results)

    config.hyperparams.LEARNING_RATE = 1.25e-5
    run_save_results (config, device, all_experiment_results)
    
#   reset to defaults
    config = load_config(config_folder)
    
    
#   Save all the results to the file for later analysis
    
    all_model_results_pickle_file = config.programsettings.REPORTS_DIR + "multi_model_experiment_results_" + str(datetime.now()).replace(":", "_").replace(".", "_") + ".pkl"
    with open(all_model_results_pickle_file, "wb") as f:
        pickle.dump(all_experiment_results, f)      