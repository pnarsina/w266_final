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
    train_inputs, train_label_ids, train_preds, train_loss, dev_inputs, dev_label_ids, dev_loss, dev_preds, \
    test_inputs, test_preds, test_labels, test_loss = run_model(config, device)
    
#   save missed cases
    save_missed_cases_to_file(config, "BIOBERT_fc_dev_missedcases_" , dev_preds, dev_label_ids, train_inputs)
    save_missed_cases_to_file(config, "BIOBERT_fc_test_missedcases_" , test_preds, test_labels, test_inputs)
    
#   save results stats
    train_mcc, train_f1_score, train_df_results, train_label_matches_df = calculate_stats(train_label_ids,train_preds )
    dev_mcc, dev_f1_score, dev_df_results, dev_label_matches_df = calculate_stats(dev_label_ids,dev_preds )
    test_mcc, test_f1_score, test_df_results, test_label_matches_df = calculate_stats(test_labels,test_preds )

    
    config_json     = configEncoder().encode(config)
    
    all_experiment_results.append([config_json, train_loss, dev_loss, train_mcc, train_f1_score,dev_mcc,dev_f1_score, 
                               dev_label_ids, dev_preds,train_label_ids,train_preds,train_inputs,test_inputs, test_preds, test_labels,  \
                               test_mcc,test_f1_score   ])    
    
    


def run_all_experiments_save(device):


    config_folder = "config/"
    config = load_config(config_folder)

    all_experiment_results = []
    # Run with default configuration

    #   Try with different CNN configuraitons, default is 256 sequence lenth with customer softmax

    run_save_results (config, device, all_experiment_results)

#     #   Try with shorter sequence length
#     config.hyperparams.MAX_SEQ_LENGTH = 128
#     config.modelconfig.MAX_SEQ_LENGTH = 128
#     run_save_results (config, device, all_experiment_results)


#     #    Try with regular softmax, set back to 256 seq length
#     config.hyperparams.MAX_SEQ_LENGTH = 256
#     config.modelconfig.MAX_SEQ_LENGTH = 256
#     config.modelconfig.CUST_SFTMX_CLASS_BETA = 0.01
#     run_save_results (config, device, all_experiment_results)
    
    
#     config.modelconfig.ACT_FUNCTION = "softmax"
#     run_save_results (config, device, all_experiment_results)
    
# #    Try with different filter size
#     config.modelconfig.KERNEL_SIZES = [8,10,16]
#     run_save_results (config, device, all_experiment_results)
    
    all_model_results_pickle_file = config.programsettings.REPORTS_DIR + "multi_model_experiment_results_" + str(datetime.now()).replace(":", "_").replace(".", "_") + ".pkl"
    with open(all_model_results_pickle_file, "wb") as f:
        pickle.dump(all_experiment_results, f)      

        
        

# All previous runs

#After softmax fix
#     config.hyperparams.MAX_SEQ_LENGTH = 256
#     run_save_results (config, device, all_experiment_results)

#     config.hyperparams.MAX_SEQ_LENGTH = 384
#     run_save_results (config, device, all_experiment_results)

    
# #   Learning Rate and reset sequence length to 128, We already test with 1e-5 lr.
# #     config.hyperparams.MAX_SEQ_LENGTH = 128
# #     config.hyperparams.LEARNING_RATE = 0.75e-5
# #     run_save_results (config, device, all_experiment_results)

#     config.hyperparams.LEARNING_RATE = 1.25e-5
#     run_save_results (config, device, all_experiment_results)

#     config.hyperparams.LEARNING_RATE = 1.5e-5
#     run_save_results (config, device, all_experiment_results)
    

# # Modify Batchsize, we already tested with 12, alos reset learning rate to 1.0e-5
#     config.hyperparams.LEARNING_RATE = 1.5e-5
# #     config.hyperparams.TRAIN_BATCH_SIZE = 16
# #     run_save_results (config, device, all_experiment_results)

#     config.hyperparams.TRAIN_BATCH_SIZE = 24
#     run_save_results (config, device, all_experiment_results)
    
# # Modify Freezing layers and reset batch size to 12 
#     config.hyperparams.TRAIN_BATCH_SIZE = 12    
#     config.hyperparams.NUM_BERT_LAYERS_FREEZE = 8
#     run_save_results (config, device, all_experiment_results)
    
#     config.hyperparams.NUM_BERT_LAYERS_FREEZE = 10
#     run_save_results (config, device, all_experiment_results)
    
# Modfiy Weights, reset freeze layers to 12  , with all equal weights are covered 
#     config.hyperparams.NUM_BERT_LAYERS_FREEZE = 10
#     config.hyperparams.LOSS_FN_CLASS_WEIGHTS = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.0,1.0 ]
#     run_save_results (config, device, all_experiment_results)
    
#     config.hyperparams.LOSS_FN_CLASS_WEIGHTS = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,4.0,1.0 ]
#     run_save_results (config, device, all_experiment_results)

#     config.hyperparams.LOSS_FN_CLASS_WEIGHTS = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,8.0,1.0 ]
#     run_save_results (config, device, all_experiment_results)


#   Try with different LR 
#     config.hyperparams.LEARNING_RATE = 0.75e-5
#     run_save_results (config, device, all_experiment_results)

#     config.hyperparams.LEARNING_RATE = 1.25e-5
#     run_save_results (config, device, all_experiment_results)

    
# #   Try with Sequence length of 256
#     config.hyperparams.MAX_SEQ_LENGTH = 256
#     run_save_results (config, device, all_experiment_results)

# #   Try with Batchsize of 24
#     config.hyperparams.TRAIN_BATCH_SIZE = 24
#     run_save_results (config, device, all_experiment_results)
    
# #   reset to defaults
#     config = load_config(config_folder)
    

#Second iteration of running with weights
#     config.hyperparams.LOSS_FN_CLASS_WEIGHTS = [0.0530, 0.0469, 0.0415, 0.0430, 0.4402, 0.0607, 0.0636, 0.2443, 0.0068]
#     run_save_results (config, device, all_experiment_results)

# Third run with CNN

# Change the Model with different learning rate
#     config.hyperparams.LEARNING_RATE = 1.25e-5
#     run_save_results (config, device, all_experiment_results)

#Fourth Run with higher sequence length 
# Change the Model with different learning rate
#     config.hyperparams.LEARNING_RATE = 2e-5
#     run_save_results (config, device, all_experiment_results)    
    
#     config.hyperparams.LOSS_FN_CLASS_WEIGHTS = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0 ]
#     config.hyperparams.MAX_SEQ_LENGTH = 512
#     config.hyperparams.TRAIN_BATCH_SIZE = 12
#     run_save_results (config, device, all_experiment_results)
    
#     config.hyperparams.LOSS_FN_CLASS_WEIGHTS = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,3.0,1.0 ]
#     run_save_results (config, device, all_experiment_results)

    
#   Save all the results to the file for later analysis

# Fifth time with different CNN configuraitons

#     config.modelconfig.KERNEL_1 = 8
#     config.modelconfig.KERNEL_2 = 12
#     config.modelconfig.KERNEL_3 = 16

#     run_save_results (config, device, all_experiment_results)
    

#     config.modelconfig.KERNEL_1 = 36
#     config.modelconfig.KERNEL_2 = 48
#     config.modelconfig.KERNEL_3 = 60

#     run_save_results (config, device, all_experiment_results)
    
#     config.hyperparams.MAX_SEQ_LENGTH = 384
#     run_save_results (config, device, all_experiment_results)
