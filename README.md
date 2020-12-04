# w266_final
This is for Berkeley's MIDS program - course w266(NLP and deep learning)  final project on NER and Relation extraction

#### Authors: Valerie Meausoone and Prabhaker Narsina

#### Objective: Find relationship between given entities. Our focus on this project is been on i2b2/n2c2 dataset, 
#### however we have tested our model with TACRED dataset also.
#### Pre-requisites for the project
        1. Python 3.6 (or higher)
        2. Pytorch 1.7 (or higher)
        3. Other regular libarries (numpy, matplotlib, pathlib, tqdm, and more)

#### Steps to reproduce
        1. clone this git repository
        2. Set confinguraiton in config/config.json as shown below
            {
        "hyperparams":
            {"NUM_BERT_LAYERS_FREEZE": 8 , 
             "MAX_SEQ_LENGTH": 128 ,
              "optimizer" : "AdamW",  
              "TRAIN_BATCH_SIZE" : 12, 
              "EVAL_BATCH_SIZE" : 8, 
              "LEARNING_RATE" : 1e-5, 
              "NUM_TRAIN_EPOCHS" : 3, 
              "NUM_WARMP_STEPS" : 100,
              "WARMUP_PROPORTION" : 0.1,
              "SCHEDULER_STEP_SIZE" : 1,
              "GRADIENT_ACCUMULATION_STEPS" : 1,
              "PIN_MEMORY" : "True",
              "NUM_WORKERS" : 4,
              "LOSS_FN_CLASS_WEIGHTS": [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0 ]
            },
        "programsettings":
            {"DATA_DIR": "data_divided/"  ,
              "BERT_MODEL" : "bert-base-cased",  
              "TASK_NAME" : "re", 
              "OUTPUT_DIR" : "outputs/re/", 
              "REPORTS_DIR" : "reports/re/", 
              "CACHE_DIR" : "cache/", 
              "WEIGHTS_NAME" :  "pytorch_model.bin",
              "MODEL_NAME": "BioBERT_CNN_fc" , 
              "VALID_MODEL_NAMES": "BioBERT_CNN_fc, BioBERT_fc, BERT_Sequence",
              "DEBUG_PRINT" : 1
              },	
        "modelconfig":
            {"BERT_FEATURES": 768  ,
              "MAX_SEQ_LENGTH": 128 ,
              "LAYER1_FEATURES" : 256,  
              "LBAEL_CLASSES" : 9, 
              "KERNEL_1" : 2 , 
              "KERNEL_2" : 3, 
              "KERNEL_3" :  4,
              "KERNEL_SIZES" : [4,6,8],
              "STRIDE": 1,
              "OUT_SIZE" : 64,
              "DROP_OUT" : 0.01,
              "ACT_FUNCTION": "softmax", 
              "VALID_ACT_FNS": "softmax, cust_softmax, lsoftmax",
              "CUST_SFTMX_CLASS_BETA": 0.005
              }	

        }
      3. create folder data_divided in main folder and download n2c2 dataset.
      4. use data_prep_bert.ipynb for preparing data for BERT
      5. Run below scripts in iPython or Jupyter notebook
         from experiment import run_model
         run_model

         
#### Directory Structure: 
        Main folder 
        1. Train.py (Used for Training the model)
        2. eval.py (Used for Evaluating the model and saving the results)
        3. experiment.py (Run through experiment for given configuration)
        4. experiment_batch.py (Run through multipl experiments together and save the 
                                results to given location in the file system)

        data_pre folder:

        Util Folder:
        1 DataLoader.py (Used for preparing the data for BERT)
        2.Tools.py (configuraiton and related helper methods)

        Model folder:
         1. MedClinical.py ( Machine Learning models defined in this file)
         2. model_config.py (All the configruaiton needed for Models, driven by Config file in Config folder)

        Config Folder:
          1.config.json (3 different sections 1.Program settings, 2. hyper parameters 3. Model configuraiton)
             Has informaiton about where data(train/dev/test) is located, different hyper parameters 
                like learning rate, batch size, tpe of model etc.
          2. id_2_label.json (Has mapping from classids to labels)
          3. label_2.id.json (Has mapping from class labels to ids)

         ouput:
           Is used for saving different models that are exprimented through. We save with the extension of .bin

         reports:
           This used for saving results from different runs, so that we can analyze results of different experiments together.
           1. multi_model_results - Stores each experiment results including train/dev labels, loss, mcc, f1-score and more
           2. Test_results - These are the results from running the selected models on Test data. This file includes labels, 
               predictions and mislabeled data

         data_divided:
           1. *.tsv files - train,dev and test.tsv files - Input data for our training
           2. *.pkl files - cached model ready dataset

         notebooks:
          1. Experiments.ipynb (Used for running single experiment and analyze)
          2. FinalTest.ipynb (Use for running saved models to test on Test data set)
          3. MissedPredictionAnalysis.ipynb (Used to analyze test results and especially where the model went wrong)
