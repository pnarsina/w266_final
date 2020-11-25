# w266_final
This is for Berkeley's MIDS program - course w266(NLP and deep learning)  final project on NER and Relation extraction

#### Authors: Valerie Meausoone and Prabhaker Narsina

#### Objective: Find relationship between given entities. Our focus on this project is been on i2b2/n2c2 dataset, however we have tested our model with TACRED dataset also.

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
                   1. multi_model_results - Stores each experiment results including train/dev labels, loss, msc, f1-score and more
                   2. Test_results - These are the results from running the selected models on Test data. This file includes labels, predictions and mislabeled data
                  
                 data_divided:
                   1. *.tsv files - train,dev and test.tsv files - Input data for our training
                   2. *.pkl files - cached model ready dataset
                
                 notebooks:
                  1. Experiments.ipynb (Used for running single experiment and analyze)
                  2. FinalTest.ipynb (Use for running saved models to test on Test data set)
                  3. MissedPredictionAnalysis.ipynb (Used to analyze test results and especially where the model went wrong)
