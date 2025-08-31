This file explains the use of each of the functions that can be found in the code inside this folder. The functions follow a hierarchical structure that ends with visualize_all_results().Therefore by running the final  28 lines, starting  with 'methodlist = ["No Preproc", "PCA", "RESAMPLING", "L1"]' one can run all results as the functions will trigger each other. In those lines, the path to the data needs to be replaced by the location where the data is saved (line 524), the output folder needs to be specified (line 538), and in lines 72,134 and 158, the local paths to the model files need to be replaced. Unfortunately, the model files couldn't be uploaded to this git project due to size limitations. In line 89, leave a path where intermediate output from vulBERTa will be stored.
The file dependencies.txt will contain all dependencies that have to  be installed in order for the code to run smoothly. The pre-trained models and dataset have been uploaded to Google Drive for easy access.
They can be downloaded from the following link: https://drive.google.com/drive/folders/1_2rS7MWai-J-8dcHvzjnYFdaTuw3vbvE?usp=share_link

The following functions can be found:

- finetune_in_memory_nn:
	Takes in the name of the NN architecture to be used, as well as the data to be used for fine-tuning

- return_features_nn:
	Takes in the name of the NN architecture to be used, as well as the data from which to extract these features.
	What this function does is read in the source code, apply the neural network architecture of choice and return the last layer of the neural network as described in the thesis. These features will then serve as input to the pre-processing methods when applied. 

- preprocess:
	Takes in the pre-processing method to be used, the data it should be applied to and the name of the model
	This function takes in data, applies the relevant pre-processing method to it and returns the manipulated data to then be used by the models for prediction

- run_model_base:
	Takes in which model should be run and the data that it should be applied to
	This function runs the relevant base model, training it on our data and returning a confusion matrix with the predicted vs actual outcomes that can then be used to calculate relevant performance measures

- run_model_nn:
	Takes in which NN architecture and which preprocessing method should be applied, as well as the extracted features and output labels
	This function calls on preprocess() to preprocess the generated features and adds a predictive layer on top, makes predictions and outputs a confusion matrix that can be used to calculate relevant performance measures

- combine_results_base:
	Takes in a list of models and preprocessing methods to consider, as well as the data and an output folder where to store the results
	This function loops over all base models and preprocessing methods, calculates accuracy, false positive rate and false negative rate for all scenarios and stores the results in a dataframe. It calls on preprocess() and run_model_base() to manipulate the data according to each preprocessing method and run the models

- combine_results_base_multiple_runs:
	Takes in all input to combine_results_base and the number of runs
	This function runs the combine_results_base function multiple times and averages over the results

- combine_results_nn:
	Takes in a list of models and preprocessing methods to consider, as well as the data and an output folder where to store the results
	This function  loops over all  nn models and preprocessing methods, calculateing accuracy, false positive rate and false negative rate for all scenarios and stores the results in a dataframe. It calls on run_model_nn() to do the preprocessing and run the models

-combine_results_nn_multiple_runs:
	Takes in all input to combine_results_nn and the number of runs
	This function runs the combine_results_nn function multiple times and averages over the results

- visualize_all_results:
	Takes in a base and an NN  model list, a list of preprocesing methods, and data to be fed to the base and NN models, as well as an output folder to store the results
	This function calls on combine_results_base_multiple_runs() and combine_results_nn_multiple_runs() to generate all results, stores them and creates graph that summarizes the most relevant results
