from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import Lasso
import random
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
import torch
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset
from joblib import Parallel, delayed
from matplotlib.ticker import PercentFormatter  
import gc


def return_features_nn(model_name, X, X_train, y_train, X_test, y_test, tokenizer=None, fine_tune=True):
    import os
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_paths = {
        "vulBERTa": "/home/paperspace/Neural_Networks_Models/vulBERTa",
        "vulDeePecker": "/home/paperspace/Neural_Networks_Models/vuldeepecker",
        "codebert": "/home/paperspace/Neural_Networks_Models/model_dir"
    }

    if model_name not in model_paths:
        raise ValueError(f"Model '{model_name}' not recognized. Choose from {list(model_paths.keys())}")

    model_path = model_paths[model_name]

    if fine_tune:
        model, tokenizer = fine_tune_in_memory_nn(model_path, X_train, y_train, X_test, y_test)
    else:
        print(f">>> Loading pretrained {model_name} without fine-tuning...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True, num_labels=2)

    model.to(device)
    model.eval()

    X["combined_code"] = X["func_before"].astype(str).str.strip()
    X = X[X["combined_code"] != ""].copy()

    codes = X["combined_code"].tolist()
    if len(codes) == 0:
        raise ValueError("No valid code snippets found after filtering.")

    output_chunk_dir = f"/home/paperspace/feature_chunks_{model_name}"
    os.makedirs(output_chunk_dir, exist_ok=True)
    chunk_size = 10000
    batch_size = 64
    all_embeddings = []

    for chunk_start in range(0, len(codes), chunk_size):
        chunk = codes[chunk_start:chunk_start + chunk_size]
        chunk_embeddings = []
        for i in range(0, len(chunk), batch_size):
            batch = chunk[i:i + batch_size]
            
            inputs = tokenizer(
                batch, padding="max_length", truncation=True, max_length=512, return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                if model_name == "vulDeePecker":
                    outputs = model(**inputs)
                    hidden_states = outputs[0]
                elif model_name == "codebert":
                    outputs = model.roberta(**inputs)
                    hidden_states = outputs.last_hidden_state
                else:  # vulBERTa
                    outputs = model(**inputs, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[-1]

            if hidden_states.ndim == 3:
                emb = hidden_states[:, 0, :].cpu().numpy()
            elif hidden_states.ndim == 2:
                emb = hidden_states.cpu().numpy()
            else:
                raise ValueError(f"Unexpected hidden state shape: {hidden_states.shape}")

            chunk_embeddings.append(emb)

            del inputs, outputs, hidden_states
            torch.cuda.empty_cache()
            gc.collect()

        chunk_embeddings = np.vstack(chunk_embeddings)
        all_embeddings.append(chunk_embeddings)
        np.save(os.path.join(output_chunk_dir, f"chunk_{chunk_start // chunk_size}.npy"), chunk_embeddings)

    features = np.vstack(all_embeddings)
    return features
    
    
def fine_tune_in_memory_nn(model_path, X_train, y_train, X_test, y_test):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True, num_labels=2)

    def tokenize_fn(example):
        return tokenizer(example["combined_code"], truncation=True, padding="max_length", max_length=512)

    train_ds = Dataset.from_pandas(pd.DataFrame({"combined_code": X_train["func_before"].str.strip(), "labels": y_train}))
    test_ds = Dataset.from_pandas(pd.DataFrame({"combined_code": X_test["func_before"].str.strip(), "labels": y_test}))

    train_ds = train_ds.map(tokenize_fn, batched=True)
    test_ds = test_ds.map(tokenize_fn, batched=True)

    training_args = TrainingArguments(
        output_dir="./tmp",
        logging_dir="./tmp/logs",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        save_total_limit=1,
        report_to="none",  
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
    )

    trainer.train()
    return model, tokenizer
    
def preprocess(method,X_train,X_test,y_train,model):
    if method=="PCA":
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)  
        X_test_scaled = scaler.transform(X_test)  
        
        pca = PCA()
        pca.fit(X_train_scaled)
        
        explained_variance = pca.explained_variance_ratio_

        cumulative_variance = np.cumsum(explained_variance)

        n_components = min(50, int(X_train.shape[1] * 2 / 3))
        pca = PCA(n_components=n_components)
        X_train = pca.fit_transform(X_train_scaled)  
        X_test = pca.transform(X_test_scaled)
       
       
    if method=="RESAMPLING":
        oversample = RandomOverSampler(sampling_strategy={0: (2*y_train.value_counts()[0]),1:y_train.value_counts()[1]}, random_state=42)
        X_train, y_train = oversample.fit_resample(X_train, y_train)
        
    if method=="L1":
        lasso = Lasso(alpha=0.0005, max_iter=1000)
        lasso.fit(X_train, y_train)
        selected_features = np.where(lasso.coef_ != 0)[0]
        if isinstance(X_train, np.ndarray):
            X_train = X_train[:, selected_features]
            X_test = X_test[:, selected_features]
        else:
            X_train = X_train.iloc[:, selected_features]
            X_test = X_test.iloc[:, selected_features]
      
    return X_train,X_test,y_train

def run_model_base(X_train,y_train,X_test,y_test,model):
    if model=="LR":
        model = LogisticRegression(max_iter=100,solver='lbfgs',penalty=None)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test,y_pred)
        confusion_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
        return confusion_df
    
    if model=="DT":
        model = DecisionTreeClassifier(criterion='gini', max_depth=6, min_samples_split=2,max_features=None)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test,y_pred)
        confusion_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
        return confusion_df
    
    if model=="XGB":
        model = xgb.XGBClassifier(eval_metric='logloss',n_estimators=100,learning_rate=0.3,reg_lambda=1.0,reg_alpha=0.0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test,y_pred)
        confusion_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
        return confusion_df
    
    if model=="CATB":
        catboost_model = CatBoostClassifier(iterations=500, learning_rate=0.1, depth=6, verbose=0,bootstrap_type='Bayesian',l2_leaf_reg=3.0)
        catboost_model.fit(X_train, y_train)
        y_pred = catboost_model.predict(X_test)
        cm = confusion_matrix(y_test,y_pred)
        confusion_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
        return confusion_df
    
    if model=="RF":
        model = RandomForestClassifier(n_estimators=100, max_features='sqrt',min_samples_split=2,n_jobs=None,criterion='gini',bootstrap=True)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test,y_pred)
        confusion_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
        return confusion_df


def run_model_nn(model, prepmethod, features,y):
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2)
    X_train, X_test, y_train = preprocess(prepmethod, X_train, X_test, y_train,model)

    class PCAClassifier(nn.Module):
        def __init__(self, input_dim, num_classes=2):
            super(PCAClassifier, self).__init__()
            self.fc = nn.Linear(input_dim, num_classes)

        def forward(self, x):
            return self.fc(x)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.long)
    pca_classifier = PCAClassifier(input_dim=X_train.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(pca_classifier.parameters(), lr=0.001)


    num_epochs = 1000
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = pca_classifier(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    with torch.no_grad():
        logits = pca_classifier(X_test_tensor)
        predictions = torch.argmax(logits, dim=-1).numpy()

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    confusion_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
    
    return confusion_df

def combine_results_base(modellist,methodlist,X,y,output_folder):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results = pd.DataFrame(columns=["Model","Method","Accuracy","False positives","False negtives"])
    i = 1
    for model in modellist:
        print(model)
        for method in methodlist:
            print(method)
            X_train_use,X_test_use,y_train_use = preprocess(method,X_train,X_test,y_train,model)
            confusion_df = run_model_base(X_train_use,y_train_use,X_test_use,y_test,model)
            results.at[i,"Model"] = model
            results.at[i,"Method"] = method
            results.at[i,"Accuracy"] = (confusion_df.values[0, 0] + confusion_df.values[1, 1]) / confusion_df.values.sum()
            results.at[i,"False positives"] =  confusion_df.values[0,1]/confusion_df.values.sum()
            results.at[i,"False negatives"] =  confusion_df.values[1,0]/confusion_df.values.sum()
            i = i+1
            
    return results

def combine_results_base_multiple_runs(modellist, methodlist, X, y, output_folder, n_runs=10):
    all_runs = []

    for run in range(n_runs):
        results = combine_results_base(modellist, methodlist, X, y, output_folder)
        results["Run"] = run + 1
        all_runs.append(results)

    combined_df = pd.concat(all_runs, ignore_index=True)

    averaged_results = (
        combined_df
        .groupby(["Model", "Method"])
        [["Accuracy", "False positives", "False negatives"]]
        .mean()
        .reset_index()
    )

    return averaged_results

def combine_results_nn(modellist, methodlist, data, output_folder):
    results = pd.DataFrame(columns=["Model", "Method", "Accuracy", "False positives","False negatives"])
    X = data[['func_before', 'func_after']]
    y = data['vul']
    i = 1

    for model in modellist:
        print(model)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        features = return_features_nn(model, X, X_train,y_train,X_test,y_test, fine_tune=True)
        for method in methodlist:
            print(method)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Ignore all warnings
                confusion_df = run_model_nn(model, method, features, y)

            results.at[i, "Model"] = model
            results.at[i, "Method"] = method
            results.at[i, "Accuracy"] = (confusion_df.values[0, 0] + confusion_df.values[1, 1]) / confusion_df.values.sum()
            total = confusion_df.values.sum()
            results.at[i, "False positives"] = confusion_df.values[0, 1] / total
            results.at[i, "False negatives"] = confusion_df.values[1, 0] / total
            i += 1

    return results

def combine_results_nn_multiple_runs(modellist, methodlist, X, y, output_folder, n_runs=10):
    all_runs = []

    for run in range(n_runs):
        data = X.copy()
        data["vul"] = y
        results = combine_results_nn(modellist, methodlist, data, output_folder)
        results["Run"] = run + 1
        all_runs.append(results)

    combined_df = pd.concat(all_runs, ignore_index=True)

    averaged_results = (
        combined_df
        .groupby(["Model", "Method"])
        [["Accuracy", "False positives", "False negatives"]]
        .mean()
        .reset_index()
    )

    return averaged_results

def visualize_all_results(modellist_base,modellist_nn,methodlist,X_base,X_nn,y,output_folder):
    results_base = combine_results_base_multiple_runs(modellist_base,methodlist,X_base,y,output_folder)
    results_nn = combine_results_nn_multiple_runs(modellist_nn,methodlist,X_nn,y,output_folder)
    allresults = pd.concat([results_base,results_nn],axis=0)
    
    #Visualize FP vs Accuracy
    
    sns.set(style="whitegrid")

    g = sns.FacetGrid(allresults, col="Model", col_wrap=3, height=4, sharex=False, sharey=False)
    g.map_dataframe(sns.scatterplot, x="False positives", y="Accuracy", hue="Method", style="Method", s=100)
    g.set_titles("{col_name}")
    for ax in g.axes.flatten():
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    g.add_legend()
    g.fig.suptitle("Accuracy vs. False Positive Rate (per Model)", fontsize=14)
    plt.subplots_adjust(top=0.88)

    plotname = str(output_folder) + "/accuracy_vs_fp_base_facet.png"
    g.savefig(plotname, dpi=300, bbox_inches="tight")
    plt.close()
    
    #Visualize FP vs FN:
        
    sns.set(style="whitegrid")

    g = sns.FacetGrid(allresults, col="Model", col_wrap=3, height=4, sharex=False, sharey=False)
    g.map_dataframe(sns.scatterplot, x="False positives", y="False negatives", hue="Method", style="Method", s=100)
    g.set_titles("{col_name}")
    g.set_axis_labels("False Positive Rate", "False Negative Rate")
    for ax in g.axes.flatten():
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    g.add_legend()
    g.fig.suptitle("False Negatives vs. False Positives (per Model)", fontsize=14)
    plt.subplots_adjust(top=0.88)

    plotname = str(output_folder) + "/fn_vs_fp_base_facet.png"
    g.savefig(plotname, dpi=300, bbox_inches="tight")
    plt.close()    
    
    return allresults

methodlist = ["No Preproc", "PCA", "RESAMPLING", "L1"]
modellist_nn = ["vulDeePecker","codebert","vulBERTa"]
modellist_base = ["LR", "DT", "XGB", "CATB", "RF"]

data = pd.read_csv("/home/paperspace/MSR_data_cleaned.csv")

X_base = data[['Confidentiality','Complexity','Attack Origin','Authentication Required',
                           'Availability','Integrity','Vulnerability Classification',
                           'add_lines','del_lines','lang','project']]
X_base = pd.get_dummies(X_base, columns=[
            'Confidentiality','Complexity','Attack Origin','Authentication Required',
            'Availability','Integrity','Vulnerability Classification','lang','project'
        ], drop_first=True)

X_nn = data[['func_before', 'func_after']]
y = data['vul']
X_nn['vul'] = y

output_folder = "/home/paperspace/results"
final = visualize_all_results(
            modellist_base=modellist_base,
            modellist_nn=modellist_nn,
            methodlist=methodlist,
            X_base=X_base,
            X_nn=X_nn,
            y=y,
            output_folder=output_folder
        )
