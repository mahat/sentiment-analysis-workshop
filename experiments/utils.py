import re
import pandas as pd
import seaborn as sns
import numpy as np
from datasets import load_from_disk


import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return re.sub(TAG_RE, '', text.lower())
    
def evaluator(y_pred, y_true):
    # calculates accuracy, recall, precision, f1 score and confusion matrix given preds and true labels
    metrics = {}
    # Calculate metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['recall'] = recall_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred)
    metrics['f1_score'] = f1_score(y_true, y_pred)
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred,labels=[0,1])
    return metrics

def plot_cm(cm,title=""):
    df_cm = pd.DataFrame(
        cm, index=[0,1], columns=[0,1], 
    )
    # fig = plt.figure(figsize=figsize)
    fig = plt.figure()
    ax = sns.heatmap(df_cm, annot=True, fmt="d")
    ax.yaxis.set_ticklabels(ax.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=12)
    ax.xaxis.set_ticklabels(ax.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=12)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    return fig

def temp_check_helper(model,trange=np.arange(0.2,1.4,0.2)):
    metrics = []
    ta_func = lambda p,l: (np.array(p) == np.array(l)).sum() / len(l)
    for t in trange:
        print(t)
        model.temperature = t
        metric = model().eval()
        metric['temp'] = t
        metric['acc'] = ta_func(model.preds,model.labels)
        metrics.append(metric)
    return metrics

def minP_check_helper(model,temp=0.2,prange=np.arange(0.05,0.6,0.05)):
    ta_func = lambda p,l: (np.array(p) == np.array(l)).sum() / len(l)
    metrics = []
    for p in prange:
        print(p)
        # update temp
        model.temperature = temp
        model.min_p = p
        # run experiment
        metric = model().eval()
        metric['min_p'] = p
        metric['acc'] = ta_func(model.preds,model.labels)
        metrics.append(metric)
    return metrics

def plot_metrics(metric_list,x_key,y_key,title=""):
    fig = plt.figure()
    y = [e[y_key] for e in metric_list]
    x = [e[x_key] for e in metric_list]
    ax = sns.lineplot(x=x,y=y)
    ax.set_title(title)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    return fig

def load_ds():
    valid_ds_mini = load_from_disk('../data/valid_mini.hf')

    valid_ds_small = load_from_disk('../data/valid_small.hf')
    incontext_ds_small = load_from_disk('../data/incontext_small.hf')

    valid_ds_big = load_from_disk('../data/valid_big.hf')
    incontext_ds_big = load_from_disk('../data/incontext_big.hf')
    return valid_ds_mini, valid_ds_small, incontext_ds_small, valid_ds_big, incontext_ds_big

def load_test():
    return load_from_disk('../data/test.hf')