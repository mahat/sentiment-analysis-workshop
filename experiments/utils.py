import os
import re
import pandas as pd
import seaborn as sns
import numpy as np
import seaborn as sns
from pathlib import Path
from datasets import load_from_disk


import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
)

TAG_RE = re.compile(r"<[^>]+>")


def remove_tags(text):
    return re.sub(TAG_RE, "", text.lower())


def evaluator(y_pred, y_true):
    # calculates accuracy, recall, precision, f1 score and confusion matrix given preds and true labels
    metrics = {}
    # Calculate metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["recall"] = recall_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred)
    metrics["f1_score"] = f1_score(y_true, y_pred)
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return metrics


def plot_cm(cm, title=""):
    df_cm = pd.DataFrame(
        cm,
        index=[0, 1],
        columns=[0, 1],
    )
    # fig = plt.figure(figsize=figsize)
    fig = plt.figure()
    ax = sns.heatmap(df_cm, annot=True, fmt="d")
    ax.yaxis.set_ticklabels(
        ax.yaxis.get_ticklabels(), rotation=0, ha="right", fontsize=12
    )
    ax.xaxis.set_ticklabels(
        ax.xaxis.get_ticklabels(), rotation=45, ha="right", fontsize=12
    )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title(title)
    return fig


def temp_check_helper(model, trange=np.arange(0.2, 1.4, 0.2)):
    metrics = []
    ta_func = lambda p, l: (np.array(p) == np.array(l)).sum() / len(l)
    for t in trange:
        print(t)
        model.temperature = t
        metric = model().eval()
        metric["temp"] = t
        metric["acc"] = ta_func(model.preds, model.labels)
        metrics.append(metric)
    return metrics


def minP_check_helper(model, temp=0.2, prange=np.arange(0.05, 0.6, 0.05)):
    ta_func = lambda p, l: (np.array(p) == np.array(l)).sum() / len(l)
    metrics = []
    for p in prange:
        print(p)
        # update temp
        model.temperature = temp
        model.min_p = p
        # run experiment
        metric = model().eval()
        metric["min_p"] = p
        metric["acc"] = ta_func(model.preds, model.labels)
        metrics.append(metric)
    return metrics


def plot_metrics(metric_list, x_key, y_key, title=""):
    fig = plt.figure()
    y = [e[y_key] for e in metric_list]
    x = [e[x_key] for e in metric_list]
    ax = sns.lineplot(x=x, y=y)
    ax.set_title(title)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    return fig


def load_ds():
    valid_ds_mini = load_from_disk("../data/valid_mini.hf")

    valid_ds_small = load_from_disk("../data/valid_small.hf")
    incontext_ds_small = load_from_disk("../data/incontext_small.hf")

    valid_ds_big = load_from_disk("../data/valid_big.hf")
    incontext_ds_big = load_from_disk("../data/incontext_big.hf")
    return (
        valid_ds_mini,
        valid_ds_small,
        incontext_ds_small,
        valid_ds_big,
        incontext_ds_big,
    )


def load_test():
    return load_from_disk("../data/test.hf")


def experiment_runner(models, ds_path, out_path):
    """
    Runs each model with given datasets object in the path, and generates experiment reports
    """
    out_path = Path(out_path)
    ds = load_from_disk(ds_path)
    # check dataset format is same as valid ds
    assert "review" in ds.features, "dataset must have review column"
    assert "label" in ds.features, "dataset must have label column"
    experiment_results = {}
    for model_name in models:
        mdl = models[model_name]["instance"]()
        for p in models[model_name].get("params", {}):
            setattr(mdl, p, models[model_name]["params"][p])
        mdl.valid_ds = ds
        experiment_results[model_name] = mdl().eval()
        # create directory
        os.makedirs(out_path / model_name, exist_ok=True)
        # dump results to df
        pd.DataFrame(mdl.get_run_log()).to_csv(out_path / model_name / 'log.csv', index=False)

    # create plots
    sns.set_theme()
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_data = [
        {
            "model_name": k,
            "metric": "f1_score",
            "val": experiment_results[k]["f1_score"],
        }
        for k in experiment_results
    ] + [
        {
            "model_name": k,
            "metric": "valid_answer_ratio",
            "val": experiment_results[k]["valid_answer_ratio"],
        }
        for k in experiment_results
    ]
    ax = sns.barplot(pd.DataFrame(plot_data), x="model_name", y="val", hue="metric", ax=ax)
    ax.set_title('Compare Models based on F1 and Valid Score')
    plt.xticks(rotation=30)
    fig.savefig(out_path / "model_compare.png")
