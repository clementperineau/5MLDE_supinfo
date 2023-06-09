import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from prefect import task, flow

sns.set() # set seaborn style

DPI = 300 # set DPI value for the generated figures

@task(name='label_share', tags=['plots_func']) # create a task named label_share and tag it with 'plots_func'
def label_share(share: pd.Series, fp: str) -> None:
    # Compute the normalized share of each label
    share_norm = share / share.sum()
    # Create a bar plot with seaborn
    fig, ax = plt.subplots()
    bar = sns.barplot(x=share_norm.index, y=share_norm.values)
    # Add annotations to each bar of the plot with the share and the normalized share
    for idx, p in enumerate(bar.patches):
        bar.annotate('{:.2f}\n({})'.format(share_norm[idx], share[idx]),
                     (p.get_x() + p.get_width() / 2, p.get_height() / 2),
                     ha='center', va='center', color='white', fontsize='large')
                     
    # Set the x and y labels of the plot, as well as the title, and adjust the layout
    ax.set_xlabel('Label')
    ax.set_ylabel('Share')
    ax.set_title('Label Share')
    fig.tight_layout()
    # Save the plot to a file and close it
    fig.savefig(fp, dpi=DPI)
    plt.close(fig)

@task(name='corr_matrix', tags=['plots_func']) # create a task named corr_matrix and tag it with 'plots_func'
def corr_matrix(corr: pd.DataFrame, fp: str) -> None:
    # Create a correlation matrix plot with seaborn
    fig, ax = plt.subplots()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask, k=1)] = True
    sns.heatmap(corr, vmin=-1, vmax=1, mask=mask,
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                linewidths=0.5, cbar=True, square=True, ax=ax)
    ax.set_title('Correlation Matrix')
    fig.tight_layout()
    fig.savefig(fp, dpi=DPI)
    plt.close(fig)

@task(name='confusion_matrix', tags=['plots_func']) # create a task named confusion_matrix and tag it with 'plots_func'
def confusion_matrix(cm: np.array, fp: str, norm_axis: int =1) -> None:
    """
    [TN, FP]
    [FN, TP]
    """
    # Normalize the confusion matrix by either the rows or the columns
    cm_norm = cm / cm.sum(axis=norm_axis, keepdims=True)
    # Extract the values of the confusion matrix and the normalized confusion matrix
    TN, FP, FN, TP = cm.ravel()
    TN_norm, FP_norm, FN_norm, TP_norm = cm_norm.ravel()
    # Create an array of annotations for each cell of the confusion matrix
    annot = np.array([
        [f'TN: {TN}\n({TN_norm:.3f})', f'FP: {FP}\n({FP_norm:.3f})'],
        [f'FN: {FN}\n({FN_norm:.3f})', f'TP: {TP}\n({TP_norm:.3f})']
    ])

    # Create a heatmap of the confusion matrix with seaborn
    fig, ax = plt.subplots()
    sns.heatmap(cm_norm, cmap='Blues', vmin=0, vmax=1,
                annot=annot, fmt='s', annot_kws={'fontsize': 'large'},
                linewidths=0.2, cbar=True, square=True, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    fig.savefig(fp, dpi=DPI)
    plt.close(fig)

@task(name='metric', tags=['plots_func'])
def metric(metrics: np.array, fp: str) -> None:
    fig, ax = plt.subplots()
    for idx, data in enumerate(metrics):
        line = ax.plot(data['values'], label='fold{}'.format(idx), zorder=1)[0]
        ax.scatter(data['best_iteration'], data['values'][data['best_iteration'] - 1],
                   s=60, c=[line.get_color()], edgecolors='k', linewidths=1, zorder=2)
    ax.set_xlabel('Iterations')
    ax.set_ylabel(metrics[0]['name'])
    ax.set_title('Metric History (marker on each line represents the best iteration)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(fp, dpi=DPI)
    plt.close(fig)

@task(name='feature_importance', tags=['plots_func'])
def feature_importance(features: np.array, feature_importances: np.array, title: str, fp: str) -> None:
    fig, ax = plt.subplots()
    idxes = np.argsort(feature_importances)[::-1]
    y = np.arange(len(feature_importances))
    ax.barh(y, feature_importances[idxes][::-1], align='center', height=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(features[idxes][::-1])
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(fp, dpi=DPI)
    plt.close(fig)

@task(name='scores', tags=['plots_func'])
def scores(scores: dict, fp: str) -> None:
    array = np.array([v for v in scores.values()]).reshape((2, 2))
    annot = np.array(['{}: {:.3f}'.format(k, v) for k, v in scores.items()]).reshape((2, 2))
    fig, ax = plt.subplots()
    sns.heatmap(array, cmap='Blues', vmin=0, vmax=1,
                annot=annot, fmt='s', annot_kws={'fontsize': 'large'},
                linewidths=0.1, cbar=True, square=True, ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Average Classification Scores')
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    fig.savefig(fp, dpi=DPI)
    plt.close(fig)

@task(name='roc_curve', tags=['plots_func'])
def roc_curve(fpr: np.array, tpr: np.array, auc: float, fp: str) -> None:
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], 'k:')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title(f'ROC Curve (AUC: {auc:.3f})')
    fig.tight_layout()
    fig.savefig(fp, dpi=DPI)
    plt.close(fig)

@task(name='pr_curve', tags=['plots_func'])
def pr_curve(pre: np.array, rec: np.array, auc: float, fp: str) -> None:
    fig, ax = plt.subplots()
    ax.plot(pre, rec)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Presision')
    ax.set_title(f'Precision-Recall Curve (AUC: {auc:.3f})')
    fig.tight_layout()
    fig.savefig(fp, dpi=DPI)
    plt.close(fig)
