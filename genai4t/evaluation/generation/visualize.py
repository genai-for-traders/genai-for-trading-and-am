from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
from .util import create_real_synt_xy
from matplotlib import pyplot as plt
from genai4t.plot_style import plot_style


def visualize_time_series(
        real_data: np.ndarray,
        synt_data: np.ndarray):
    """
    shape: [seq, n_feat]
    """
    n_feat = real_data.shape[-1]
    f, axs=  plt.subplots(n_feat, 1, figsize=(10, n_feat * 5), sharey='row')

    for i, tax in enumerate(axs):
        tax.plot(real_data[:, i], label='real')
        tax.plot(synt_data[:, i], label='synthetic')
        tax.set_title(f'time serie: {i}')
        tax.legend()
        plot_style.apply_grid(tax)
        plot_style.apply_plot_style(tax)
    plt.show()


def visualize_2d_pca(
    real_data: np.ndarray,
    synt_data: np.ndarray,
    title: str = None):

    title = title or 'PCA'
    test_and_synt_data, is_real = create_real_synt_xy(real_data, synt_data)

    pca = PCA(n_components=2)
    pca.fit(real_data)

    encoded_test_and_generated = pca.transform(test_and_synt_data)
    pd_pca = pd.DataFrame(encoded_test_and_generated, columns=['P1', 'P2'])
    pd_pca['real'] = is_real
    f, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(
        x='P1',
        y='P2',
        data=pd_pca.query('real == 0'),
        ax=ax,
        alpha=0.5,
        label='synthetic')
    sns.scatterplot(
        x='P1',
        y='P2',
        data=pd_pca.query('real == 1'),
        ax=ax,
        label='real')
    ax.set_title(title)
    plt.legend()
    plot_style.apply_plot_style(ax)
    plot_style.apply_grid(ax)
    plt.show()

def visualize_2d_tsne(
    real_data: np.ndarray,
    synt_data: np.ndarray,
    title: str = None):
    title = title or 'TSNE'
    test_and_synt_data, is_real = create_real_synt_xy(real_data, synt_data)

    tsne = TSNE(n_components=2)
    tsne_encoded_test_and_generated = tsne.fit_transform(test_and_synt_data)

    pd_tsne = pd.DataFrame(tsne_encoded_test_and_generated, columns=['P1', 'P2'])
    pd_tsne['real'] = is_real

    f, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(
        x='P1',
        y='P2',
        data=pd_tsne.query('real == 0'),
        ax=ax,
        alpha=0.5,
        label='synthetic')
    sns.scatterplot(
        x='P1',
        y='P2',
        data=pd_tsne.query('real == 1'),
        ax=ax,
        label='real')
    ax.set_title(title)
    plt.legend()
    plot_style.apply_plot_style(ax)
    plot_style.apply_grid(ax)
    plt.show()
