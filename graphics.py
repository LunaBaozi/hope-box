import os, argparse
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import Draw

from scipy.stats import gaussian_kde
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import PIL.ImageOps
import seaborn as sns
from collections import Counter

from scripts.aurk_int_preprocess import read_aurora_kinase_interactions
from scripts.gen_mols_preprocess import load_mols_from_sdf_folder
from scipy.stats import mannwhitneyu


# SMILES
def plot_len_smiles(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    min_len = int(df['len_smiles'].min())
    max_len = int(df['len_smiles'].max())
    bins = np.linspace(min_len - 0.5, max_len + 0.5, num=30)
    ax.hist(df['len_smiles'].dropna(), bins=bins, color='green', edgecolor='black', rwidth=0.8, alpha=0.6, label='Histogram')
    # KDE line for continuous distribution
    try:
        values = df['len_smiles'].dropna().values
        kde = gaussian_kde(values)
        x_vals = np.linspace(min_len, max_len, 300)
        kde_vals = kde(x_vals) * len(values) * (bins[1] - bins[0])
        ax.plot(x_vals, kde_vals, color='darkgreen', lw=2, label='KDE')
    except ImportError:
        pass  # KDE will not be plotted if scipy is not available
    mean_val = df['len_smiles'].mean()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_val:.2f}')
    ax.set_xlabel('SMILES length')
    ax.set_ylabel('Count of Molecules')
    ax.set_xticks(np.linspace(min_len, max_len, 10, dtype=int))
    ax.set_xlim([min_len - 0.5, max_len + 0.5])
    ax.set_title('Histogram of SMILES Length')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(image_dir, f'len_smiles_{epoch}_{num_gen}_{known_binding_site}_{aurora}.png'))
    return

def pie_chart_len_smiles(df):
    # Define bins for SMILES length
    bins = [0, 20, 40, 60, 80, float('inf')]
    labels = ['≤20 chars',
              '21-40 chars',
              '41-60 chars',
              '61-80 chars',
              '>80 chars']
    df['len_bin'] = pd.cut(df['len_smiles'], bins=bins, labels=labels, right=True, include_lowest=True)
    counts = df['len_bin'].value_counts().reindex(labels)

    plt.figure(figsize=(8, 6))
    plt.pie(counts,
            labels=counts.index,
            autopct='%1.1f%%',
            startangle=140,
            colors=plt.cm.Set2.colors)
    plt.title('SMILES Length Distribution')
    plt.axis('equal')
    plt.savefig(os.path.join(image_dir, f'len_smiles_pie_{epoch}_{num_gen}_{known_binding_site}_{aurora}.png'))
    plt.show()
    return

# SA_SCORE
def plot_sa_score(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    bin_centers = np.arange(1, 11)
    bins = np.arange(0.75, 10.25, 0.5)
    values = df['SA_score'].dropna().values
    ax.hist(values, bins=bins, color='orange', edgecolor='black', rwidth=0.8, alpha=0.6, label='Histogram')
    try:
        kde = gaussian_kde(values)
        x_vals = np.linspace(1, 10, 300)
        kde_vals = kde(x_vals) * len(values) * (bins[1] - bins[0])
        ax.plot(x_vals, kde_vals, color='darkorange', lw=2, label='KDE')
    except ImportError:
        pass  
    mean_val = np.mean(values)
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_val:.2f}')
    ax.set_xlabel('SA_Score (centered bins 1-10)')
    ax.set_ylabel('Count of Molecules')
    ax.set_xticks(bin_centers)
    ax.set_xlim([0.5, 10.5])
    ax.set_title('Histogram of SA_Score (centered bins 1-10)')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend()

    if 'smiles' in df.columns:
        top_idx = df['SA_score'].idxmin()
        top_smiles = df.loc[top_idx, 'smiles']
        top_score = df.loc[top_idx, 'SA_score']
        mol = Chem.MolFromSmiles(top_smiles)
        if mol is not None:
            img = Draw.MolToImage(mol, size=(1800, 1800))  
            img = PIL.ImageOps.expand(img, border=8, fill='black')
            top_filename = df.loc[top_idx, 'filename'] if 'filename' in df.columns else ''
            text = f'Filename:\n{top_filename}\nSA_score: {top_score:.2f}'

            legend = ax.get_legend()
            if legend:
                fig.canvas.draw()  # Needed to get correct legend position
                legend_box = legend.get_window_extent(fig.canvas.get_renderer())
                legend_box = legend_box.transformed(fig.transFigure.inverted())
                inset_width = 0.3  # fraction of figure width
                inset_height = 0.6  # fraction of figure height
                inset_x = legend_box.x1 - inset_width
                inset_y = legend_box.y0 - inset_height - 0.02  # 0.02 gap below legend
                inset_ax = fig.add_axes([inset_x, inset_y, inset_width, inset_height])
                inset_ax.axis('off')
                inset_ax.imshow(img)
                inset_ax.text(0.5, 0.05, text,
                              ha='center', va='bottom',
                              fontsize=6,
                              wrap=True,
                              color='white',
                              bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.3'),
                              transform=inset_ax.transAxes)
    plt.show()
    plt.savefig(os.path.join(image_dir, f'sa_score_{epoch}_{num_gen}_{known_binding_site}_{aurora}.png'))
    return

# SCSCORE
def plot_sc_score(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    bin_centers = np.arange(1, 6)
    bins = np.arange(0.75, 5.25, 0.25)
    values = df['SCScore'].dropna().values
    ax.hist(values, bins=bins, color='blue', edgecolor='black', rwidth=0.8, alpha=0.6, label='Histogram')
    try:
        kde = gaussian_kde(values)
        x_vals = np.linspace(1, 5, 300)
        kde_vals = kde(x_vals) * len(values) * (bins[1] - bins[0])
        ax.plot(x_vals, kde_vals, color='darkblue', lw=2, label='KDE')
    except ImportError:
        pass  
    mean_val = np.mean(values)
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_val:.2f}')
    ax.set_xlabel('SCScore')
    ax.set_ylabel('Count of Molecules')
    ax.set_xticks(bin_centers)
    ax.set_xlim([0.5, 5.5])
    ax.set_title('Histogram of SCScore')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend()

    if 'smiles' in df.columns:
        top_idx = df['SCScore'].idxmin()
        top_smiles = df.loc[top_idx, 'smiles']
        top_score = df.loc[top_idx, 'SCScore']
        mol = Chem.MolFromSmiles(top_smiles)
        if mol is not None:
            img = Draw.MolToImage(mol, size=(1800, 1800))
            img = PIL.ImageOps.expand(img, border=8, fill='black')
            top_filename = df.loc[top_idx, 'filename'] if 'filename' in df.columns else ''
            text = f'Filename:\n{top_filename}\nSCScore: {top_score:.2f}'

            legend = ax.get_legend()
            if legend:
                fig.canvas.draw()
                legend_box = legend.get_window_extent(fig.canvas.get_renderer())
                legend_box = legend_box.transformed(fig.transFigure.inverted())
                inset_width = 0.3
                inset_height = 0.6
                # Align left sides: set inset_x to legend_box.x0
                inset_x = legend_box.x0
                inset_y = legend_box.y0 - inset_height - 0.02
                inset_ax = fig.add_axes([inset_x, inset_y, inset_width, inset_height])
                inset_ax.axis('off')
                inset_ax.imshow(img)
                inset_ax.text(0.5, 0.05, text,
                              ha='center', va='bottom',
                              fontsize=6,
                              wrap=True,
                              color='white',
                              bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.3'),
                              transform=inset_ax.transAxes)
    plt.show()
    plt.savefig(os.path.join(image_dir, f'sc_score_{epoch}_{num_gen}_{known_binding_site}_{aurora}.png'))
    return


# NP_SCORE
def plot_np_score(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    bin_centers = np.arange(-5, 6)
    bins = np.arange(-5.75, 5.25, 0.5)
    values = df['NP_score'].dropna().values
    ax.hist(values, bins=bins, color='pink', edgecolor='black', rwidth=0.8, alpha=0.6, label='Histogram')
    try:
        kde = gaussian_kde(values)
        x_vals = np.linspace(-5.5, 5.5, 300)
        kde_vals = kde(x_vals) * len(values) * (bins[1] - bins[0])
        ax.plot(x_vals, kde_vals, color='deeppink', lw=2, label='KDE')
    except ImportError:
        pass  
    mean_val = np.mean(values)
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_val:.2f}')
    ax.set_xlabel('NP_score')
    ax.set_ylabel('Count of Molecules')
    ax.set_xticks(bin_centers)
    ax.set_xlim([-5.5, 5.5])
    ax.set_title('Histogram of NP_score')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend()

    if 'smiles' in df.columns:
        # Best NP_score is the lowest
        top_idx = df['NP_score'].idxmin()
        top_smiles = df.loc[top_idx, 'smiles']
        top_score = df.loc[top_idx, 'NP_score']
        mol = Chem.MolFromSmiles(top_smiles)
        if mol is not None:
            img = Draw.MolToImage(mol, size=(1800, 1800))
            img = PIL.ImageOps.expand(img, border=8, fill='black')
            top_filename = df.loc[top_idx, 'filename'] if 'filename' in df.columns else ''
            text = f'Filename:\n{top_filename}\nNP_score: {top_score:.2f}'

            legend = ax.get_legend()
            if legend:
                fig.canvas.draw()
                legend_box = legend.get_window_extent(fig.canvas.get_renderer())
                legend_box = legend_box.transformed(fig.transFigure.inverted())
                inset_width = 0.3
                inset_height = 0.6
                # Align right sides: set inset_x so that inset_ax's right matches legend_box.x1
                inset_x = legend_box.x1 - inset_width
                inset_y = legend_box.y0 - inset_height - 0.02
                inset_ax = fig.add_axes([inset_x, inset_y, inset_width, inset_height])
                inset_ax.axis('off')
                inset_ax.imshow(img)
                inset_ax.text(0.5, 0.05, text,
                              ha='center', va='bottom',
                              fontsize=6,
                              wrap=True,
                              color='white',
                              bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.3'),
                              transform=inset_ax.transAxes)
    plt.show()
    plt.savefig(os.path.join(image_dir, f'np_score_{epoch}_{num_gen}_{known_binding_site}_{aurora}.png'))
    return


def plot_tSNE(fps):
    n_samples = len(fps)
    # Set perplexity dynamically: must be < n_samples, typically between 5 and 50
    if n_samples <= 5:
        perplexity = max(2, n_samples - 1)
    elif n_samples < 50:
        perplexity = max(5, n_samples // 3)
    else:
        perplexity = 30
    # Ensure perplexity is strictly less than n_samples
    if perplexity >= n_samples:
        perplexity = max(1, n_samples - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    tsne_results = tsne.fit_transform(np.stack(fps))

    plt.figure(figsize=(8,6))
    plt.scatter(tsne_results[:,0], tsne_results[:,1], c=synth_df['SA_score'].values, cmap='viridis', alpha=0.7, s=30)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('t-SNE of Chemical Space (colored by SA_score)')
    plt.colorbar(label='SA_score')
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, f'tSNE_{epoch}_{num_gen}_{known_binding_site}_{aurora}.png'))
    print(f'tSNE plot saved to {image_dir}')
    return


def plot_synthesizability_boxplot(df):
    # Normalize scores to 0-1
    sa_norm = (df['SA_score'] - 1) / 9      # SA_score: 1-10
    sc_norm = (df['SCScore'] - 1) / 4       # SCScore: 1-5
    np_norm = (df['NP_score'] + 5) / 10     # NP_score: -5 to 5

    data = [sa_norm.dropna().values, sc_norm.dropna().values, np_norm.dropna().values]
    scores = ['SA_score', 'SCScore', 'NP_score']

    fig, ax = plt.subplots(figsize=(8, 5))
    box = ax.boxplot(data, patch_artist=True, labels=scores, showmeans=True,
                        boxprops=dict(facecolor='lightgray', color='black'),
                        medianprops=dict(color='red'),
                        meanprops=dict(marker='o', markerfacecolor='blue', markeredgecolor='black'))

    ax.set_ylabel('Normalized Score Value (0-1)')
    ax.set_title('Boxplot of Normalized Synthesizability Scores')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, f'boxplot_{epoch}_{num_gen}_{known_binding_site}_{aurora}.png'))
    plt.show()
    return

def plot_synthesizability_violinplot(df):
    # Normalize scores to 0-1
    sa_norm = (df['SA_score'] - 1) / 9      # SA_score: 1-10
    sc_norm = (df['SCScore'] - 1) / 4       # SCScore: 1-5
    np_norm = (df['NP_score'] + 5) / 10     # NP_score: -5 to 5

    data = [sa_norm.dropna().values, sc_norm.dropna().values, np_norm.dropna().values]
    scores = ['SA_score', 'SCScore', 'NP_score']

    # Statistical significance: pairwise Mann-Whitney U test
    sig_labels = scores.copy()
    alpha = 0.05 / 3  # Bonferroni correction for 3 comparisons
    pairs = [(0,1), (0,2), (1,2)]
    sig_matrix = [False, False, False]
    for idx, (i, j) in enumerate(pairs):
        stat, p = mannwhitneyu(data[i], data[j], alternative='two-sided')
        if p < alpha:
            sig_matrix[i] = True
            sig_matrix[j] = True
    # Add '***' if significant
    for i in range(3):
        if sig_matrix[i]:
            sig_labels[i] += ' ***'

    fig, ax = plt.subplots(figsize=(8, 5))
    parts = ax.violinplot(data, showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(['orange', 'blue', 'pink'][i])
        pc.set_alpha(0.5)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(sig_labels)
    ax.set_ylabel('Normalized Score Value (0-1)')
    ax.set_title('Violin Plot of Normalized Synthesizability Scores')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, f'violin_plot_{epoch}_{num_gen}_{known_binding_site}_{aurora}.png'))
    plt.show()
    return


def plot_synthesizability_hybridplot(df):
    sa_norm = (df['SA_score'] - 1) / 9
    sc_norm = (df['SCScore'] - 1) / 4
    np_norm = (df['NP_score'] + 5) / 10

    data = [sa_norm.dropna(), sc_norm.dropna(), np_norm.dropna()]
    labels = ['SA_score', 'SCScore', 'NP_score']
    colors = ['orange', 'blue', 'pink']

    # Means and standard errors
    means = [d.mean() for d in data]
    errors = [d.std(ddof=1) / np.sqrt(len(d)) for d in data]

    # Statistical significance: Mann-Whitney U tests
    pairs = [(0, 1), (0, 2), (1, 2)]
    sig_results = []
    for (i, j) in pairs:
        stat, p = mannwhitneyu(data[i], data[j], alternative='two-sided')
        if p < 0.05:
            sig_results.append(((i, j), p))

    # Plot setup
    fig, ax = plt.subplots(figsize=(8, 5))

    # Violin plot
    parts = ax.violinplot(data, showmeans=False, showmedians=False, widths=0.8)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.4)
        pc.set_edgecolor('black')

    # Bar plot overlay
    positions = np.arange(1, 4)
    ax.bar(positions, means, yerr=errors, color=colors, alpha=0.8, capsize=6, width=0.5, edgecolor='black', linewidth=0.8)

    # Axes
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Normalized Score Value (0–1)')
    ax.set_title('Hybrid Plot of Synthesizability Scores')
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    # Bracket layout
    y_max = max([max(d) for d in data])
    y_range = y_max - min([min(d) for d in data])
    y_offset = y_range * 0.08
    height_step = y_range * 0.12
    bracket_y = y_max + y_offset

    # Helper for significance stars
    def get_sig_stars(p):
        if p <= 0.001:
            return '***'
        elif p <= 0.01:
            return '**'
        elif p <= 0.05:
            return '*'
        else:
            return ''

    # Transparent brackets with stars + p-values
    for idx, ((i, j), p) in enumerate(sig_results):
        x1, x2 = i + 1, j + 1
        y = bracket_y + idx * height_step
        alpha_bracket = 0.5
        ax.plot([x1, x1, x2, x2],
                [y, y + y_offset, y + y_offset, y],
                lw=1.5, c='k', alpha=alpha_bracket)
        
        # Label with p-value and stars
        label = f'p = {p:.3g} {get_sig_stars(p)}'
        ax.text((x1 + x2) / 2, y + y_offset * 1.1,
                label, ha='center', va='bottom',
                fontsize=10, color='k', alpha=alpha_bracket)

    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, f'hybrid_plot_{epoch}_{num_gen}_{known_binding_site}_{aurora}.png'))
    plt.show()
    return



def plot_pairplot(df):

    cols = ['SA_score', 'SCScore', 'NP_score', 'len_smiles']
    data = df[cols].dropna()
    sns.set(style='ticks', color_codes=True)
    g = sns.pairplot(data, diag_kind='kde', plot_kws={'alpha':0.6, 's':30})
    g.fig.suptitle('Pairplot of Synthesizability and SMILES Length', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, f'pairplot_{epoch}_{num_gen}_{known_binding_site}_{aurora}.png'))
    plt.show()
    return

def plot_lipinski_violations_piechart(df):
    violation_counts = df['failed'].fillna('').apply(lambda x: 0 if x.strip() == '' else len(x.split(';')))

    violation_summary = violation_counts.value_counts().sort_index()

    plt.figure(figsize=(8, 6))
    plt.pie(
        violation_summary,
        labels=[f'{i} rule{"s" if i != 1 else ""} violated' for i in violation_summary.index],
        autopct='%1.1f%%',
        startangle=140,
        colors=plt.cm.Pastel1.colors
    )
    plt.title('Lipinski\'s Rule Violations per Molecule')
    plt.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle.
    plt.savefig(os.path.join(image_dir, f'lipinski_violations_{epoch}_{num_gen}_{known_binding_site}_{aurora}.png'))
    plt.show()
    return


def pie_chart_sa_score(df):
    labels = [
        'SA_score ≤ 2 (Very easy)',
        '2 < SA_score ≤ 3 (Easy)',
        '3 < SA_score ≤ 6.5 (Moderate)',
        'SA_score > 6.5 (Hard)'
    ]

    df['SA_bin'] = pd.cut(
        df['SA_score'],
        bins=[-float('inf'), 2, 3, 6.5, float('inf')],
        labels=labels
    )

    # Count molecules in each bin
    sa_counts = df['SA_bin'].value_counts().reindex(labels)

    # Plot the pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(
        sa_counts,
        labels=sa_counts.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=plt.cm.Set3.colors
    )
    plt.title('Synthetic Accessibility (SA_score) Distribution')
    plt.axis('equal')
    plt.savefig(os.path.join(image_dir, f'sa_score_pie_{epoch}_{num_gen}_{known_binding_site}_{aurora}.png'))
    return

def pie_chart_sc_score(df):
    labels = [
        'SCScore ≤ 1 (Very easy)',
        '1 < SCScore ≤ 3 (Easy)',
        '3 < SCScore ≤ 4.5 (Moderate)',
        'SCScore > 4.5 (Hard)'
    ]

    df['SC_bin'] = pd.cut(
        df['SCScore'],
        bins=[-float('inf'), 2, 3, 4.5, float('inf')],
        labels=labels
    )

    # Count molecules in each bin
    sa_counts = df['SC_bin'].value_counts().reindex(labels)

    # Plot the pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(
        sa_counts,
        labels=sa_counts.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=plt.cm.Set3.colors
    )
    plt.title('Synthetic Accessibility (SA_score) Distribution')
    plt.axis('equal')
    plt.savefig(os.path.join(image_dir, f'sc_score_pie_{epoch}_{num_gen}_{known_binding_site}_{aurora}.png'))
    return

def pie_chart_np_score(df):
    labels = [
        'NP_score ≤ -2 (Very synthetic-like)',
        '-2 < NP_score ≤ 0 (Balanced)',
        '0 < NP_score ≤ 2 (Natural-like)',
        'NP_score > 2 (Very natural-like)'
    ]

    df['NP_bin'] = pd.cut(
        df['NP_score'],
        bins=[-float('inf'), -2, 0, 2, float('inf')],
        labels=labels
    )

    np_counts = df['NP_bin'].value_counts().reindex(labels)

    plt.figure(figsize=(8, 6))
    plt.pie(
        np_counts,
        labels=np_counts.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=plt.cm.Set3.colors
    )
    plt.title('Natural Product-likeness (NP_score) Distribution')
    plt.axis('equal')
    plt.savefig(os.path.join(image_dir, f'np_score_pie_{epoch}_{num_gen}_{known_binding_site}_{aurora}.png'))
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Wrapper for CADD pipeline targeting Aurora protein kinases.')
    parser.add_argument('--num_gen', type=int, required=False, default=0, help='Desired number of generated molecules (int, positive)')
    parser.add_argument('--epoch', type=int, required=False, default=0, help='Epoch number the model will use to generate molecules (int, 0-99)')
    parser.add_argument('--known_binding_site', type=str, required=False, default='0', help='Allow model to use binding site information (True, False)')
    parser.add_argument('--aurora', type=str, required=False, default='B', help='Aurora kinase type (str, A, B)')
    parser.add_argument('--pdbid', type=str, required=False, default='4af3', help='Aurora kinase type (str, A, B)')
    args = parser.parse_args()

    num_gen = args.num_gen
    known_binding_site = args.known_binding_site
    epoch = args.epoch
    aurora = args.aurora
    pdbid= args.pdbid.lower()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
    sdf_folder = os.path.join(parent_dir, f'trained_model_reduced_dataset_100_epochs/gen_mols_epoch_{epoch}_mols_{num_gen}_bs_{known_binding_site}_pdbid_{pdbid}/sdf')
    known_inhib_file = os.path.join(script_dir, f'data/aurora_kinase_{aurora}_interactions.csv')
    results_dir = os.path.join(script_dir, f'results_epoch_{epoch}_mols_{num_gen}_bs_{known_binding_site}_pdbid_{pdbid}')
    image_dir = os.path.join(script_dir, f'results_epoch_{epoch}_mols_{num_gen}_bs_{known_binding_site}_pdbid_{pdbid}/images')
    # output_csv = os.path.join(results_dir, f'tanimoto_intra_{epoch}_{num_gen}_{known_binding_site}_{aurora}.csv')
    
    # os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    if epoch != 0:
        # Calculating scores for generated molecules
        mols, smiles, filenames, fps = load_mols_from_sdf_folder(sdf_folder)

    else:
        # Calculating scores for Aurora inhibitors
        mols, smiles, filenames, fps = read_aurora_kinase_interactions(known_inhib_file)

    synth_df = pd.read_csv(os.path.join(results_dir, f'synthesizability_scores_{epoch}_{num_gen}_{known_binding_site}_{pdbid}.csv'))
    lipinski_df = pd.read_csv(os.path.join(results_dir, f'lipinski_pass_{epoch}_{num_gen}_{known_binding_site}_{pdbid}.csv'))
    tanimoto_inter_df = pd.read_csv(os.path.join(results_dir, f'tanimoto_inter_{epoch}_{num_gen}_{known_binding_site}_{pdbid}.csv'))

    plot_len_smiles(synth_df)
    plot_sa_score(synth_df)
    plot_sc_score(synth_df)
    plot_np_score(synth_df)
    plot_tSNE(fps)
    plot_synthesizability_boxplot(synth_df)
    plot_synthesizability_violinplot(synth_df)
    plot_pairplot(synth_df)
    plot_lipinski_violations_piechart(lipinski_df)
    pie_chart_sa_score(synth_df)
    pie_chart_sc_score(synth_df)
    pie_chart_np_score(synth_df)
    pie_chart_len_smiles(synth_df)
    plot_synthesizability_hybridplot(synth_df)

    print(f'Plots saved to {image_dir}')