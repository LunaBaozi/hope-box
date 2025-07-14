import os
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import PIL.ImageOps
from dataclasses import dataclass, field
from enum import Enum

from rdkit import Chem
from rdkit.Chem import Draw

from scipy.stats import gaussian_kde, mannwhitneyu
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import PIL.ImageOps
import seaborn as sns
from collections import Counter

from scripts.aurk_int_preprocess import read_aurora_kinase_interactions
from scripts.gen_mols_preprocess import load_mols_from_sdf_folder
from scripts.load_config_paths import PipelinePaths


class ScoreType(Enum):
    """Enumeration for different molecular score types"""
    SA_SCORE = "SA_score"
    SC_SCORE = "SCScore"
    NP_SCORE = "NP_score"
    SMILES_LENGTH = "len_smiles"


@dataclass
class PlotConfiguration:
    """Centralized configuration for all plot parameters"""
    # Figure settings
    FIGURE_SIZE: Tuple[int, int] = (8, 5)
    PIE_FIGURE_SIZE: Tuple[int, int] = (8, 6)
    DPI: int = 300
    
    # Colors
    COLORS: Dict[str, str] = field(default_factory=lambda: {
        'sa_score': 'orange',
        'sc_score': 'blue', 
        'np_score': 'pink',
        'len_smiles': 'green'
    })
    
    # Score configurations
    SCORE_CONFIGS: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'SA_score': {
            'bins': np.arange(0.75, 10.25, 0.5),
            'x_range': (0.5, 10.5),
            'title': 'Histogram of SA_Score',
            'xlabel': 'SA_Score (centered bins 1-10)',
            'minimize': True,
            'norm_range': (1, 10)
        },
        'SCScore': {
            'bins': np.arange(0.75, 5.25, 0.25),
            'x_range': (0.5, 5.5),
            'title': 'Histogram of SCScore',
            'xlabel': 'SCScore',
            'minimize': True,
            'norm_range': (1, 5)
        },
        'NP_score': {
            'bins': np.arange(-5.75, 5.25, 0.5),
            'x_range': (-5.5, 5.5),
            'title': 'Histogram of NP_score',
            'xlabel': 'NP_score',
            'minimize': True,
            'norm_range': (-5, 5)
        },
        'len_smiles': {
            'title': 'Histogram of SMILES Length',
            'xlabel': 'SMILES length',
            'minimize': False,
            'norm_range': None
        }
    })
    
    # Pie chart configurations
    PIE_CHART_CONFIGS: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'SA_score': {
            'bins': [-float('inf'), 2, 3, 6.5, float('inf')],
            'labels': ['SA_score ≤ 2 (Very easy)', '2 < SA_score ≤ 3 (Easy)',
                      '3 < SA_score ≤ 6.5 (Moderate)', 'SA_score > 6.5 (Hard)'],
            'title': 'Synthetic Accessibility (SA_score) Distribution'
        },
        'SCScore': {
            'bins': [-float('inf'), 2, 3, 4.5, float('inf')],
            'labels': ['SCScore ≤ 1 (Very easy)', '1 < SCScore ≤ 3 (Easy)',
                      '3 < SCScore ≤ 4.5 (Moderate)', 'SCScore > 4.5 (Hard)'],
            'title': 'Synthetic Complexity (SCScore) Distribution'
        },
        'NP_score': {
            'bins': [-float('inf'), -2, 0, 2, float('inf')],
            'labels': ['NP_score ≤ -2 (Very synthetic-like)', '-2 < NP_score ≤ 0 (Balanced)',
                      '0 < NP_score ≤ 2 (Natural-like)', 'NP_score > 2 (Very natural-like)'],
            'title': 'Natural Product-likeness (NP_score) Distribution'
        },
        'len_smiles': {
            'bins': [0, 20, 40, 60, 80, float('inf')],
            'labels': ['≤20 chars', '21-40 chars', '41-60 chars', '61-80 chars', '>80 chars'],
            'title': 'SMILES Length Distribution'
        }
    })


@dataclass
class PlotContext:
    """Context object containing all necessary parameters for plotting"""
    epoch: int
    num_gen: int
    known_binding_site: str
    aurora: str
    pdbid: str
    image_dir: Path
    
    def get_filename(self, plot_type: str) -> str:
        """Generate standardized filename for plots"""
        return f'{plot_type}.png'
    
    def get_filepath(self, plot_type: str) -> Path:
        """Get full filepath for saving plots"""
        return self.image_dir / self.get_filename(plot_type)


class MolecularVisualizationSuite:
    """
    A comprehensive suite for visualizing molecular properties and synthesizability metrics.
    """
    
    # Color schemes and plotting constants
    COLORS = {
        'primary': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
        'synthesizability': ['orange', 'blue', 'pink'],
        'gradients': {'green': ('green', 'darkgreen'), 'orange': ('orange', 'darkorange'), 
                     'blue': ('blue', 'darkblue'), 'pink': ('pink', 'deeppink')}
    }
    
    FIGURE_SIZE = (8, 5)
    LARGE_FIGURE_SIZE = (8, 6)
    MOL_IMAGE_SIZE = (1800, 1800)
    
    def __init__(self, epoch: int, num_gen: int, known_binding_site: str, 
                 aurora: str, pdbid: str, image_dir: str):
        """Initialize the visualization suite with experiment parameters."""
        self.epoch = epoch
        self.num_gen = num_gen
        self.known_binding_site = known_binding_site
        self.aurora = aurora
        self.pdbid = pdbid
        self.image_dir = Path(image_dir)
        
        # Ensure image directory exists
        self.image_dir.mkdir(parents=True, exist_ok=True)
        
    def _save_figure(self, filename: str, dpi: int = 300) -> None:
        """Save figure with consistent formatting."""
        filepath = self.image_dir / f"{filename}.png"
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        
    def _add_kde_overlay(self, ax, values: np.ndarray, x_range: Tuple[float, float], 
                        color: str, bins: np.ndarray) -> None:
        """Add KDE overlay to histogram."""
        try:
            # Check for sufficient data diversity
            if len(values) < 3 or np.std(values) < 1e-10:
                return  # Skip KDE for insufficient or non-diverse data
                
            kde = gaussian_kde(values)
            x_vals = np.linspace(x_range[0], x_range[1], 300)
            kde_vals = kde(x_vals) * len(values) * (bins[1] - bins[0])
            ax.plot(x_vals, kde_vals, color=color, lw=2, label='KDE')
        except (ImportError, np.linalg.LinAlgError):
            pass  # Skip KDE if not available or singular matrix
            
    def _add_molecule_inset(self, fig, ax, df: pd.DataFrame, score_column: str, 
                           minimize: bool = True) -> None:
        """Add molecule structure inset to plot."""
        if 'smiles' not in df.columns:
            return
            
        idx = df[score_column].idxmin() if minimize else df[score_column].idxmax()
        smiles = df.loc[idx, 'smiles']
        score = df.loc[idx, score_column]
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return
            
        img = Draw.MolToImage(mol, size=self.MOL_IMAGE_SIZE)
        img = PIL.ImageOps.expand(img, border=8, fill='black')
        
        filename = df.loc[idx, 'filename'] if 'filename' in df.columns else ''
        text = f'Filename:\n{filename}\n{score_column}: {score:.2f}'
        
        # Position inset relative to legend
        legend = ax.get_legend()
        if legend:
            fig.canvas.draw()
            legend_box = legend.get_window_extent(fig.canvas.get_renderer())
            legend_box = legend_box.transformed(fig.transFigure.inverted())
            
            inset_width, inset_height = 0.3, 0.6
            inset_x = legend_box.x1 - inset_width
            inset_y = legend_box.y0 - inset_height - 0.02
            
            inset_ax = fig.add_axes([inset_x, inset_y, inset_width, inset_height])
            inset_ax.axis('off')
            inset_ax.imshow(img)
            inset_ax.text(0.5, 0.05, text, ha='center', va='bottom', fontsize=6,
                         color='white', bbox=dict(facecolor='black', alpha=0.6, 
                         boxstyle='round,pad=0.3'), transform=inset_ax.transAxes)
    
    def _create_histogram_base(self, df: pd.DataFrame, column: str, title: str, 
                              xlabel: str, color: str, bins: np.ndarray = None,
                              x_range: Optional[Tuple[float, float]] = None) -> Tuple[plt.Figure, plt.Axes]:
        """Create base histogram with common formatting."""
        fig, ax = plt.subplots(figsize=self.FIGURE_SIZE)
        values = df[column].dropna().values
        
        if bins is None:
            bins = np.linspace(values.min() - 0.5, values.max() + 0.5, 30)
        if x_range is None:
            x_range = (values.min(), values.max())
            
        ax.hist(values, bins=bins, color=color, edgecolor='black', 
               rwidth=0.8, alpha=0.6, label='Histogram')
        
        # Add KDE overlay
        kde_color = self.COLORS['gradients'].get(color, (color, color))[1]
        self._add_kde_overlay(ax, values, x_range, kde_color, bins)
        
        # Add mean line
        mean_val = values.mean()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean = {mean_val:.2f}')
        
        # Formatting
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Count of Molecules')
        ax.set_title(title)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend()
        plt.tight_layout()
        
        return fig, ax
    
    def plot_smiles_length_distribution(self, df: pd.DataFrame) -> None:
        """Plot SMILES length distribution with histogram and KDE."""
        fig, ax = self._create_histogram_base(
            df, 'len_smiles', 'Histogram of SMILES Length', 'SMILES length', 'green'
        )
        plt.show()
        self._save_figure('len_smiles')
        
    def plot_smiles_length_pie_chart(self, df: pd.DataFrame) -> None:
        """Plot SMILES length distribution as pie chart."""
        bins = [0, 20, 40, 60, 80, float('inf')]
        labels = ['≤20 chars', '21-40 chars', '41-60 chars', '61-80 chars', '>80 chars']
        
        df_temp = df.copy()
        df_temp['len_bin'] = pd.cut(df_temp['len_smiles'], bins=bins, labels=labels, 
                                   right=True, include_lowest=True)
        counts = df_temp['len_bin'].value_counts().reindex(labels)
        
        plt.figure(figsize=self.LARGE_FIGURE_SIZE)
        plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140,
               colors=plt.cm.Set2.colors)
        plt.title('SMILES Length Distribution')
        plt.axis('equal')
        plt.show()
        self._save_figure('len_smiles_pie')
        
    def plot_synthesizability_score(self, df: pd.DataFrame, score_type: str) -> None:
        """Plot synthesizability scores (SA_score, SCScore, or NP_score)."""
        config = {
            'SA_score': {'range': (1, 10), 'bins': np.arange(0.75, 10.25, 0.5), 
                        'color': 'orange', 'title': 'Histogram of SA_Score (centered bins 1-10)'},
            'SCScore': {'range': (1, 5), 'bins': np.arange(0.75, 5.25, 0.25), 
                       'color': 'blue', 'title': 'Histogram of SCScore'},
            'NP_score': {'range': (-5, 5), 'bins': np.arange(-5.75, 5.25, 0.5), 
                        'color': 'pink', 'title': 'Histogram of NP_score'}
        }
        
        if score_type not in config:
            raise ValueError(f"Unknown score type: {score_type}")
            
        cfg = config[score_type]
        fig, ax = self._create_histogram_base(
            df, score_type, cfg['title'], score_type, cfg['color'], 
            bins=cfg['bins'], x_range=cfg['range']
        )
        
        # Add molecule inset for best score
        self._add_molecule_inset(fig, ax, df, score_type, minimize=True)
        
        plt.show()
        self._save_figure(score_type.lower().replace('_', ''))
        
    def plot_tsne_chemical_space(self, fps: List, synth_df: pd.DataFrame) -> None:
        """Plot t-SNE visualization of chemical space."""
        n_samples = len(fps)
        
        # Dynamic perplexity calculation
        if n_samples <= 5:
            perplexity = max(2, n_samples - 1)
        elif n_samples < 50:
            perplexity = max(5, n_samples // 3)
        else:
            perplexity = 30
        perplexity = min(perplexity, n_samples - 1)
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        tsne_results = tsne.fit_transform(np.stack(fps))
        
        plt.figure(figsize=self.LARGE_FIGURE_SIZE)
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], 
                            c=synth_df['SA_score'].values, cmap='viridis', 
                            alpha=0.7, s=30)
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.title('t-SNE of Chemical Space (colored by SA_score)')
        plt.colorbar(scatter, label='SA_score')
        plt.tight_layout()
        plt.show()
        self._save_figure('tSNE')
        
    def plot_synthesizability_comparison(self, df: pd.DataFrame, plot_type: str = 'boxplot') -> None:
        """Plot comparison of synthesizability scores."""
        # Normalize scores to 0-1 range
        normalized_data = {
            'SA_score': (df['SA_score'] - 1) / 9,
            'SCScore': (df['SCScore'] - 1) / 4,
            'NP_score': (df['NP_score'] + 5) / 10
        }
        
        data = [normalized_data[score].dropna().values for score in normalized_data]
        labels = list(normalized_data.keys())
        
        fig, ax = plt.subplots(figsize=self.FIGURE_SIZE)
        
        if plot_type == 'boxplot':
            self._create_boxplot(ax, data, labels)
        elif plot_type == 'violinplot':
            self._create_violinplot(ax, data, labels)
        elif plot_type == 'hybrid':
            self._create_hybrid_plot(ax, data, labels)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
            
        plt.tight_layout()
        plt.show()
        self._save_figure(plot_type)
        
    def _create_boxplot(self, ax, data: List, labels: List[str]) -> None:
        """Create boxplot of synthesizability scores."""
        ax.boxplot(data, patch_artist=True, tick_labels=labels, showmeans=True,
                  boxprops=dict(facecolor='lightgray', color='black'),
                  medianprops=dict(color='red'),
                  meanprops=dict(marker='o', markerfacecolor='blue', markeredgecolor='black'))
        ax.set_ylabel('Normalized Score Value (0-1)')
        ax.set_title('Boxplot of Normalized Synthesizability Scores')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
    def _create_violinplot(self, ax, data: List, labels: List[str]) -> None:
        """Create violin plot with statistical significance testing."""
        # Statistical significance testing
        sig_labels = self._add_significance_labels(data, labels)
        
        parts = ax.violinplot(data, showmeans=True, showmedians=True)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(self.COLORS['synthesizability'][i])
            pc.set_alpha(0.5)
            
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(sig_labels)
        ax.set_ylabel('Normalized Score Value (0-1)')
        ax.set_title('Violin Plot of Normalized Synthesizability Scores')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
    def _create_hybrid_plot(self, ax, data: List, labels: List[str]) -> None:
        """Create hybrid violin + bar plot with significance brackets."""
        colors = self.COLORS['synthesizability']
        
        # Calculate means and errors
        means = [np.mean(d) for d in data]
        errors = [np.std(d, ddof=1) / np.sqrt(len(d)) for d in data]
        
        # Violin plot base
        parts = ax.violinplot(data, showmeans=False, showmedians=False, widths=0.8)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.4)
            pc.set_edgecolor('black')
            
        # Bar plot overlay
        positions = np.arange(1, 4)
        ax.bar(positions, means, yerr=errors, color=colors, alpha=0.8, 
              capsize=6, width=0.5, edgecolor='black', linewidth=0.8)
        
        # Add significance brackets
        self._add_significance_brackets(ax, data)
        
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Normalized Score Value (0–1)')
        ax.set_title('Hybrid Plot of Synthesizability Scores')
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        
    def _add_significance_labels(self, data: List, labels: List[str]) -> List[str]:
        """Add significance markers to labels based on Mann-Whitney U tests."""
        sig_labels = labels.copy()
        alpha = 0.05 / 3  # Bonferroni correction
        pairs = [(0, 1), (0, 2), (1, 2)]
        sig_matrix = [False] * 3
        
        for i, j in pairs:
            _, p = mannwhitneyu(data[i], data[j], alternative='two-sided')
            if p < alpha:
                sig_matrix[i] = sig_matrix[j] = True
                
        for i in range(3):
            if sig_matrix[i]:
                sig_labels[i] += ' ***'
                
        return sig_labels
        
    def _add_significance_brackets(self, ax, data: List) -> None:
        """Add significance brackets to hybrid plot."""
        pairs = [(0, 1), (0, 2), (1, 2)]
        sig_results = []
        
        for i, j in pairs:
            _, p = mannwhitneyu(data[i], data[j], alternative='two-sided')
            if p < 0.05:
                sig_results.append(((i, j), p))
                
        if not sig_results:
            return
            
        y_max = max(max(d) for d in data)
        y_range = y_max - min(min(d) for d in data)
        y_offset = y_range * 0.08
        height_step = y_range * 0.12
        bracket_y = y_max + y_offset
        
        def get_sig_stars(p):
            if p <= 0.001: return '***'
            elif p <= 0.01: return '**'
            elif p <= 0.05: return '*'
            return ''
            
        for idx, ((i, j), p) in enumerate(sig_results):
            x1, x2 = i + 1, j + 1
            y = bracket_y + idx * height_step
            
            ax.plot([x1, x1, x2, x2], [y, y + y_offset, y + y_offset, y],
                   lw=1.5, c='k', alpha=0.5)
            
            label = f'p = {p:.3g} {get_sig_stars(p)}'
            ax.text((x1 + x2) / 2, y + y_offset * 1.1, label,
                   ha='center', va='bottom', fontsize=10, color='k', alpha=0.5)
                   
    def plot_property_pairplot(self, df: pd.DataFrame) -> None:
        """Create pairplot of molecular properties."""
        cols = ['SA_score', 'SCScore', 'NP_score', 'len_smiles']
        data = df[cols].dropna()
        
        sns.set(style='ticks', color_codes=True)
        g = sns.pairplot(data, diag_kind='kde', plot_kws={'alpha': 0.6, 's': 30})
        g.fig.suptitle('Pairplot of Synthesizability and SMILES Length', y=1.02)
        plt.tight_layout()
        plt.show()
        self._save_figure('pairplot')
        
    def plot_lipinski_violations(self, df: pd.DataFrame) -> None:
        """Plot Lipinski rule violations as pie chart."""
        violation_counts = df['failed'].fillna('').apply(
            lambda x: 0 if x.strip() == '' else len(x.split(';'))
        )
        violation_summary = violation_counts.value_counts().sort_index()
        
        plt.figure(figsize=self.LARGE_FIGURE_SIZE)
        plt.pie(violation_summary,
               labels=[f'{i} rule{"s" if i != 1 else ""} violated' for i in violation_summary.index],
               autopct='%1.1f%%', startangle=140, colors=plt.cm.Pastel1.colors)
        plt.title('Lipinski\'s Rule Violations per Molecule')
        plt.axis('equal')
        plt.show()
        self._save_figure('lipinski_violations')
        
    def create_score_pie_chart(self, df: pd.DataFrame, score_type: str) -> None:
        """Create pie chart for score distributions."""
        config = {
            'SA_score': {
                'bins': [-float('inf'), 2, 3, 6.5, float('inf')],
                'labels': ['SA_score ≤ 2 (Very easy)', '2 < SA_score ≤ 3 (Easy)',
                          '3 < SA_score ≤ 6.5 (Moderate)', 'SA_score > 6.5 (Hard)'],
                'title': 'Synthetic Accessibility (SA_score) Distribution'
            },
            'SCScore': {
                'bins': [-float('inf'), 2, 3, 4.5, float('inf')],
                'labels': ['SCScore ≤ 1 (Very easy)', '1 < SCScore ≤ 3 (Easy)',
                          '3 < SCScore ≤ 4.5 (Moderate)', 'SCScore > 4.5 (Hard)'],
                'title': 'Synthetic Complexity (SCScore) Distribution'
            },
            'NP_score': {
                'bins': [-float('inf'), -2, 0, 2, float('inf')],
                'labels': ['NP_score ≤ -2 (Very synthetic-like)', '-2 < NP_score ≤ 0 (Balanced)',
                          '0 < NP_score ≤ 2 (Natural-like)', 'NP_score > 2 (Very natural-like)'],
                'title': 'Natural Product-likeness (NP_score) Distribution'
            }
        }
        
        if score_type not in config:
            raise ValueError(f"Unknown score type: {score_type}")
            
        cfg = config[score_type]
        df_temp = df.copy()
        df_temp['score_bin'] = pd.cut(df_temp[score_type], bins=cfg['bins'], 
                                     labels=cfg['labels'])
        counts = df_temp['score_bin'].value_counts().reindex(cfg['labels'])
        
        plt.figure(figsize=self.LARGE_FIGURE_SIZE)
        plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140,
               colors=plt.cm.Set3.colors)
        plt.title(cfg['title'])
        plt.axis('equal')
        plt.show()
        self._save_figure(f'{score_type.lower().replace("_", "")}_pie')
        
    def generate_all_plots(self, synth_df: pd.DataFrame, lipinski_df: pd.DataFrame, 
                          tanimoto_inter_df: pd.DataFrame, fps: List,
                          tanimoto_intra_df: Optional[pd.DataFrame] = None) -> None:
        """Generate all visualization plots."""
        print("Generating molecular property visualizations...")
        
        # SMILES length plots
        self.plot_smiles_length_distribution(synth_df)
        self.plot_smiles_length_pie_chart(synth_df)
        
        # Synthesizability score plots
        for score_type in ['SA_score', 'SCScore', 'NP_score']:
            self.plot_synthesizability_score(synth_df, score_type)
            self.create_score_pie_chart(synth_df, score_type)
            
        # Comparison plots
        self.plot_synthesizability_comparison(synth_df, 'boxplot')
        self.plot_synthesizability_comparison(synth_df, 'violinplot')
        self.plot_synthesizability_comparison(synth_df, 'hybrid')
        
        # Chemical space and property relationships
        self.plot_tsne_chemical_space(fps, synth_df)
        self.plot_property_pairplot(synth_df)
        
        # Lipinski violations
        self.plot_lipinski_violations(lipinski_df)
        
        # Tanimoto similarity analysis
        print("Generating Tanimoto similarity visualizations...")
        try:
            if not tanimoto_inter_df.empty:
                self.plot_tanimoto_similarity_heatmap(tanimoto_inter_df, 'inter')
                self.plot_tanimoto_distribution(tanimoto_inter_df, 'inter')
        except Exception as e:
            print(f"Warning: Failed to generate inter-molecular Tanimoto plots: {e}")
        
        try:
            if tanimoto_intra_df is not None and not tanimoto_intra_df.empty:
                self.plot_tanimoto_similarity_heatmap(tanimoto_intra_df, 'intra')
                self.plot_tanimoto_distribution(tanimoto_intra_df, 'intra')
        except Exception as e:
            print(f"Warning: Failed to generate intra-molecular Tanimoto plots: {e}")
            
        # Comparison plot if both datasets available
        try:
            if (not tanimoto_inter_df.empty and 
                tanimoto_intra_df is not None and not tanimoto_intra_df.empty):
                self.plot_tanimoto_summary_comparison(tanimoto_inter_df, tanimoto_intra_df)
            elif not tanimoto_inter_df.empty:
                self.plot_tanimoto_summary_comparison(tanimoto_inter_df)
        except Exception as e:
            print(f"Warning: Failed to generate Tanimoto comparison plot: {e}")
        
        print(f"All plots saved to {self.image_dir}")
    
    def plot_tanimoto_similarity_heatmap(self, tanimoto_df: pd.DataFrame, 
                                        similarity_type: str = 'inter',
                                        threshold: float = 0.5,
                                        show_labels: bool = True) -> None:
        """
        Plot Tanimoto similarity as a heatmap with selective labeling.
        
        Args:
            tanimoto_df: DataFrame containing similarity matrix and filenames
            similarity_type: Type of similarity ('inter' or 'intra')
            threshold: Threshold for displaying labels (default: 0.5)
            show_labels: Whether to show filename labels for high-similarity pairs
        """
        try:
            # Extract similarity matrix and filenames from DataFrame
            mat, filenames1, filenames2 = self._extract_similarity_data(tanimoto_df)
            
            if mat is None:
                print(f"Warning: Could not extract similarity matrix from {similarity_type} DataFrame")
                return
                
            self._create_tanimoto_heatmap(mat, filenames1, filenames2, similarity_type, threshold, show_labels)
            
        except Exception as e:
            print(f"Error creating {similarity_type} Tanimoto heatmap: {e}")
            print(f"DataFrame shape: {tanimoto_df.shape}")
            print(f"DataFrame columns: {list(tanimoto_df.columns)}")
            
    def _extract_similarity_data(self, tanimoto_df: pd.DataFrame) -> Tuple[Optional[np.ndarray], List[str], List[str]]:
        """Extract similarity matrix and filenames from DataFrame with robust error handling."""
        mat = None
        filenames1 = filenames2 = []
        
        # Method 1: Try to extract from pairwise similarity data
        if 'tanimoto' in tanimoto_df.columns or 'similarity' in tanimoto_df.columns:
            try:
                mat, filenames1, filenames2 = self._build_matrix_from_pairs(tanimoto_df)
                if mat is not None:
                    return mat, filenames1, filenames2
            except Exception as e:
                print(f"Failed to build matrix from pairs: {e}")
        
        # Method 2: Try serialized matrix
        if 'similarity_matrix' in tanimoto_df.columns:
            try:
                import ast
                mat_data = []
                for row in tanimoto_df['similarity_matrix']:
                    if isinstance(row, str):
                        parsed = ast.literal_eval(row)
                        mat_data.append(parsed)
                    else:
                        mat_data.append(row)
                mat = np.array(mat_data, dtype=float)
            except Exception as e:
                print(f"Failed to parse serialized matrix: {e}")
        
        # Method 3: Try matrix columns
        if mat is None:
            matrix_cols = [col for col in tanimoto_df.columns if col.startswith('mol_')]
            if matrix_cols:
                try:
                    mat = tanimoto_df[matrix_cols].values.astype(float)
                except Exception as e:
                    print(f"Failed to convert matrix columns to float: {e}")
                    # Try to handle mixed data types
                    try:
                        mat_data = []
                        for _, row in tanimoto_df[matrix_cols].iterrows():
                            row_data = []
                            for val in row:
                                if pd.isna(val):
                                    row_data.append(0.0)
                                elif isinstance(val, (int, float)):
                                    row_data.append(float(val))
                                elif isinstance(val, str):
                                    try:
                                        row_data.append(float(val))
                                    except ValueError:
                                        row_data.append(0.0)
                                else:
                                    row_data.append(0.0)
                            mat_data.append(row_data)
                        mat = np.array(mat_data, dtype=float)
                    except Exception as e2:
                        print(f"Failed to manually convert matrix data: {e2}")
        
        # Get filenames
        if 'filename1' in tanimoto_df.columns and 'filename2' in tanimoto_df.columns:
            filenames1 = tanimoto_df['filename1'].unique().tolist()
            filenames2 = tanimoto_df['filename2'].unique().tolist()
        elif 'filename' in tanimoto_df.columns:
            filenames1 = filenames2 = tanimoto_df['filename'].unique().tolist()
        elif mat is not None:
            # Generate default filenames based on matrix size
            n = mat.shape[0]
            filenames1 = filenames2 = [f'mol_{i}' for i in range(n)]
        
        return mat, filenames1, filenames2
        
    def _build_matrix_from_pairs(self, df: pd.DataFrame) -> Tuple[Optional[np.ndarray], List[str], List[str]]:
        """Build similarity matrix from pairwise similarity data."""
        similarity_col = 'tanimoto_similarity' if 'tanimoto' in df.columns else 'similarity'
        
        if 'filename1' in df.columns and 'filename2' in df.columns:
            # Pairwise format
            filenames1 = sorted(df['filename1'].unique())
            filenames2 = sorted(df['filename2'].unique())
            
            n1, n2 = len(filenames1), len(filenames2)
            mat = np.zeros((n2, n1))  # Note: transposed for proper indexing
            
            filename1_to_idx = {name: i for i, name in enumerate(filenames1)}
            filename2_to_idx = {name: i for i, name in enumerate(filenames2)}
            
            for _, row in df.iterrows():
                i = filename2_to_idx.get(row['filename2'])
                j = filename1_to_idx.get(row['filename1'])
                if i is not None and j is not None:
                    try:
                        mat[i, j] = float(row[similarity_col])
                    except (ValueError, TypeError):
                        mat[i, j] = 0.0
                        
            return mat, filenames1, filenames2
        
        return None, [], []
        
    def _create_tanimoto_heatmap(self, mat: np.ndarray, filenames1: List[str], 
                                filenames2: List[str], similarity_type: str,
                                threshold: float, show_labels: bool) -> None:
        """Create the actual Tanimoto similarity heatmap."""
        # Create mask for lower triangle (for inter-molecular comparisons)
        mask = np.tril(np.ones_like(mat, dtype=bool))
        
        plt.figure(figsize=(10, 8))
        
        # Apply mask to show only lower triangle
        masked_mat = np.where(mask, mat, np.nan)
        
        # Create heatmap
        im = plt.imshow(masked_mat, cmap='viridis', vmin=0, vmax=1, aspect='auto')
        plt.colorbar(im, label='Tanimoto Similarity', shrink=0.8)
        
        plt.title(f'{"Inter" if similarity_type == "inter" else "Intra"}-molecular Tanimoto Similarity Heatmap')
        plt.xlabel('Molecule Index')
        plt.ylabel('Molecule Index')
        
        if show_labels and len(filenames1) < 50:  # Only show labels for manageable number of molecules
            # Find high-scoring pairs
            high_pairs = np.argwhere(
                (mat >= threshold) & mask & 
                (np.arange(mat.shape[0])[:, None] > np.arange(mat.shape[1]))
            )
            
            # Initialize empty label lists
            n_labels1, n_labels2 = len(filenames1), len(filenames2)
            xtick_labels = ['' for _ in range(n_labels1)]
            ytick_labels = ['' for _ in range(n_labels2)]
            
            # Set labels only for high-similarity pairs
            for i, j in high_pairs:
                if i < len(filenames2) and j < len(filenames1):
                    ytick_labels[i] = filenames2[i]
                    xtick_labels[j] = filenames1[j]
            
            # Apply tick labels
            plt.xticks(
                ticks=range(n_labels1),
                labels=xtick_labels,
                rotation=45, ha='right', fontsize=7
            )
            plt.yticks(
                ticks=range(n_labels2),
                labels=ytick_labels,
                rotation=0, ha='right', fontsize=7
            )
        
        plt.tight_layout()
        plt.show()
        self._save_figure(f'tanimoto_heatmap_{similarity_type}')
        
    def plot_tanimoto_distribution(self, tanimoto_df: pd.DataFrame, 
                                  similarity_type: str = 'inter') -> None:
        """
        Plot distribution of Tanimoto similarity values.
        
        Args:
            tanimoto_df: DataFrame containing similarity values
            similarity_type: Type of similarity ('inter' or 'intra')
        """
        try:
            # Extract similarity values
            similarities = self._extract_similarity_values(tanimoto_df)
            
            if similarities is None or len(similarities) == 0:
                print(f"Warning: No similarity values found in {similarity_type} DataFrame")
                return
            
            # Create histogram
            fig, ax = self._create_histogram_base(
                pd.DataFrame({'tanimoto_similarity': similarities}),
                'tanimoto_similarity',
                f'Distribution of {similarity_type.title()}-molecular Tanimoto Similarities',
                'Tanimoto Similarity',
                'skyblue',
                bins=np.linspace(0, 1, 21)  # 20 bins from 0 to 1
            )
            
            # Add statistics text box
            stats_text = f"""Statistics:
Mean: {similarities.mean():.3f}
Median: {similarities.median():.3f}
Std: {similarities.std():.3f}
Min: {similarities.min():.3f}
Max: {similarities.max():.3f}
N pairs: {len(similarities)}"""
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            plt.show()
            self._save_figure(f'tanimoto_distribution_{similarity_type}')
            
        except Exception as e:
            print(f"Error creating {similarity_type} Tanimoto distribution: {e}")
            
    def _extract_similarity_values(self, tanimoto_df: pd.DataFrame) -> Optional[pd.Series]:
        """Extract similarity values from DataFrame with robust error handling."""
        # Try direct similarity columns first
        for col_name in ['tanimoto_similarity', 'similarity']:
            if col_name in tanimoto_df.columns:
                try:
                    similarities = tanimoto_df[col_name].dropna()
                    # Convert to numeric, handling string values
                    similarities = pd.to_numeric(similarities, errors='coerce').dropna()
                    if len(similarities) > 0:
                        return similarities
                except Exception as e:
                    print(f"Failed to extract from {col_name}: {e}")
        
        # Try to extract from matrix format
        try:
            mat, _, _ = self._extract_similarity_data(tanimoto_df)
            if mat is not None:
                # Get upper triangle values (excluding diagonal)
                triu_indices = np.triu_indices_from(mat, k=1)
                similarities = mat[triu_indices]
                similarities = pd.Series(similarities).dropna()
                return similarities
        except Exception as e:
            print(f"Failed to extract from matrix: {e}")
        
        return None
        
    def plot_tanimoto_summary_comparison(self, tanimoto_inter_df: pd.DataFrame, 
                                       tanimoto_intra_df: Optional[pd.DataFrame] = None) -> None:
        """
        Create comparison plot of inter vs intra-molecular similarities.
        
        Args:
            tanimoto_inter_df: DataFrame with inter-molecular similarities
            tanimoto_intra_df: Optional DataFrame with intra-molecular similarities
        """
        try:
            fig, axes = plt.subplots(1, 2 if tanimoto_intra_df is not None else 1, 
                                    figsize=(12 if tanimoto_intra_df is not None else 6, 5))
            
            if tanimoto_intra_df is None:
                axes = [axes]
            
            # Extract similarities using robust method
            datasets = [('Inter-molecular', tanimoto_inter_df, 'blue')]
            if tanimoto_intra_df is not None:
                datasets.append(('Intra-molecular', tanimoto_intra_df, 'red'))
            
            similarities_data = []
            for name, df, color in datasets:
                sims = self._extract_similarity_values(df)
                if sims is not None and len(sims) > 0:
                    similarities_data.append((name, sims, color))
                else:
                    print(f"Warning: No valid similarity data found for {name}")
            
            if not similarities_data:
                print("Warning: No valid similarity data found for comparison")
                return
            
            # Plot distributions
            for i, (name, sims, color) in enumerate(similarities_data):
                ax = axes[i] if len(axes) > 1 else axes[0]
                
                # Histogram
                ax.hist(sims, bins=20, alpha=0.7, color=color, edgecolor='black', density=True)
                
                # KDE overlay
                try:
                    kde = gaussian_kde(sims)
                    x_vals = np.linspace(0, 1, 100)
                    ax.plot(x_vals, kde(x_vals), color='darkred' if 'red' in color else 'darkblue', lw=2)
                except:
                    pass
                
                # Statistics
                mean_sim = sims.mean()
                ax.axvline(mean_sim, color='red', linestyle='--', linewidth=2, 
                          label=f'Mean = {mean_sim:.3f}')
                
                ax.set_xlabel('Tanimoto Similarity')
                ax.set_ylabel('Density')
                ax.set_title(f'{name} Similarities')
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
            
            # If both datasets, add comparison
            if len(similarities_data) == 2:
                try:
                    # Statistical comparison
                    _, p_value = mannwhitneyu(similarities_data[0][1], similarities_data[1][1], 
                                            alternative='two-sided')
                    
                    fig.suptitle(f'Tanimoto Similarity Comparison (p-value: {p_value:.3e})', fontsize=14)
                except Exception as e:
                    print(f"Warning: Statistical comparison failed: {e}")
            
            plt.tight_layout()
            plt.show()
            self._save_figure('tanimoto_comparison')
            
        except Exception as e:
            print(f"Error creating Tanimoto comparison plot: {e}")

    # ...existing code...


def load_molecular_data(paths: PipelinePaths, epoch: int, num_gen: int, 
                       known_binding_site: str, pdbid: str, aurora: str):
    """Load molecular data from various sources."""
    if epoch != 0:
        # Load generated molecules
        sdf_folder = paths.graphbp_sdf_path(epoch, num_gen, known_binding_site, pdbid)
        mols, smiles, filenames, fps = load_mols_from_sdf_folder(sdf_folder)
    else:
        # Load Aurora inhibitors
        aurora_data_file = paths.aurora_data_path(aurora)
        mols, smiles, filenames, fps = read_aurora_kinase_interactions(aurora_data_file)
    
    # Load analysis results
    synth_df = pd.read_csv(paths.synthesizability_output_path(epoch, num_gen, known_binding_site, pdbid))
    lipinski_df = pd.read_csv(paths.lipinski_output_path(epoch, num_gen, known_binding_site, pdbid))
    tanimoto_inter_df = pd.read_csv(paths.output_path(epoch, num_gen, known_binding_site, pdbid, None, 'tanimoto_inter'))
    
    # Try to load intra-molecular Tanimoto data if available
    try:
        tanimoto_intra_df = pd.read_csv(paths.output_path(epoch, num_gen, known_binding_site, pdbid, None, 'tanimoto_intra'))
    except FileNotFoundError:
        tanimoto_intra_df = None
        print("Intra-molecular Tanimoto data not found, skipping...")
    
    return (mols, smiles, filenames, fps), synth_df, lipinski_df, tanimoto_inter_df, tanimoto_intra_df


def main():
    """Main execution function."""
    # Setup paths and argument parsing
    paths = PipelinePaths()
    parser = argparse.ArgumentParser(
        description='Comprehensive molecular visualization suite for CADD pipeline targeting Aurora kinases.'
    )
    
    parser.add_argument('--num_gen', type=int, default=0,
                       help='Number of generated molecules (default: 0)')
    parser.add_argument('--epoch', type=int, default=0,
                       help='Model epoch for generation (0-99, default: 0)')
    parser.add_argument('--known_binding_site', type=str, default='0',
                       help='Use binding site information (default: "0")')
    parser.add_argument('--aurora', type=str, default='B', choices=['A', 'B'],
                       help='Aurora kinase type (default: B)')
    parser.add_argument('--pdbid', type=str, default='4af3',
                       help='PDB ID for analysis (default: 4af3)')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Custom output file path (optional)')
    
    args = parser.parse_args()
    
    # Process arguments
    epoch, num_gen = args.epoch, args.num_gen
    known_binding_site, aurora = args.known_binding_site, args.aurora
    pdbid = args.pdbid.lower()
    
    # Setup directories
    results_dir = Path(paths.synthesizability_output_path(epoch, num_gen, known_binding_site, pdbid)).parent
    image_dir = results_dir / 'images'
    
    # Create output directories
    output_csv = paths.output_path(epoch, num_gen, known_binding_site, pdbid, args.output_file, 'tanimoto_inter')
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    
    # Load molecular data
    print("Loading molecular data...")
    (mols, smiles, filenames, fps), synth_df, lipinski_df, tanimoto_inter_df, tanimoto_intra_df = load_molecular_data(
        paths, epoch, num_gen, known_binding_site, pdbid, aurora
    )
    
    # Initialize visualization suite
    viz_suite = MolecularVisualizationSuite(
        epoch, num_gen, known_binding_site, aurora, pdbid, str(image_dir)
    )
    
    # Generate all plots
    viz_suite.generate_all_plots(synth_df, lipinski_df, tanimoto_inter_df, fps, tanimoto_intra_df)


if __name__ == '__main__':
    main()