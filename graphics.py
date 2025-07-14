import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
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
    Comprehensive molecular visualization suite for drug discovery analysis.
    
    This class encapsulates all visualization functionality with a clean, modular design
    that eliminates code duplication and provides consistent, high-quality plots.
    """
    
    def __init__(self, context: PlotContext, config: Optional[PlotConfiguration] = None):
        """
        Initialize the visualization suite.
        
        Args:
            context: Plot context containing parameters and paths
            config: Optional custom configuration (uses default if None)
        """
        self.context = context
        self.config = config or PlotConfiguration()
        
        # Ensure image directory exists
        self.context.image_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_histogram_base(self, df: pd.DataFrame, score_col: str) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a base histogram with consistent styling and common elements.
        
        This method eliminates code duplication by providing a common foundation
        for all histogram plots.
        """
        try:
            # Get score configuration
            score_config = self.config.SCORE_CONFIGS.get(score_col, {})
            
            # Setup figure
            fig, ax = plt.subplots(figsize=self.config.FIGURE_SIZE)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Get data
            values = df[score_col].dropna().values
            if len(values) == 0:
                raise ValueError(f"No valid values found for {score_col}")
            
            # Handle dynamic binning for SMILES length
            if score_col == 'len_smiles':
                min_len, max_len = int(values.min()), int(values.max())
                bins = np.linspace(min_len - 0.5, max_len + 0.5, num=30)
                x_range = (min_len - 0.5, max_len + 0.5)
            else:
                bins = score_config['bins']
                x_range = score_config['x_range']
            
            # Create histogram
            color = self.config.COLORS.get(score_col.lower().replace('score', '_score'), 'gray')
            ax.hist(values, bins=bins, color=color, edgecolor='black', 
                   rwidth=0.8, alpha=0.6, label='Histogram')
            
            # Add KDE overlay with robust error handling
            self._add_kde_overlay(ax, values, f'dark{color}', x_range, bins)
            
            # Add mean line
            self._add_mean_line(ax, values)
            
            # Apply styling
            ax.set_xlabel(score_config.get('xlabel', score_col))
            ax.set_ylabel('Count of Molecules')
            ax.set_xlim(x_range)
            ax.set_title(score_config.get('title', f'Distribution of {score_col}'))
            ax.legend()
            
            return fig, ax
            
        except Exception as e:
            print(f"Error creating histogram for {score_col}: {e}")
            raise
    
    def _add_kde_overlay(self, ax: plt.Axes, values: np.ndarray, color: str, 
                        x_range: Tuple[float, float], bins: np.ndarray) -> None:
        """Add KDE overlay with robust error handling"""
        try:
            # Check for sufficient data diversity
            if len(values) < 3 or np.std(values) < 1e-10:
                return  # Skip KDE for insufficient or non-diverse data
                
            kde = gaussian_kde(values)
            x_vals = np.linspace(*x_range, 300)
            kde_vals = kde(x_vals) * len(values) * (bins[1] - bins[0])
            ax.plot(x_vals, kde_vals, color=color, lw=2, label='KDE')
            
        except Exception as e:
            # Silently skip KDE on any error (singular matrix, etc.)
            pass
    
    def _add_mean_line(self, ax: plt.Axes, values: np.ndarray) -> None:
        """Add mean line to plot"""
        try:
            mean_val = np.mean(values)
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean = {mean_val:.2f}')
        except Exception:
            pass  # Skip mean line on error
    
    def _add_molecule_inset(self, fig: plt.Figure, ax: plt.Axes, df: pd.DataFrame, 
                           score_col: str, minimize: bool = True) -> None:
        """Add molecule structure inset to plot with improved positioning"""
        try:
            if 'smiles' not in df.columns or len(df) == 0:
                return
                
            idx = df[score_col].idxmin() if minimize else df[score_col].idxmax()
            smiles = df.loc[idx, 'smiles']
            score = df.loc[idx, score_col]
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return
                
            img = Draw.MolToImage(mol, size=(1200, 1200))
            img = PIL.ImageOps.expand(img, border=8, fill='black')
            
            filename = df.loc[idx, 'filename'] if 'filename' in df.columns else 'N/A'
            text = f'File: {filename[:20]}...\n{score_col}: {score:.2f}'
            
            # Position inset relative to legend
            legend = ax.get_legend()
            if legend:
                fig.canvas.draw()
                legend_box = legend.get_window_extent(fig.canvas.get_renderer())
                legend_box = legend_box.transformed(fig.transFigure.inverted())
                
                inset_width, inset_height = 0.25, 0.5
                inset_x = max(0.02, legend_box.x1 - inset_width)
                inset_y = max(0.02, legend_box.y0 - inset_height - 0.02)
                
                inset_ax = fig.add_axes([inset_x, inset_y, inset_width, inset_height])
                inset_ax.axis('off')
                inset_ax.imshow(img)
                inset_ax.text(0.5, 0.02, text, ha='center', va='bottom', fontsize=8,
                             color='white', transform=inset_ax.transAxes,
                             bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))
                             
        except Exception as e:
            print(f"Warning: Could not add molecule inset: {e}")
    
    def plot_synthesizability_score(self, df: pd.DataFrame, score_type: ScoreType) -> None:
        """
        Plot synthesizability score distribution with unified interface.
        
        This method replaces separate plot_sa_score, plot_sc_score, plot_np_score functions
        with a single, configurable method.
        """
        try:
            score_col = score_type.value
            fig, ax = self._create_histogram_base(df, score_col)
            
            # Add molecule inset for score plots (not for SMILES length)
            if score_type != ScoreType.SMILES_LENGTH:
                score_config = self.config.SCORE_CONFIGS[score_col]
                self._add_molecule_inset(fig, ax, df, score_col, score_config['minimize'])
            
            plt.tight_layout()
            plt.show()
            
            # Save plot
            plot_type = score_col.lower().replace('_', '')
            self._save_plot(fig, plot_type)
            
        except Exception as e:
            print(f"Error plotting {score_type.value}: {e}")
            raise
        finally:
            plt.close(fig)
    
    def plot_smiles_length(self, df: pd.DataFrame) -> None:
        """Plot SMILES length distribution"""
        self.plot_synthesizability_score(df, ScoreType.SMILES_LENGTH)
    
    def plot_sa_score(self, df: pd.DataFrame) -> None:
        """Plot SA score distribution"""
        self.plot_synthesizability_score(df, ScoreType.SA_SCORE)
    
    def plot_sc_score(self, df: pd.DataFrame) -> None:
        """Plot SC score distribution"""
        self.plot_synthesizability_score(df, ScoreType.SC_SCORE)
    
    def plot_np_score(self, df: pd.DataFrame) -> None:
        """Plot NP score distribution"""
        self.plot_synthesizability_score(df, ScoreType.NP_SCORE)
    
    def _create_pie_chart_base(self, data: pd.Series, title: str, plot_type: str) -> None:
        """Create a base pie chart with consistent styling"""
        try:
            fig, ax = plt.subplots(figsize=self.config.PIE_FIGURE_SIZE)
            
            # Filter out zero values for cleaner pie charts
            data_filtered = data[data > 0]
            
            wedges, texts, autotexts = ax.pie(
                data_filtered, 
                labels=data_filtered.index, 
                autopct='%1.1f%%', 
                startangle=140,
                colors=plt.cm.Set3.colors[:len(data_filtered)]
            )
            
            # Improve text readability
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            plt.axis('equal')
            plt.tight_layout()
            plt.show()
            
            self._save_plot(fig, plot_type)
            
        except Exception as e:
            print(f"Error creating pie chart {plot_type}: {e}")
            raise
        finally:
            plt.close(fig)
    
    def plot_score_pie_chart(self, df: pd.DataFrame, score_type: ScoreType) -> None:
        """Create pie chart for score distribution with unified interface"""
        try:
            score_col = score_type.value
            pie_config = self.config.PIE_CHART_CONFIGS[score_col]
            
            # Create bins
            df_temp = df.copy()
            bin_col = f'{score_col}_bin'
            df_temp[bin_col] = pd.cut(
                df_temp[score_col], 
                bins=pie_config['bins'], 
                labels=pie_config['labels'],
                right=True, 
                include_lowest=True
            )
            
            counts = df_temp[bin_col].value_counts().reindex(pie_config['labels'])
            plot_type = f"{score_col.lower().replace('_', '')}_pie"
            
            self._create_pie_chart_base(counts, pie_config['title'], plot_type)
            
        except Exception as e:
            print(f"Error creating pie chart for {score_type.value}: {e}")
            raise
    
    def plot_lipinski_violations_pie(self, df: pd.DataFrame) -> None:
        """Plot Lipinski rule violations as pie chart"""
        try:
            violation_counts = df['failed'].fillna('').apply(
                lambda x: 0 if x.strip() == '' else len(x.split(';'))
            )
            violation_summary = violation_counts.value_counts().sort_index()
            
            # Create meaningful labels
            violation_summary.index = [
                f'{i} rule{"s" if i != 1 else ""} violated' 
                for i in violation_summary.index
            ]
            
            self._create_pie_chart_base(
                violation_summary, 
                'Lipinski\'s Rule Violations per Molecule', 
                'lipinski_violations'
            )
            
        except Exception as e:
            print(f"Error creating Lipinski violations pie chart: {e}")
            raise
    
    def _normalize_scores(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Normalize scores to 0-1 range for statistical comparison"""
        normalized = {}
        
        for score_col, config in self.config.SCORE_CONFIGS.items():
            if config['norm_range'] and score_col in df.columns:
                min_val, max_val = config['norm_range']
                normalized[score_col] = (df[score_col] - min_val) / (max_val - min_val)
        
        return normalized
    
    def plot_statistical_comparison(self, df: pd.DataFrame, plot_type: str = 'boxplot') -> None:
        """Create statistical comparison plots (boxplot or violin plot)"""
        try:
            normalized = self._normalize_scores(df)
            if not normalized:
                print("Warning: No valid scores for statistical comparison")
                return
            
            data = [normalized[key].dropna().values for key in normalized.keys()]
            labels = list(normalized.keys())
            
            fig, ax = plt.subplots(figsize=self.config.FIGURE_SIZE)
            
            if plot_type == 'boxplot':
                box = ax.boxplot(
                    data, 
                    patch_artist=True, 
                    labels=labels, 
                    showmeans=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2),
                    meanprops=dict(marker='o', markerfacecolor='blue', markeredgecolor='black')
                )
                title = 'Boxplot of Normalized Synthesizability Scores'
                
            elif plot_type == 'violin':
                parts = ax.violinplot(data, showmeans=True, showmedians=True)
                colors = ['orange', 'blue', 'pink'][:len(data)]
                
                for i, pc in enumerate(parts['bodies']):
                    pc.set_facecolor(colors[i])
                    pc.set_alpha(0.6)
                
                ax.set_xticks(range(1, len(labels) + 1))
                ax.set_xticklabels(labels)
                title = 'Violin Plot of Normalized Synthesizability Scores'
            
            ax.set_ylabel('Normalized Score Value (0-1)')
            ax.set_title(title)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.show()
            
            self._save_plot(fig, f'{plot_type}_synthesizability')
            
        except Exception as e:
            print(f"Error creating {plot_type}: {e}")
            raise
        finally:
            plt.close(fig)
    
    def plot_correlation_analysis(self, df: pd.DataFrame) -> None:
        """Create pairplot for correlation analysis"""
        try:
            cols = ['SA_score', 'SCScore', 'NP_score', 'len_smiles']
            available_cols = [col for col in cols if col in df.columns]
            
            if len(available_cols) < 2:
                print("Warning: Insufficient columns for correlation analysis")
                return
            
            data = df[available_cols].dropna()
            
            if len(data) < 5:  # Need at least 5 points for meaningful correlation
                print(f"Warning: Insufficient data points for correlation analysis (need ≥5, got {len(data)})")
                return
            
            sns.set_style('ticks')
            g = sns.pairplot(data, diag_kind='kde', plot_kws={'alpha': 0.6, 's': 30})
            g.fig.suptitle('Molecular Properties Correlation Analysis', y=1.02, fontsize=16)
            
            plt.tight_layout()
            plt.show()
            
            # Save the figure
            g.savefig(
                self.context.get_filepath('correlation_pairplot'),
                dpi=self.config.DPI,
                bbox_inches='tight'
            )
            
        except Exception as e:
            print(f"Error creating correlation analysis: {e}")
            raise
        finally:
            plt.close('all')
    
    def plot_chemical_space_tsne(self, fps: List, synth_df: pd.DataFrame) -> None:
        """Create t-SNE plot of chemical space with robust parameter selection"""
        try:
            n_samples = len(fps)
            if n_samples < 5:  # Need at least 5 samples for meaningful t-SNE
                print(f"Warning: Need at least 5 samples for t-SNE (got {n_samples})")
                return
            
            # Dynamic perplexity with better bounds - must be less than n_samples
            perplexity = min(30, max(2, min(n_samples // 3, n_samples - 1)))
            
            # Ensure perplexity is valid
            if perplexity >= n_samples:
                perplexity = max(1, n_samples - 1)
            
            print(f"Running t-SNE with {n_samples} samples, perplexity={perplexity}")
            
            tsne = TSNE(
                n_components=2, 
                random_state=42, 
                perplexity=perplexity,
                n_iter=300,  # Reduced iterations for small datasets
                init='pca'
            )
            
            tsne_results = tsne.fit_transform(np.stack(fps))
            
            fig, ax = plt.subplots(figsize=self.config.FIGURE_SIZE)
            
            # Use SA_score for coloring if available
            if 'SA_score' in synth_df.columns:
                colors = synth_df['SA_score'].values
                scatter = ax.scatter(
                    tsne_results[:, 0], tsne_results[:, 1], 
                    c=colors, cmap='viridis', alpha=0.7, s=50, edgecolors='black', linewidth=0.5
                )
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('SA_score', fontsize=12)
            else:
                ax.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.7, s=50)
            
            ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
            ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
            ax.set_title('Chemical Space Visualization (t-SNE)', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            self._save_plot(fig, 'chemical_space_tsne')
            
        except Exception as e:
            print(f"Error creating t-SNE plot: {e}")
            # Don't re-raise the exception to avoid breaking the entire pipeline
        finally:
            plt.close('all')
    
    def plot_tanimoto_similarity_heatmap(self, similarity_csv_path: str, 
                                        similarity_threshold: float = 0.5) -> None:
        """
        Create a lower triangular Tanimoto similarity heatmap from pre-calculated similarities.
        
        Args:
            similarity_csv_path: Path to CSV file with columns: mol_1, smi_1, mol_2, smi_2, tanimoto
            similarity_threshold: Threshold for displaying labels (default: 0.5)
        """
        try:
            # Load similarity data from CSV
            print(f"Loading similarity data from: {similarity_csv_path}")
            similarity_df = pd.read_csv(similarity_csv_path)
            
            # Validate required columns
            required_cols = ['mol_1', 'smi_1', 'mol_2', 'smi_2', 'tanimoto']
            missing_cols = [col for col in required_cols if col not in similarity_df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in CSV: {missing_cols}")
            
            # Get unique molecule filenames
            all_molecules = set(similarity_df['mol_1'].tolist() + similarity_df['mol_2'].tolist())
            filenames = sorted(list(all_molecules))
            n_mols = len(filenames)
            
            if n_mols < 2:
                print("Warning: Need at least 2 molecules for similarity analysis")
                return
            
            if n_mols > 100:
                print(f"Warning: Large dataset ({n_mols} molecules). Heatmap may be difficult to read.")
            
            print(f"Creating similarity matrix for {n_mols} molecules from {len(similarity_df)} similarity pairs...")
            
            # Create filename to index mapping
            filename_to_idx = {filename: idx for idx, filename in enumerate(filenames)}
            
            # Initialize similarity matrix
            similarity_matrix = np.zeros((n_mols, n_mols))
            
            # Fill similarity matrix from CSV data
            for _, row in similarity_df.iterrows():
                mol1, mol2, similarity = row['mol_1'], row['mol_2'], row['tanimoto']
                
                if mol1 in filename_to_idx and mol2 in filename_to_idx:
                    i, j = filename_to_idx[mol1], filename_to_idx[mol2]
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity  # Make symmetric
            
            # Set diagonal to 1.0 (self-similarity)
            for i in range(n_mols):
                similarity_matrix[i, i] = 1.0
            
            # Create lower triangular mask
            mask = np.tril(np.ones_like(similarity_matrix, dtype=bool))
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(max(8, n_mols * 0.3), max(8, n_mols * 0.3)))
            
            # Apply mask and create heatmap
            masked_matrix = np.where(mask, similarity_matrix, np.nan)
            im = ax.imshow(masked_matrix, cmap='viridis', vmin=0, vmax=1, aspect='equal')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Tanimoto Similarity', fontsize=12)
            
            # Set title and labels
            ax.set_title('Lower Triangular Tanimoto Similarity Heatmap', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Molecule Index', fontsize=12)
            ax.set_ylabel('Molecule Index', fontsize=12)
            
            # Find high-similarity pairs for selective labeling
            high_similarity_pairs = np.argwhere(
                (similarity_matrix >= similarity_threshold) & 
                mask & 
                (np.arange(similarity_matrix.shape[0])[:, None] > np.arange(similarity_matrix.shape[1]))
            )
            
            # Prepare tick labels
            n_labels = len(filenames)
            xtick_labels = ['' for _ in range(n_labels)]
            ytick_labels = ['' for _ in range(n_labels)]
            
            # Set labels only for high-similarity pairs
            for i, j in high_similarity_pairs:
                # Clean filename for display (remove extension and truncate if too long)
                clean_filename_i = self._clean_filename_for_display(filenames[i])
                clean_filename_j = self._clean_filename_for_display(filenames[j])
                
                ytick_labels[i] = clean_filename_i
                xtick_labels[j] = clean_filename_j
            
            # Set ticks and labels
            ax.set_xticks(range(n_labels))
            ax.set_yticks(range(n_labels))
            ax.set_xticklabels(xtick_labels, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(ytick_labels, rotation=0, ha='right', fontsize=8)
            
            # Add statistics annotation
            stats_text = self._calculate_similarity_stats(similarity_matrix, mask)
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.show()
            
            self._save_plot(fig, 'tanimoto_similarity_heatmap')
            
            # Also create a summary plot of similarity distribution
            self._plot_similarity_distribution(similarity_matrix, mask)
            
        except Exception as e:
            print(f"Error creating Tanimoto similarity heatmap: {e}")
            raise
        finally:
            plt.close('all')
    
    def _clean_filename_for_display(self, filename: str, max_length: int = 20) -> str:
        """Clean and truncate filename for display in plots"""
        # Remove file extension
        name = filename.split('.')[0] if '.' in filename else filename
        
        # Truncate if too long
        if len(name) > max_length:
            name = name[:max_length-3] + '...'
        
        return name
    
    def _calculate_similarity_stats(self, similarity_matrix: np.ndarray, mask: np.ndarray) -> str:
        """Calculate and format similarity statistics"""
        # Get lower triangular values (excluding diagonal)
        lower_tri_mask = mask & (similarity_matrix != 1.0)
        similarities = similarity_matrix[lower_tri_mask]
        
        if len(similarities) == 0:
            return "No similarity data"
        
        mean_sim = np.mean(similarities)
        median_sim = np.median(similarities)
        std_sim = np.std(similarities)
        max_sim = np.max(similarities)
        min_sim = np.min(similarities)
        
        # Count high similarity pairs
        high_sim_count = np.sum(similarities >= 0.7)
        total_pairs = len(similarities)
        
        stats_text = (
            f"Similarity Statistics:\n"
            f"Mean: {mean_sim:.3f}\n"
            f"Median: {median_sim:.3f}\n"
            f"Std: {std_sim:.3f}\n"
            f"Range: {min_sim:.3f} - {max_sim:.3f}\n"
            f"High similarity (≥0.7): {high_sim_count}/{total_pairs}"
        )
        
        return stats_text
    
    def _plot_similarity_distribution(self, similarity_matrix: np.ndarray, mask: np.ndarray) -> None:
        """Create a distribution plot of Tanimoto similarities"""
        try:
            # Get lower triangular values (excluding diagonal)
            lower_tri_mask = mask & (similarity_matrix != 1.0)
            similarities = similarity_matrix[lower_tri_mask]
            
            if len(similarities) == 0:
                print("Warning: No similarity data for distribution plot")
                return
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Histogram with KDE
            ax1.hist(similarities, bins=30, density=True, alpha=0.7, color='skyblue', 
                    edgecolor='black', label='Histogram')
            
            # Add KDE if possible
            try:
                if len(similarities) > 2 and np.std(similarities) > 1e-10:
                    kde = gaussian_kde(similarities)
                    x_vals = np.linspace(0, 1, 200)
                    kde_vals = kde(x_vals)
                    ax1.plot(x_vals, kde_vals, color='red', lw=2, label='KDE')
            except Exception:
                pass
            
            ax1.axvline(np.mean(similarities), color='orange', linestyle='--', 
                       linewidth=2, label=f'Mean = {np.mean(similarities):.3f}')
            ax1.set_xlabel('Tanimoto Similarity')
            ax1.set_ylabel('Density')
            ax1.set_title('Distribution of Tanimoto Similarities')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Box plot
            box_data = [similarities]
            bp = ax2.boxplot(box_data, patch_artist=True, labels=['All Pairs'])
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][0].set_alpha(0.7)
            
            ax2.set_ylabel('Tanimoto Similarity')
            ax2.set_title('Box Plot of Tanimoto Similarities')
            ax2.grid(True, alpha=0.3)
            
            # Add summary statistics
            stats_text = (
                f"n = {len(similarities)}\n"
                f"μ = {np.mean(similarities):.3f}\n"
                f"σ = {np.std(similarities):.3f}\n"
                f"Q1 = {np.percentile(similarities, 25):.3f}\n"
                f"Q3 = {np.percentile(similarities, 75):.3f}"
            )
            
            ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.show()
            
            self._save_plot(fig, 'tanimoto_similarity_distribution')
            
        except Exception as e:
            print(f"Error creating similarity distribution plot: {e}")
        finally:
            plt.close(fig)
    
    def _save_plot(self, fig: plt.Figure, plot_type: str) -> None:
        """Save plot with consistent settings"""
        try:
            filepath = self.context.get_filepath(plot_type)
            fig.savefig(filepath, dpi=self.config.DPI, bbox_inches='tight', facecolor='white')
            print(f"Saved: {filepath}")
        except Exception as e:
            print(f"Warning: Could not save plot {plot_type}: {e}")
    
    def generate_comprehensive_report(self, synth_df: pd.DataFrame, 
                                    lipinski_df: pd.DataFrame, fps: List, 
                                    similarity_csv_path: Optional[str] = None) -> None:
        """
        Generate a comprehensive visualization report.
        
        This is the main method that orchestrates all visualizations in a logical order.
        
        Args:
            synth_df: DataFrame with synthesizability scores
            lipinski_df: DataFrame with Lipinski analysis results
            fps: List of molecular fingerprints
            similarity_csv_path: Optional path to CSV file with pre-calculated similarities
        """
        print("Starting Molecular Visualization Suite")
        print(f"Dataset: {len(synth_df)} molecules")
        print(f"Fingerprints: {len(fps)} molecules")
        print(f"Lipinski data: {len(lipinski_df)} molecules")
        print(f"Output directory: {self.context.image_dir}")
        
        # Check for data consistency
        if len(synth_df) != len(fps):
            print(f"Warning: Mismatch between synthesizability data ({len(synth_df)}) and fingerprints ({len(fps)})")
        
        if len(synth_df) < 3:
            print(f"Note: Small dataset ({len(synth_df)} molecules) - some advanced plots may be skipped")
        
        # Track successful plots
        successful_plots = []
        failed_plots = []
        
        try:
            # 1. Distribution Analysis
            print("\nGenerating distribution plots...")
            for plot_func, plot_name in [
                (lambda: self.plot_smiles_length(synth_df), "SMILES length"),
                (lambda: self.plot_sa_score(synth_df), "SA score"),
                (lambda: self.plot_sc_score(synth_df), "SC score"),
                (lambda: self.plot_np_score(synth_df), "NP score")
            ]:
                try:
                    plot_func()
                    successful_plots.append(plot_name)
                except Exception as e:
                    print(f"Warning: Failed to generate {plot_name} plot: {e}")
                    failed_plots.append(plot_name)
            
            # 2. Categorical Analysis
            print("\nGenerating pie charts...")
            for plot_func, plot_name in [
                (lambda: self.plot_score_pie_chart(synth_df, ScoreType.SMILES_LENGTH), "SMILES length pie"),
                (lambda: self.plot_score_pie_chart(synth_df, ScoreType.SA_SCORE), "SA score pie"),
                (lambda: self.plot_score_pie_chart(synth_df, ScoreType.SC_SCORE), "SC score pie"),
                (lambda: self.plot_score_pie_chart(synth_df, ScoreType.NP_SCORE), "NP score pie"),
                (lambda: self.plot_lipinski_violations_pie(lipinski_df), "Lipinski violations pie")
            ]:
                try:
                    plot_func()
                    successful_plots.append(plot_name)
                except Exception as e:
                    print(f"Warning: Failed to generate {plot_name}: {e}")
                    failed_plots.append(plot_name)
            
            # 3. Statistical Comparison
            print("\nGenerating statistical comparisons...")
            for plot_func, plot_name in [
                (lambda: self.plot_statistical_comparison(synth_df, 'boxplot'), "boxplot"),
                (lambda: self.plot_statistical_comparison(synth_df, 'violin'), "violin plot")
            ]:
                try:
                    plot_func()
                    successful_plots.append(plot_name)
                except Exception as e:
                    print(f"Warning: Failed to generate {plot_name}: {e}")
                    failed_plots.append(plot_name)
            
            # 4. Correlation Analysis
            print("\nGenerating correlation analysis...")
            try:
                self.plot_correlation_analysis(synth_df)
                successful_plots.append("correlation analysis")
            except Exception as e:
                print(f"Warning: Failed to generate correlation analysis: {e}")
                failed_plots.append("correlation analysis")
            
            # 5. Chemical Space Visualization
            print("\nGenerating chemical space visualization...")
            try:
                self.plot_chemical_space_tsne(fps, synth_df)
                successful_plots.append("t-SNE plot")
            except Exception as e:
                print(f"Warning: Failed to generate t-SNE plot: {e}")
                failed_plots.append("t-SNE plot")
            
            # 6. Molecular Similarity Analysis
            if similarity_csv_path and Path(similarity_csv_path).exists():
                print("\nGenerating molecular similarity analysis...")
                try:
                    self.plot_tanimoto_similarity_heatmap(similarity_csv_path)
                    successful_plots.append("similarity heatmap")
                except Exception as e:
                    print(f"Warning: Failed to generate similarity analysis: {e}")
                    failed_plots.append("similarity heatmap")
            else:
                print("\nSkipping similarity analysis (no CSV file provided or file not found)")
                if similarity_csv_path:
                    print(f"Expected CSV file: {similarity_csv_path}")
            
            # Summary
            print(f"\nVisualization suite completed!")
            print(f"Successful plots ({len(successful_plots)}): {', '.join(successful_plots)}")
            if failed_plots:
                print(f"Failed plots ({len(failed_plots)}): {', '.join(failed_plots)}")
            print(f"All plots saved to: {self.context.image_dir}")
            
        except Exception as e:
            print(f"\nCritical error during visualization generation: {e}")
            # Don't re-raise to allow the pipeline to continue


def load_molecular_data(paths: PipelinePaths, epoch: int, num_gen: int, 
                       known_binding_site: str, aurora: str, pdbid: str) -> Tuple:
    """
    Load molecular data and analysis results with comprehensive error handling.
    
    Returns:
        Tuple containing mols, smiles, filenames, fps, synth_df, lipinski_df
    """
    try:
        print(f"Loading molecular data for epoch {epoch}...")
        
        if epoch != 0:
            # Load generated molecules
            sdf_folder = paths.graphbp_sdf_path(epoch, num_gen, known_binding_site, pdbid)
            print(f"Loading from SDF folder: {sdf_folder}")
            mols, smiles, filenames, fps = load_mols_from_sdf_folder(sdf_folder)
        else:
            # Load Aurora kinase inhibitors
            known_inhib_file = paths.aurora_data_path(aurora)
            print(f"Loading Aurora kinase data: {known_inhib_file}")
            mols, smiles, filenames, fps = read_aurora_kinase_interactions(known_inhib_file)
        
        print(f"Loaded {len(mols)} molecules")
        
        # Load analysis results with graceful handling
        synth_file = f'synthesizability_scores.csv'
        synth_path = paths.hope_box_results_path(epoch, num_gen, known_binding_site, pdbid, synth_file)
        print(f"Loading synthesizability data: {synth_path}")
        
        try:
            synth_df = pd.read_csv(synth_path)
            print(f"Synthesizability data loaded: {len(synth_df)} records")
        except FileNotFoundError:
            print(f"Warning: Synthesizability file not found: {synth_path}")
            # Create empty dataframe with expected columns
            synth_df = pd.DataFrame(columns=['filename', 'smiles', 'SA_score', 'SCScore', 'NP_score', 'len_smiles'])
        
        lipinski_file = f'lipinski_pass.csv'
        lipinski_path = paths.hope_box_results_path(epoch, num_gen, known_binding_site, pdbid, lipinski_file)
        print(f"Loading Lipinski data: {lipinski_path}")
        
        try:
            lipinski_df = pd.read_csv(lipinski_path)
            print(f"Lipinski data loaded: {len(lipinski_df)} records")
        except FileNotFoundError:
            print(f"Warning: Lipinski file not found: {lipinski_path}")
            # Create empty dataframe with expected columns
            lipinski_df = pd.DataFrame(columns=['filename', 'smiles', 'failed'])
        
        return mols, smiles, filenames, fps, synth_df, lipinski_df
        
    except FileNotFoundError as e:
        print(f"Required data file not found: {e}")
        print("Please ensure all analysis results are available before running graphics generation.")
        raise
    except Exception as e:
        print(f"Error loading molecular data: {e}")
        raise


def main():
    """
    Main function to run the molecular visualization suite.
    
    This function provides a clean interface for generating comprehensive
    molecular analysis visualizations with proper error handling and logging.
    """
    parser = argparse.ArgumentParser(
        description='Molecular Visualization Suite for Drug Discovery Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python graphics.py --epoch 99 --num_gen 100 --aurora A
  python graphics.py --epoch 0  # Analyze Aurora kinase inhibitors
  python graphics.py --known_binding_site True --pdbid 4af3
  python graphics.py --epoch 99 --similarity-csv similarity_data.csv  # Include similarity analysis
        """
    )
    
    parser.add_argument('--num_gen', type=int, default=0, 
                       help='Number of generated molecules (default: 0)')
    parser.add_argument('--epoch', type=int, default=0, 
                       help='Epoch number for model generation (0-99, default: 0)')
    parser.add_argument('--known_binding_site', type=str, default='0', 
                       help='Use binding site information (True/False, default: 0)')
    parser.add_argument('--aurora', type=str, default='B', choices=['A', 'B'],
                       help='Aurora kinase type (default: B)')
    parser.add_argument('--pdbid', type=str, default='4af3', 
                       help='PDB ID for the target (default: 4af3)')
    parser.add_argument('--similarity-csv', type=str, 
                       help='Path to CSV file with pre-calculated Tanimoto similarities')
    parser.add_argument('--config', type=str, help='Path to custom configuration file (optional)')
    parser.add_argument('--output-dir', type=str, help='Custom output directory (optional)')
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline paths
        print("🔧 Initializing pipeline configuration...")
        paths = PipelinePaths()
        
        # Setup output directory
        if args.output_dir:
            image_dir = Path(args.output_dir) / 'images'
        else:
            results_dir = Path(paths.hope_box_results_path(
                args.epoch, args.num_gen, args.known_binding_site, 
                args.pdbid.lower(), 'dummy.csv'
            )).parent
            image_dir = results_dir / 'images'
        
        # Create plot context
        context = PlotContext(
            epoch=args.epoch,
            num_gen=args.num_gen,
            known_binding_site=args.known_binding_site,
            aurora=args.aurora,
            pdbid=args.pdbid.lower(),
            image_dir=image_dir
        )
        
        # Load custom configuration if provided
        config = PlotConfiguration()
        if args.config:
            print(f"Loading custom configuration from {args.config}")
            # Here you could add custom config loading logic
        
        # Load molecular data
        print("Loading molecular data and analysis results...")
        mols, smiles, filenames, fps, synth_df, lipinski_df = load_molecular_data(
            paths, args.epoch, args.num_gen, args.known_binding_site, 
            args.aurora, args.pdbid.lower()
        )
        
        # Validate data
        if len(mols) == 0:
            print("Warning: No molecules found in the dataset")
            print("Creating empty output directory and exiting gracefully...")
            context.image_dir.mkdir(parents=True, exist_ok=True)
            return 0
        if len(synth_df) == 0:
            print("Warning: No synthesizability data found")
        if len(lipinski_df) == 0:
            print("Warning: No Lipinski data found")
        
        # Create visualization suite
        print("🎨 Initializing Molecular Visualization Suite...")
        viz_suite = MolecularVisualizationSuite(context, config)
        
        # Generate comprehensive report
        viz_suite.generate_comprehensive_report(synth_df, lipinski_df, fps, args.similarity_csv)
        
        print("\nMolecular visualization suite completed!")
        print(f"Processed {len(mols)} molecules")
        print(f"All plots saved to: {context.image_dir}")
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        return 1
    except FileNotFoundError as e:
        print(f"\nFile not found: {e}")
        print("Tip: Ensure all analysis results are generated before running visualization")
        # Don't return 1, let the pipeline continue with warning
        print("Continuing with available data...")
    except ValueError as e:
        print(f"\nData validation error: {e}")
        # Don't return 1, let the pipeline continue with warning
        print("Continuing with available data...")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("Continuing with available data...")
    
    return 0


if __name__ == '__main__':
    exit(main())