import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma

def read_data(filepath):
    # Read the data assuming whitespace delimiter and no header
    data = pd.read_csv(filepath, delim_whitespace=True, header=None,
                       names=['BPE_x', 'BPE_y', 'BPE_z', 'RPE_x', 'RPE_y'])
    # Convert all columns to numeric, coercing errors which will convert non-numeric to NaN
    data = data.apply(pd.to_numeric, errors='coerce')
    return data

def fit_and_plot_histograms(data, columns, dists, fig_title, x_label):
    fig, axs = plt.subplots(1, len(columns), figsize=(5 * len(columns), 5))
    for i, column in enumerate(columns):
        ax = axs[i]
        values = data[column].dropna()  # Drop NaN values which might arise from conversion
        if dists[i] == 'norm':
            mu, std = norm.fit(values)
            # Plot the histogram
            ax.hist(values, bins=100, density=True, alpha=0.6, color='r')
            # Plot the PDF.
            xmin, xmax = ax.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            ax.plot(x, p, 'k', linewidth=2)
            title = f'{column}: μ={mu:.2f}, σ={std:.2f}'
        elif dists[i] == 'gamma':
            shape, loc, scale = gamma.fit(values)
            mean = gamma.mean(shape, loc, scale)
            variance = gamma.var(shape, loc, scale)
            std = np.sqrt(variance)
            # Plot the histogram
            ax.hist(values, bins=100, density=True, alpha=0.6, color='r')
            # Plot the PDF
            xmin, xmax = ax.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = gamma.pdf(x, shape, loc, scale)
            ax.plot(x, p, 'k', linewidth=2)
            title = (f'{column}: α={shape:.2f}, β={scale:.2f}, '
                     f'μ={mean:.2f}, σ={std:.2f}')
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel('Density')
    plt.tight_layout()
    plt.suptitle(fig_title)
    plt.subplots_adjust(top=0.85)
    plt.show()

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <filename>")
        sys.exit(1)
    filepath = sys.argv[1]
    
    data = read_data(filepath)
    
    # Calculate L2 error for BPE and RPE terms
    data['L2_error_BPE'] = np.sqrt(data['BPE_x']**2 + data['BPE_y']**2 + data['BPE_z']**2)
    data['L2_error_RPE'] = np.sqrt(data['RPE_x']**2 + data['RPE_y']**2)
    
    # Fit and plot histograms including the L2 error
    fit_and_plot_histograms(
        data, 
        ['BPE_x', 'BPE_y', 'BPE_z', 'L2_error_BPE'], 
        ['norm', 'norm', 'norm', 'gamma'], 
        fig_title='Backprojection Error', x_label='meters'
    )

    fit_and_plot_histograms(
        data,
        ['RPE_x', 'RPE_y', 'L2_error_RPE'],
        ['norm', 'norm', 'gamma'],
        fig_title='Reprojection Error', x_label='pixels'
    )

if __name__ == '__main__':
    main()
