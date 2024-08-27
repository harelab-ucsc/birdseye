import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
import argparse

# Function to read data from a text file
def read_errors_from_file(filename):
    try:
        with open(filename, 'r') as file:
            errors = np.array([float(line.strip()) for line in file])
        return errors
    except Exception as e:
        print(f"Error reading file: {e}")
        exit(1)

# Parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Backprojection L2 error terms with Gamma fit.')
    parser.add_argument('filename', type=str, help='Path to the text file containing L2 error terms.')
    parser.add_argument('--bins', type=int, default=20, help='Number of bins for the histogram (default: 20).')
    return parser.parse_args()

# Main function
def main():
    # Parse arguments
    args = parse_arguments()
    
    # Read errors from the file
    errors = read_errors_from_file(args.filename)
    
    # Compute histogram
    hist, bin_edges = np.histogram(errors, bins=args.bins, density=True)
    
    # Compute bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Fit Gamma distribution
    shape, loc, scale = gamma.fit(errors, floc=0)  # Fix location parameter to 0 for Gamma distribution
    
    # Generate points for the Gamma fit
    x = np.linspace(min(errors), max(errors), 100)
    pdf = gamma.pdf(x, shape, loc, scale)
    
    # Compute mean and standard deviation
    mean = np.mean(errors)
    std_dev = np.std(errors)
    
    # Plot histogram and Gamma fit
    plt.figure(figsize=(10, 6))
    
    # Plot histogram
    plt.hist(errors, bins=args.bins, density=True, alpha=0.6, color='g', label='Histogram')
    
    # Plot Gamma fit
    plt.plot(x, pdf, 'k', linewidth=2, label='Gamma fit')
    
    # Add labels and title
    plt.xlabel('Error [meters]')
    plt.ylabel('Density')
    plt.title('Backprojection L2 error terms with Gamma fit')
    
    # Display Gamma parameters in the plot
    plt.legend()
    plt.grid(True)
    plt.annotate(f'α : {shape:.2f}\nβ: {1/scale:.2f}\nμ: {mean:.2f} m\nσ: {std_dev:.2f} m',
                 xy=(0.7, 0.8), xycoords='axes fraction',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                 fontsize=12)
    
    # Show plot
    plt.show()

if __name__ == "__main__":
    main()
