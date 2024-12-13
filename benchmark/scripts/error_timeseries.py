import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Function to read data from a text file and plot it
def plot_time_series_from_file(file_path):
    # Read the data from the text file
    with open(file_path, 'r') as file:
        data_lines = file.readlines()

    # Extract the header (column names) and data values
    columns = data_lines[0].strip().split()
    data_values = [list(map(float, line.split())) for line in data_lines[1:]]

    # Create a DataFrame
    df = pd.DataFrame(data_values, columns=columns)

    # Plot each column as a time series
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    # plt.tight_layout()
    time_steps = np.arange(len(df))

    for column in df.columns:
        print(column)
        if column[0] == 'B':
            ax[0].plot(time_steps, df[column], label=column)
        elif column[0] == 'R':
            ax[1].plot(time_steps, df[column], label=column)

    ax[0].set_title("Time Series Plot of Columns")
    ax[1].set_xlabel("Time Step")
    ax[0].set_ylabel("BPE (m)")
    ax[1].set_ylabel("RPE (px)")
    ax[0].legend()
    ax[0].grid()
    ax[1].legend()
    ax[1].grid()
    plt.show()

# Example usage:
plot_time_series_from_file('C:\\Users\\mwmasters\\Downloads\\sim3_report.txt')
