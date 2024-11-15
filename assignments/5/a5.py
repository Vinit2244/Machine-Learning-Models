import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
import librosa
import librosa.display
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from models.kde.kde import KDE
from models.gmm.gmm import GMM

# Colors for printing for better readability
BLUE = "\033[34m"
GREEN = "\033[32m"
RED = "\033[31m"
MAGENTA = "\033[35m"
RESET = "\033[0m"

def generate_circle_points(num_points, radius, center=(0, 0), noise=0.1):
    angles = 2 * np.pi * np.random.rand(num_points)
    radii = radius * np.sqrt(np.random.rand(num_points))
    x = center[0] + radii * np.cos(angles) + noise * np.random.randn(num_points)
    y = center[1] + radii * np.sin(angles) + noise * np.random.randn(num_points)
    return x, y

def save_all_mfccs_to_csvs(folder_path, save_csv_path):
    people = ['george', 'jackson', 'lucas', 'nicolas', 'theo', 'yweweler']
    digits = range(10)
    
    if not os.path.exists(save_csv_path):
        os.makedirs(save_csv_path)

    for person in people:
        for digit in digits:
            for file_index in range(50):  # For 50 audio clips per digit
                filename = f"{digit}_{person}_{file_index}.wav"
                file_path = os.path.join(folder_path, filename)
                
                if os.path.isfile(file_path):
                    # Load the audio file
                    y, sr = librosa.load(file_path, sr=None)
                    # Extract MFCC features
                    n_fft = min(2048, len(y))
                    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft)
                    
                    # Save the MFCCs to a CSV file
                    mfcc_csv_filename = f"{digit}_{person}_{file_index}_mfcc.csv"
                    mfcc_csv_path = os.path.join(save_csv_path, mfcc_csv_filename)
                    pd.DataFrame(mfccs).to_csv(mfcc_csv_path, index=False, header=False)
                else:
                    print(f"File not found: {filename}")

def generate_mfcc_plots_from_csvs(save_csv_path, save_plot_path):
    people = ['george', 'jackson', 'lucas', 'nicolas', 'theo', 'yweweler']
    digits = range(10)

    if not os.path.exists(save_plot_path):
        os.makedirs(save_plot_path)

    for person in people:
        for digit in digits:
            # Create a figure for the grid of heatmaps (5 rows, 10 columns)
            fig, axes = plt.subplots(5, 10, figsize=(30, 15))
            fig.suptitle(f'MFCC Heatmaps for {person} - Digit {digit}', fontsize=24)

            for i in range(5):  # Iterate through 5 rows
                for j in range(10):  # Iterate through 10 columns
                    file_index = i * 10 + j
                    mfcc_csv_filename = f"{digit}_{person}_{file_index}_mfcc.csv"
                    mfcc_csv_path = os.path.join(save_csv_path, mfcc_csv_filename)

                    # Check if the CSV file exists
                    if os.path.isfile(mfcc_csv_path):
                        # Read MFCCs from CSV
                        mfccs = pd.read_csv(mfcc_csv_path, header=None).values

                        # Plot the MFCC heatmap on the grid
                        ax = axes[i, j]
                        librosa.display.specshow(mfccs, x_axis='time', ax=ax, cmap='viridis')
                        ax.set_title(f'Clip {file_index}')
                        ax.axis('off')  # Hide axis for better visibility
                    else:
                        # If the file doesn't exist, display a placeholder message
                        ax = axes[i, j]
                        ax.text(0.5, 0.5, 'File not found', fontsize=10, ha='center', va='center')
                        ax.axis('off')

            # Adjust layout and save the plot
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust the top space for the title
            save_file_path = os.path.join(save_plot_path, f'{person}_digit_{digit}_mfcc_heatmap.png')
            plt.savefig(save_file_path, dpi=300)
            plt.close(fig)

if __name__ == "__main__":
    # # =============================================================================
    # #                            Generate synthetic data
    # # =============================================================================
    # x_large, y_large = generate_circle_points(num_points=3000, radius=2, noise=0.2)
    # x_small, y_small = generate_circle_points(num_points=500, radius=0.25, center=(1, 1), noise=0.05)
    # x = np.concatenate([x_large, x_small])
    # y = np.concatenate([y_large, y_small])
    # data = np.column_stack([x, y])
    # print(f"{GREEN}Synthetic Data Generated{RESET}")
    # print(f"\n{BLUE}------------------------------------------------------------------------------------------------------{RESET}\n")

    # # =============================================================================
    # #                            Visualize synthetic data
    # # =============================================================================
    # plt.figure(figsize=(8, 8))
    # plt.scatter(x, y, color='black', s=1, alpha=0.6)
    # plt.title("Synthetic Data")
    # plt.xlim(-4, 4)
    # plt.ylim(-4, 4)
    # plt.grid(True)
    # plt.savefig("./assignments/5/figures/synthetic_data.png")
    # plt.close()
    # print(f"{GREEN}Synthetic Data Visualized and Saved{RESET}")
    # print(f"\n{BLUE}------------------------------------------------------------------------------------------------------{RESET}\n")

    # # =============================================================================
    # #                           Apply KDE on synthetic data
    # # =============================================================================
    # kde = KDE(bandwidth=0.5, kernel='gaussian')
    # kde.fit(data)
    # start_time = time.time()
    # kde.visualize(plot_type='3d', save_as="./assignments/5/figures/kde_vs_gmm/kde_3d.gif")
    # end_time = time.time()
    # print(f"{GREEN}KDE 3D plot saved in time {round(end_time - start_time, 3)}s{RESET}")
    # start_time = time.time()
    # kde.visualize(plot_type='contour', save_as="./assignments/5/figures/kde_vs_gmm/kde_contour.png")
    # end_time = time.time()
    # print(f"{GREEN}KDE contour plot saved in time {round(end_time - start_time, 3)}s{RESET}")
    # print(f"\n{BLUE}------------------------------------------------------------------------------------------------------{RESET}\n")

    # # =============================================================================
    # #                            Apply GMM on synthetic data
    # # =============================================================================
    # for n_components in range(2, 6):
    #     print(f"{MAGENTA}Fitting GMM with n_components = {n_components}{RESET}")
    #     gmm_model = GMM(n_components)
    #     gmm_model.load_data(data)
    #     start_time = time.time()
    #     epochs_taken = gmm_model.fit()
    #     end_time = time.time()
    #     print(f"{GREEN}GMM fitted, epochs taken = {epochs_taken}{RESET}")
    #     print(f"\t{GREEN}Time taken to fit: {round(end_time - start_time, 5)} s{RESET}")
    #     overall_log_likelihood = round(gmm_model.get_log_likelihood(), 5)
    #     print(f"\t{GREEN}Final Log Likelihood: {overall_log_likelihood}{RESET}")
    #     gmm_model.visualise(save_as=f"./assignments/5/figures/kde_vs_gmm/gmm_3d_{n_components}.png", plot_type='3d')
    #     print(f"{GREEN}GMM 3D Plot saved{RESET}")
    #     gmm_model.visualise(save_as=f"./assignments/5/figures/kde_vs_gmm/gmm_contour_{n_components}.png", plot_type='contour')
    #     print(f"{GREEN}GMM Contour Plot saved\n{RESET}")

    # print(f"\n{BLUE}------------------------------------------------------------------------------------------------------{RESET}\n")

    # # =============================================================================
    # #                    Save MFCCs to CSVs and generate heatmaps
    # # =============================================================================
    # folder_path = './data/external/free-spoken-digit-dataset-fsdd/recordings'
    # save_plot_path = './assignments/5/figures/mfcc_heatmaps'
    # save_csv_path = './data/interim/5/mfcc'

    # # Check if already exist
    # expected_csv_count = 3000  # 10 digits * 6 people * 50 recordings
    # csv_files = [file for file in os.listdir(save_csv_path) if file.endswith('.csv')]

    # if len(csv_files) == expected_csv_count:
    #     print(f"{GREEN}All MFCC CSV files already exist. Skipping CSV generation.{RESET}")
    # else:
    #     start_time = time.time()
    #     save_all_mfccs_to_csvs(folder_path, save_csv_path)
    #     end_time = time.time()
    #     print(f"{GREEN}All MFCC CSV files generated and saved in time {round(end_time - start_time, 3)}s{RESET}")

    # # Generate Heatmaps
    # start_time = time.time()
    # generate_mfcc_plots_from_csvs(save_csv_path, save_plot_path)
    # end_time = time.time()
    # print(f"{GREEN}All MFCC heatmaps generated and saved in time {round(end_time - start_time, 3)}s{RESET}")
    # print(f"\n{BLUE}------------------------------------------------------------------------------------------------------{RESET}\n")
    pass