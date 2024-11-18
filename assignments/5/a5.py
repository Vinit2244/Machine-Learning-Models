import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
import librosa
import librosa.display
import pandas as pd
import warnings
from hmmlearn import hmm

warnings.filterwarnings("ignore", category=UserWarning, message=".*tight_layout.*")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from models.kde.kde import KDE
from models.gmm.gmm import GMM

# Colors for printing for better readability
BLUE = "\033[34m"
GREEN = "\033[32m"
RED = "\033[31m"
MAGENTA = "\033[35m"
RESET = "\033[0m"

def split_into_train_test_val(features, labels, train_size=0.8, test_size=0.2, val_size=None, seed=None):
    # Calculate the validation size if not provided
    if val_size is None:
        val_size = 1.0 - train_size - test_size
        if val_size < 0.0:
            raise ValueError("Sum of train and test sizes exceeds 1.0")
    
    # Check if the sizes sum up to 1.0 (or 100%)
    if not (train_size + test_size + val_size == 1.0):
        raise ValueError("Train, test, and validation sizes must sum up to 1.0.")
    
    # Set the random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    
    # Get the total number of samples
    n_samples = len(features)
    
    # Create a shuffled array of indices
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # Compute split points
    train_end = int(n_samples * train_size)
    test_end = train_end + int(n_samples * test_size)
    
    # Split indices into training, testing, and validation based on the sizes
    train_indices = indices[:train_end]
    test_indices = indices[train_end:test_end]
    val_indices = indices[test_end:]
    
    # Split features and labels into training, testing, and validation sets
    X_train = [features[i] for i in train_indices]
    X_test = [features[i] for i in test_indices]
    X_val = [features[i] for i in val_indices]
    
    y_train = labels[train_indices]
    y_test = labels[test_indices]
    y_val = labels[val_indices]
    
    return X_train, X_test, X_val, y_train, y_test, y_val

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
                    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft) # Generally 13 MFCCs are used for speech recognition
                    
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
                        img = librosa.display.specshow(mfccs, x_axis='time', y_axis='mel', ax=ax, cmap='inferno')

                        ax.set_title(f'Clip {file_index}')
                        ax.set_xlabel('Time')
                        ax.set_ylabel('MFCC Coefficients (Hz)')
                        ax.tick_params(axis='x', rotation=90)
                    else:
                        # If the file doesn't exist, display a placeholder message
                        ax = axes[i, j]
                        ax.text(0.5, 0.5, 'File not found', fontsize=10, ha='center', va='center')
                        ax.axis('off')

            # Add colorbar to the figure for decibel scale reference
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Position of the colorbar
            fig.colorbar(img, cax=cbar_ax, format='%+2.0f dB')

            # Adjust layout and save the plot
            plt.tight_layout(rect=[0, 0, 0.9, 0.96])  # Adjust the right space for the color bar and title
            save_file_path = os.path.join(save_plot_path, f'{person}_digit_{digit}_mfcc_heatmap.png')
            plt.savefig(save_file_path, dpi=300)
            plt.close(fig)

def read_mfcc_features_with_labels(folder_path):
    data = []
    labels = []

    # Loop through all files in the directory
    for filename in os.listdir(folder_path):
        # Check if the file is a CSV file
        if filename.endswith(".csv"):
            # Extract the label from the filename (first digit)
            label = int(filename.split('_')[0])
            
            # Read the CSV file into a DataFrame
            file_path = os.path.join(folder_path, filename)
            mfcc_features = pd.read_csv(file_path, header=None).to_numpy()
            
            data.append(mfcc_features)
            labels.append(label)

    return data, np.array(labels)

def sequences_data_generator(num_sequences, min_length=0, max_length=16):
    sequences = []
    labels = []

    for _ in range(num_sequences):
        length = np.random.randint(1, max_length + 1)  # Random length between 1 and max_length
        sequence = np.random.randint(0, 2, length)  # Random binary sequence of the given length
        sequences.append(sequence)
        labels.append(np.sum(sequence))  # Count of '1's in the sequence

    return sequences, np.array(labels)

if __name__ == "__main__":
    # =============================================================================
    #                            Generate synthetic data
    # =============================================================================
    x_large, y_large = generate_circle_points(num_points=3000, radius=2, noise=0.2)
    x_small, y_small = generate_circle_points(num_points=500, radius=0.25, center=(1, 1), noise=0.05)
    x = np.concatenate([x_large, x_small])
    y = np.concatenate([y_large, y_small])
    data = np.column_stack([x, y])
    print(f"{GREEN}Synthetic Data Generated{RESET}")
    print(f"\n{BLUE}------------------------------------------------------------------------------------------------------{RESET}\n")

    # =============================================================================
    #                            Visualize synthetic data
    # =============================================================================
    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, color='black', s=1, alpha=0.6)
    plt.title("Synthetic Data")
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.grid(True)
    plt.savefig("./assignments/5/figures/synthetic_data.png")
    plt.close()
    print(f"{GREEN}Synthetic Data Visualized and Saved{RESET}")
    print(f"\n{BLUE}------------------------------------------------------------------------------------------------------{RESET}\n")

    # =============================================================================
    #                           Apply KDE on synthetic data
    # =============================================================================
    kde = KDE(bandwidth=0.5, kernel='gaussian')
    kde.fit(data)
    start_time = time.time()
    kde.visualize(plot_type='3d', save_as="./assignments/5/figures/kde_vs_gmm/kde_3d.gif")
    end_time = time.time()
    print(f"{GREEN}KDE 3D plot saved in time {round(end_time - start_time, 3)}s{RESET}")
    start_time = time.time()
    kde.visualize(plot_type='contour', save_as="./assignments/5/figures/kde_vs_gmm/kde_contour.png")
    end_time = time.time()
    print(f"{GREEN}KDE contour plot saved in time {round(end_time - start_time, 3)}s{RESET}")
    print(f"\n{BLUE}------------------------------------------------------------------------------------------------------{RESET}\n")

    # =============================================================================
    #                            Apply GMM on synthetic data
    # =============================================================================
    for n_components in range(2, 6):
        print(f"{MAGENTA}Fitting GMM with n_components = {n_components}{RESET}")
        gmm_model = GMM(n_components)
        gmm_model.load_data(data)
        start_time = time.time()
        epochs_taken = gmm_model.fit()
        end_time = time.time()
        print(f"{GREEN}GMM fitted, epochs taken = {epochs_taken}{RESET}")
        print(f"\t{GREEN}Time taken to fit: {round(end_time - start_time, 5)} s{RESET}")
        overall_log_likelihood = round(gmm_model.get_log_likelihood(), 5)
        print(f"\t{GREEN}Final Log Likelihood: {overall_log_likelihood}{RESET}")
        gmm_model.visualise(save_as=f"./assignments/5/figures/kde_vs_gmm/gmm_3d_{n_components}.gif", plot_type='3d')
        print(f"{GREEN}GMM 3D Plot saved{RESET}")
        gmm_model.visualise(save_as=f"./assignments/5/figures/kde_vs_gmm/gmm_contour_{n_components}.png", plot_type='contour')
        print(f"{GREEN}GMM Contour Plot saved\n{RESET}")
    print(f"\n{BLUE}------------------------------------------------------------------------------------------------------{RESET}\n")

    # =============================================================================
    #                    Save MFCCs to CSVs and generate heatmaps
    # =============================================================================
    folder_path = './data/external/free-spoken-digit-dataset-fsdd/recordings'
    save_plot_path = './assignments/5/figures/mfcc_heatmaps'
    save_csv_path = './data/interim/5/mfcc'

    # Check if already exist
    expected_csv_count = 3000  # 10 digits * 6 people * 50 recordings
    csv_files = [file for file in os.listdir(save_csv_path) if file.endswith('.csv')]

    if len(csv_files) == expected_csv_count:
        print(f"{GREEN}All MFCC CSV files already exist. Skipping CSV generation.{RESET}")
    else:
        start_time = time.time()
        save_all_mfccs_to_csvs(folder_path, save_csv_path)
        end_time = time.time()
        print(f"{GREEN}All MFCC CSV files generated and saved in time {round(end_time - start_time, 3)}s{RESET}")

    # Generate Heatmaps
    start_time = time.time()
    generate_mfcc_plots_from_csvs(save_csv_path, save_plot_path)
    end_time = time.time()
    print(f"{GREEN}All MFCC heatmaps generated and saved in time {round(end_time - start_time, 3)}s{RESET}")
    print(f"\n{BLUE}------------------------------------------------------------------------------------------------------{RESET}\n")

    # =============================================================================
    #                    Training HMM model for each digit
    # =============================================================================
    mfcc_folder_path = './data/interim/5/mfcc'
    data, labels = read_mfcc_features_with_labels(mfcc_folder_path)
    print(f"{GREEN}MFCC features loaded{RESET}")

    X_train, X_test, X_val, y_train, y_test, y_val = split_into_train_test_val(data, labels, train_size=0.8, test_size=0.2, seed=42)
    print(f"{GREEN}Data split into training and testing sets{RESET}")

    # Prepare separate HMM models for each digit (0-9)
    digit_models = {}
    n_components = 20  # Number of states in the HMM (tune as needed)

    for digit in range(10):
        # Filter training data for the current digit
        digit_mfccs = [mfcc for mfcc, label in zip(X_train, y_train) if label == digit]
        
        # Combine all the MFCC arrays into one for training
        lengths = [mfcc.shape[1] for mfcc in digit_mfccs]  # Sequence lengths
        concatenated_features = np.hstack(digit_mfccs)  # Combine features horizontally

        # Train the HMM for the current digit
        model = hmm.GaussianHMM(n_components=n_components, covariance_type='diag', n_iter=1000)
        model.fit(concatenated_features.T, lengths)
        
        # Save the trained model
        digit_models[digit] = model
        print(f"{GREEN}HMM model trained for digit {digit}{RESET}")

    # Evaluate models on the test set
    correct_predictions = 0

    for mfcc, true_label in zip(X_test, y_test):
        scores = {}
        
        # Calculate log-likelihood for each model
        for digit, model in digit_models.items():
            try:
                score = model.score(mfcc.T)
                scores[digit] = score
            except:
                scores[digit] = -np.inf  # If scoring fails, assign a low score

        # Predict the digit with the highest score
        predicted_label = max(scores, key=scores.get)
        
        if predicted_label == true_label:
            correct_predictions += 1

    # Calculate accuracy
    accuracy = correct_predictions / len(y_test)
    print(f'{GREEN}Accuracy on provided test set: {round(accuracy * 100, 2)}%{RESET}')
    
    # Testing the model on my own voice
    print(f"\n{MAGENTA}Now testing on my own recordings{RESET}")
    digits = range(10)
    mfcc_features = []
    labels = []

    for digit in digits:
        file_path = f"./data/interim/5/my_recordings/{digit}_vinit.wav"
        y, sr = librosa.load(file_path, sr=None)
        n_fft = min(2048, len(y))
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft) # Generally 13 MFCCs are used for speech recognition

        # Create the plot
        plt.figure(figsize=(12, 6))
        img = librosa.display.specshow(mfccs, x_axis='time', y_axis='mel', cmap='inferno')
        plt.title('MFCC Heatmap for Digit')
        plt.xlabel('Time')
        plt.ylabel('MFCC Coefficients (Hz)')
        plt.tick_params(axis='x', rotation=90)  # Rotate x-axis labels by 90 degrees
        cbar = plt.colorbar(img, format='%+2.0f dB')
        cbar.set_label('dB')  # Label for the color bar
        plt.tight_layout()
        plt.savefig(f"./assignments/5/figures/my_recordings_heatmaps/{digit}_vinit_mfcc_heatmap.png", dpi=300)
        plt.close()

        mfcc_features.append(mfccs)
        labels.append(digit)
    
    correct_predictions = 0

    for mfcc, true_label in zip(mfcc_features, labels):
        scores = {}
        
        # Calculate log-likelihood for each model
        for digit, model in digit_models.items():
            try:
                score = model.score(mfcc.T)
                scores[digit] = score
            except:
                scores[digit] = -np.inf
            
        # Predict the digit with the highest score
        predicted_label = max(scores, key=scores.get)

        if predicted_label == true_label:
            print(f"{BLUE}True Label: {true_label}, Predicted Label: {predicted_label}{RESET}")
            correct_predictions += 1

        else:
            print(f"{RED}True Label: {true_label}, Predicted Label: {predicted_label}{RESET}")

    # Calculate accuracy
    accuracy = correct_predictions / len(labels)
    print(f'{GREEN}Accuracy on my own recordings: {round(accuracy * 100, 2)}%{RESET}')
    print(f"\n{BLUE}------------------------------------------------------------------------------------------------------{RESET}\n")

    # =============================================================================
    #                                   Bits counting RNN
    # =============================================================================
    # Did in separate .ipynb file

    # =============================================================================
    #                                   OCR
    # =============================================================================
    # Did in separate .ipynb file
