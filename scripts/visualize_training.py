import matplotlib.pyplot as plt
import re
import sys
import os

def plot_loss_from_log(log_file):
    losses = []
    epochs = []
    
    if not os.path.exists(log_file):
        print(f"Error: File {log_file} not found.")
        return

    with open(log_file, 'r') as f:
        for line in f:
            # Look for "Epoch X Average Loss: Y.YYYY"
            match = re.search(r"Epoch (\d+) Average Loss: ([\d\.]+)", line)
            if match:
                epochs.append(int(match.group(1)))
                losses.append(float(match.group(2)))
                
    if not losses:
        print("No loss data found in log file.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, marker='o', linestyle='-', color='b')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True)
    output_file = log_file + '.png'
    plt.savefig(output_file)
    print(f"Loss plot saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/visualize_training.py <log_file>")
        print("Example: python scripts/visualize_training.py train_output.log")
    else:
        plot_loss_from_log(sys.argv[1])
