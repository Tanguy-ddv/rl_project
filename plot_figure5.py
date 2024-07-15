import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import wandb
from matplotlib.ticker import FuncFormatter
import io
import numpy as np

# Function to read and process the file
def read_and_process_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    delta_025 = []
    delta_05 = []
    delta_075 = []
    
    for i, line in enumerate(lines):
        delta, timesteps, mean_reward = line.strip().split(", ")
        delta_value = float(delta.split(": ")[1])
        timesteps_value = int(timesteps.split(": ")[1])
        mean_reward_value = float(mean_reward.split(": ")[1])
        
        if delta_value == 0.25:
            delta_025.append((timesteps_value, mean_reward_value))
        elif delta_value == 0.5:
            delta_05.append((timesteps_value, mean_reward_value))
        elif delta_value == 0.75:
            delta_075.append((timesteps_value, mean_reward_value))
    
    # Apply smoothing using a moving average
    def smooth_data(data, smooth_factor=0.05):
        smoothed_data = []
        for i in range(len(data)):
            if i == 0:
                smoothed_value = data[i][1]
            else:
                smoothed_value = smooth_factor * data[i][1] + (1 - smooth_factor) * smoothed_data[-1][1]
            smoothed_data.append((data[i][0], smoothed_value))
        return smoothed_data
    
    delta_025_smoothed = smooth_data(delta_025)[0:200]
    delta_05_smoothed = smooth_data(delta_05)[0:200]
    delta_075_smoothed = smooth_data(delta_075)[0:200]
    
    return delta_025_smoothed, delta_05_smoothed, delta_075_smoothed

# Read and process the file
delta_025_avg, delta_05_avg, delta_075_avg = read_and_process_file('models_delta_UDR_5M_seed:42/result.txt')

# Initialize Weights & Biases
wandb.init(project='nome_progetto')

# Create the matplotlib plot
plt.figure(figsize=(12, 6))

# Plot data for Delta: 0.25 with green dots
plt.plot([x[0] for x in delta_025_avg], [x[1] for x in delta_025_avg], 'go-', label='Delta: 0.25', linewidth=2, markersize=5)

# Plot data for Delta: 0.5 with blue dots
plt.plot([x[0] for x in delta_05_avg], [x[1] for x in delta_05_avg], 'bo-', label='Delta: 0.5', linewidth=2, markersize=5)

# Plot data for Delta: 0.75 with red dots
plt.plot([x[0] for x in delta_075_avg], [x[1] for x in delta_075_avg], 'ro-', label='Delta: 0.75', linewidth=2, markersize=5)

# Custom formatter function to display ticks in 'M'
def millions(x, pos):
    'The two args are the value and tick position'
    return '%1.1fM' % (x * 1e-6)

plt.gca().xaxis.set_major_formatter(FuncFormatter(millions))

plt.xticks(fontsize=20)
plt.yticks(fontsize=17)

plt.xlabel('Timesteps', fontsize=17)
plt.ylabel('Mean Test Reward Every 50K Timesteps', fontsize=14)
plt.legend(fontsize=14)
plt.grid(True)

# Save the plot in a memory buffer
plot_buffer = io.BytesIO()
plt.savefig(plot_buffer, format='png')
plot_buffer.seek(0)

# Read the image from the buffer
img = mpimg.imread(plot_buffer)

# Log the image to Weights & Biases
wandb.log({"Mean Test Reward Plot": wandb.Image(img)})

plt.show()

# Close the matplotlib plot
plt.close()
