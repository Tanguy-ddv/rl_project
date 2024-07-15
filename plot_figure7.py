
import json
import numpy as np
import matplotlib.pyplot as plt
import wandb

# Initialize wandb
wandb.init(project="plot_project")

# File names
file_names = ['step0.json', 'step1.json', 'step2.json', 'step4.json']

# Load data from JSON files
data = []
for file_name in file_names:
    with open(file_name, 'r') as file:
        data.append(json.load(file))

# Extract data
step0_data = data[0]
step1_data = data[1]
step2_data = data[2]
step4_data = data[3][:1350]  # Trimming the last 100 values

# Adjust x values
x_step0 = [i * 10 for i in range(len(step0_data))]
x_step1 = [i * 10 for i in range(len(step1_data))]
x_step2 = [i * 10 + 3000 for i in range(len(step2_data))]
x_step4 = [i * 10 + 1500 for i in range(len(step4_data))]

# Smooth function
def smooth(data, smooth_factor=0.05):
    smoothed = []
    last = data[0]
    for point in data:
        smoothed_val = last * (1 - smooth_factor) + point * smooth_factor
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

# Smooth the data
smooth_factor = 0.1
step0_data_smoothed = smooth(step0_data, smooth_factor)
step1_data_smoothed = smooth(step1_data, smooth_factor)
step2_data_smoothed = smooth(step2_data, smooth_factor)
step4_data_smoothed = smooth(step4_data, smooth_factor)

# Plot settings
line_width = 2.5  # Adjustable line width
point_size = 10   # Adjustable point size

# Create a plot
plt.figure(figsize=(10, 6))

# Plot the lines
plt.plot(x_step1, step1_data_smoothed, 'k', label='Pre-training in Source', linewidth=line_width)
plt.plot(x_step0, step0_data_smoothed, 'b', label='ADR-Source from the start', linewidth=line_width)
plt.plot(x_step4, step4_data_smoothed, 'g', label='ADR-Source after 1500 ep of pre-training', linewidth=line_width)
plt.plot(x_step2, step2_data_smoothed, 'r', label='ADR-Source after 3000 ep of pre-training', linewidth=line_width)




# Add red and green points at specific heights
height_red = 0.6  # Adjust the height of the red point (normalized)
height_green = 0.4  # Adjust the height of the green point (normalized)
plt.scatter([3000], [1029.512120969323], color='r', s=100, zorder=5)
plt.scatter([1500], [683.062863138861], color='g', s=100, zorder=5)

# Add labels and legend with increased fontsize
plt.xlabel('Episodes', fontsize=14)
plt.ylabel('Test Rewards', fontsize=14)
plt.legend(fontsize=12)

# Set tick parameters for both axes
plt.tick_params(axis='both', which='major', labelsize=14)

# Set x-axis limit
plt.xlim(0, 14000)

# Add custom x-axis ticks
current_ticks = plt.xticks()[0]
# Remove 2000 and add 1500, 3000
new_ticks = np.append(current_ticks[current_ticks != 2000], [1500, 3000])
plt.xticks(new_ticks)

# Show the plot on the screen
plt.show()

# Log the plot to wandb
wandb.log({"combined_plot": plt})

# Finish wandb session
wandb.finish()
