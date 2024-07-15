import os
import glob
import matplotlib.pyplot as plt

def main():
    base_dir = '.'  # Directory to search for model directories
    file_pattern = 'mean_rewards_GDR.txt'
    files = glob.glob(os.path.join(base_dir, file_pattern))

    plt.figure(figsize=(12, 6))

    colors = plt.cm.get_cmap('tab10', len(files))  # Get a colormap with enough colors

    all_std_devs = []  # List to store all standard deviations encountered
    
    for idx, file_path in enumerate(files):
        seed = file_path.split('mean_rewards_seed:')[-1].split('.')[0]
        std_devs = []
        mean_rewards = []

        with open(file_path, 'r') as file:
            for line in file:
                std_dev, mean_reward = map(float, line.strip().split())
                std_devs.append(std_dev)
                mean_rewards.append(mean_reward)
                all_std_devs.append(std_dev)  # Collect all std_devs from all files

        plt.plot(std_devs, mean_rewards, marker='o', markersize=15, linewidth=4, label=f'Seed {seed}', color=colors(idx))

    # Ensure all unique std_devs are used on the x-axis
    unique_std_devs = sorted(set(all_std_devs))
    plt.xticks(unique_std_devs, fontsize=14)

    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)

    plt.xlabel('Standard Deviation', fontsize=15)
    plt.ylabel('Test Mean Reward', fontsize=15)
    plt.grid(True)
    plt.yticks(fontsize=14)
    plt.show()

if __name__ == '__main__':
    main()