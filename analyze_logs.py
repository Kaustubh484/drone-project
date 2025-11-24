import sys
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

def analyze_episode(file_path):
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # --- 1. Auto-Detect Column Names ---
    if 'pos_e' in df.columns:
        x_col, y_col, z_col = 'pos_e', 'pos_n', 'pos_d'
        title_prefix = "Absolute"
    elif 'rel_pos_e' in df.columns:
        x_col, y_col, z_col = 'rel_pos_e', 'rel_pos_n', 'rel_pos_d'
        title_prefix = "Relative (to Target)"
    else:
        print("Error: Could not find position columns (pos_e or rel_pos_e) in CSV.")
        print("Available columns:", df.columns.tolist())
        return

    # Check for action columns (Velocity vs Thrust/Rate)
    if 'act_vz' in df.columns:
        act_z_col = 'act_vz'
        act_label = 'Velocity Z (m/s)'
    elif 'act_thrust' in df.columns:
        act_z_col = 'act_thrust'
        act_label = 'Thrust (0-1)'
    else:
        act_z_col = None

    # --- 2. Plotting ---
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # Plot 1: Top-Down Trajectory
    # Invert X/Y logic: Standard maps use East as X, North as Y
    axs[0].plot(df[x_col], df[y_col], label='Drone Path', color='blue', marker='.', markersize=2)
    axs[0].plot(df[x_col].iloc[0], df[y_col].iloc[0], 'go', label='Start') 
    axs[0].plot(df[x_col].iloc[-1], df[y_col].iloc[-1], 'rx', label='End')
    axs[0].set_title(f'{title_prefix} Trajectory (Top-Down)')
    axs[0].set_xlabel(f'East ({x_col})')
    axs[0].set_ylabel(f'North ({y_col})')
    axs[0].axis('equal') 
    axs[0].grid(True)
    axs[0].legend()

    # Plot 2: Altitude & Vertical Action
    # Note: NED 'Down' is positive, so we negate it to plot Altitude
    axs[1].plot(df['step'], -df[z_col], label='Altitude (-Down)', color='black', linewidth=2)
    
    if act_z_col:
        # Create a twin axis for action so altitude scaling doesn't squash it
        ax2 = axs[1].twinx()
        ax2.plot(df['step'], df[act_z_col], label=f'Cmd: {act_label}', color='orange', alpha=0.5)
        ax2.set_ylabel(act_label, color='orange')
        # Align zero if possible, or just let it float
    
    axs[1].set_title('Vertical Stability Check')
    axs[1].set_ylabel('Altitude (m)')
    axs[1].grid(True)
    axs[1].legend(loc='upper left')

    # Plot 3: Reward Breakdown
    # Check which reward components exist
    found_rewards = False
    if 'rew_dist' in df.columns:
        axs[2].plot(df['step'], df['rew_dist'], label='Dist Reward', color='green', alpha=0.7)
        found_rewards = True
    if 'rew_instability' in df.columns:
        axs[2].plot(df['step'], df['rew_instability'], label='Unstable Penalty', color='red', alpha=0.7)
        found_rewards = True
    if 'rew_crash' in df.columns:
        # Mark crashes
        crashes = df[df['rew_crash'] < 0]
        for step in crashes['step']:
            axs[2].axvline(x=step, color='black', linestyle='--', label='CRASH')
        found_rewards = True
    
    if not found_rewards:
        axs[2].text(0.5, 0.5, "No granular reward data found", ha='center')
    else:
        axs[2].legend()

    axs[2].set_title('Reward Components')
    axs[2].set_xlabel('Step')
    axs[2].set_ylabel('Value')
    axs[2].grid(True)

    plt.tight_layout(pad = 3.0)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_logs.py <path_to_csv>")
    else:
        analyze_episode(sys.argv[1])