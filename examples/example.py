# Note: conversion done automatically from example.Rmd
# Plot style may not be consistent
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns # spare the dependency
from pathlib import Path
from scipy import signal
from datetime import datetime, timedelta

# Set plotting style
# sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (6, 3)
plt.rcParams['figure.dpi'] = 100

# Import adi library
import adi

# ## Demo - 15-minute summary timepoints

# ### Load file, comments, and channels

# Load the file
file_path = Path("path/to/file.adicht")
f = adi.read_file(str(file_path))

# Extract comments and channels
comments = adi.extract_comments(f)
channels = adi.extract_channels(f)

# ### Get results around comment

# Find comments matching criteria
comment_tags = ["end", "HS"]
channel_tags = ["ABP", "VBP"]

# Search for matching comments
pattern = '|'.join(comment_tags)
matching_comments = comments[comments['text'].str.contains(pattern, case=False, na=False)]
print(matching_comments)

# Manually specify comment index
comment_id = 15
selected_comment = comments[comments['id'] == comment_id].iloc[0]

# Get channel IDs
selected_channels = channels[channels['name'].isin(channel_tags)]
channel_ids = selected_channels['id'].tolist()

# Extract data around comment
results_comment = adi.extract_comment_window(
    f,
    comment=selected_comment.to_dict(),
    seconds_before=10,
    seconds_after=15*60,
    channel_ids=channel_ids
)

# ### Plot

# Create the plot
fig, ax = plt.subplots(figsize=(8, 4))

# Get data
data = results_comment['data']
time = data['relative_time']
abp = data['ch1_ABP_mmHg']

# Apply Savitzky-Golay filter
abp_filtered = signal.savgol_filter(abp, window_length=201, polyorder=3)

# Plot raw and filtered data
ax.plot(time, abp, alpha=0.2, label='Raw')
ax.plot(time, abp_filtered, color='red', label='Filtered')

# Add vertical line at event time
ax.axvline(x=0, color='black', linestyle='--')

# Add event label
ax.text(0, ax.get_ylim()[0], results_comment['event'], 
        ha='center', va='bottom', bbox=dict(boxstyle='round', facecolor='white'))

# Add other comment annotations
for _, comment in comments.iterrows():
    time_diff = (comment['datetime'] - selected_comment['datetime']).total_seconds()
    if -10 < time_diff < 900:  # Only show comments in the window
        ax.text(time_diff, 10, comment['text'], rotation=45, fontsize=8)

# Labels and title
ax.set_xlabel('Relative time (s)')
ax.set_ylabel('ABP (mmHg)')
ax.set_title(f"{results_comment['event']}\nt₀ = {results_comment['time']}")
ax.legend()

plt.tight_layout()
plt.show()

# ### Get windows

# Get the original comment
comment = comments[comments['id'] == 15].iloc[0]
print(f"Original time: {comment['datetime']}")

# Generate timepoints
interval = 1 * 60 * 60  # 1 hr in seconds
timepoints = adi.generate_timepoints(comment['datetime'], interval_s=interval, n=3)

# Create timepoint windows
before = 0
after = 15 * 60  # 15 min in seconds
timepoint_windows = [adi.create_window(tp, seconds_before=before, seconds_after=after) 
                     for tp in timepoints]

print("Timepoint windows:")
for i, window in enumerate(timepoint_windows):
    print(f"Window {i}: {window[0]} to {window[1]}")

# ### Get results

# Extract data for all windows
results_windows = []
for window in timepoint_windows:
    result = adi.extract_window(f=f, window=window, channel_ids=channel_ids)
    results_windows.append(result)

# Combine all data into one DataFrame
results_df_list = []
for i, result in enumerate(results_windows):
    df = result['data'].copy()
    df['window_id'] = i
    results_df_list.append(df)

results_df = pd.concat(results_df_list, ignore_index=True)

# ### Plot combined

fig, ax = plt.subplots(figsize=(10, 4))

# Plot each window
for window_id in results_df['window_id'].unique():
    window_data = results_df[results_df['window_id'] == window_id]
    
    # Raw data
    ax.plot(window_data['datetime'], window_data['ch1_ABP_mmHg'], 
            alpha=0.2, color='gray')
    
    # Filtered data
    abp_filtered = signal.savgol_filter(window_data['ch1_ABP_mmHg'], 
                                       window_length=201, polyorder=3)
    ax.plot(window_data['datetime'], abp_filtered, 
            label=f'Window {window_id}')

ax.set_xlabel('Datetime')
ax.set_ylabel('ABP (mmHg)')
ax.set_title(f"{results_comment['event']}\n1 hr timepoints")
ax.legend()

plt.tight_layout()
plt.show()

# ### Plot overlay

fig, ax = plt.subplots(figsize=(8, 4))

# Define colors for each window
colors = plt.cm.viridis(np.linspace(0, 1, len(results_df['window_id'].unique())))

# Plot each window
for window_id, color in zip(results_df['window_id'].unique(), colors):
    window_data = results_df[results_df['window_id'] == window_id]
    
    # Raw data
    ax.plot(window_data['relative_time'], window_data['ch1_ABP_mmHg'], 
            alpha=0.2, color='gray')
    
    # Filtered data
    abp_filtered = signal.savgol_filter(window_data['ch1_ABP_mmHg'], 
                                       window_length=201, polyorder=3)
    ax.plot(window_data['relative_time'], abp_filtered, 
            color=color, label=f'Window {window_id}')

ax.set_xlabel('Relative time for each timepoint window (s)')
ax.set_ylabel('ABP (mmHg)')
ax.set_title(f"{results_comment['event']}\n1 hr timepoints")
ax.legend()

plt.tight_layout()
plt.show()

# ### Summary

# Calculate means by window
summary_stats = results_df.groupby('window_id').agg({
    'ch1_ABP_mmHg': 'mean',
    'ch2_VBP_mmHg': 'mean'
}).rename(columns={'ch1_ABP_mmHg': 'abp', 'ch2_VBP_mmHg': 'vbp'})

print(summary_stats)

# Plot summary
fig, ax = plt.subplots(figsize=(6, 4))

x = range(len(summary_stats))
ax.plot(x, summary_stats['abp'], 'o-', markersize=8)

# Set x-axis labels
labels = ["End of hemorrhage", "1 hr", "2 hr", "3 hr"]
ax.set_xticks(x)
ax.set_xticklabels(labels)

ax.set_xlabel('Timepoint')
ax.set_ylabel('ABP (mmHg)')
ax.set_title('Mean ABP at each timepoint')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()