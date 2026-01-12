#!/usr/bin/env python3
import pandas as pd
import numpy as np

# Read cleaned data
df = pd.read_csv("inference_log_detailed.csv")
print(f"Processing {len(df)} samples...\n")

# Filter only committed (exclude uncertain if any)
committed = df[df['Prediksi Model'] != 'Uncertain'].copy()
uncertain = df[df['Prediksi Model'] == 'Uncertain'].copy()

print(f"Committed: {len(committed)}")
print(f"Uncertain: {len(uncertain)}")

# Calculate latency statistics (in ms, already in CSV)
latency_metrics = []

for col_name in ['Inference Time (ms)', 'Transport Latency (ms)', 
                 'Server ACK Latency (ms)', 'Total Response Time (ms)']:
    data = committed[col_name]
    latency_metrics.append({
        'Metric': col_name.replace(' (ms)', ''),
        'Median (ms)': f"{data.median():.1f}",
        'Mean (ms)': f"{data.mean():.1f}",
        'Min (ms)': f"{data.min():.1f}",
        'Max (ms)': f"{data.max():.1f}",
        'Std Dev (ms)': f"{data.std():.1f}"
    })

# Calculate decision metrics
total = len(df)
commit_rate = len(committed) / total * 100 if total > 0 else 0
uncertain_rate = len(uncertain) / total * 100 if total > 0 else 0
avg_confidence = committed['Confidence'].mean() if len(committed) > 0 else 0

decision_metrics = [
    {
        'Metric': 'Decision Metrics',
        'Median (ms)': 'Value',
        'Mean (ms)': '',
        'Min (ms)': '',
        'Max (ms)': '',
        'Std Dev (ms)': ''
    },
    {
        'Metric': 'Commit Rate',
        'Median (ms)': f"{commit_rate:.1f}%",
        'Mean (ms)': '',
        'Min (ms)': '',
        'Max (ms)': '',
        'Std Dev (ms)': ''
    },
    {
        'Metric': 'Uncertain Rate',
        'Median (ms)': f"{uncertain_rate:.1f}%",
        'Mean (ms)': '',
        'Min (ms)': '',
        'Max (ms)': '',
        'Std Dev (ms)': ''
    },
    {
        'Metric': 'Average Confidence (Committed)',
        'Median (ms)': f"{avg_confidence:.3f}",
        'Mean (ms)': '',
        'Min (ms)': '',
        'Max (ms)': '',
        'Std Dev (ms)': ''
    }
]

# Combine all metrics
all_metrics = latency_metrics + decision_metrics

# Create DataFrame
df_metrics = pd.DataFrame(all_metrics)

# Save to CSV
output_file = "tabel7_realtime.csv"
df_metrics.to_csv(output_file, index=False)
print(f"\n✅ Saved metrics to: {output_file}")

# Display results
print("\n" + "="*70)
print("REAL-TIME INFERENCE METRICS (15x per class)")
print("="*70)
print("\nLATENCY STATISTICS:")
for idx, row in enumerate(latency_metrics):
    print(f"\n{row['Metric']}:")
    print(f"  Median: {row['Median (ms)']:>8} | Mean: {row['Mean (ms)']:>8} | Std: {row['Std Dev (ms)']:>8}")
    print(f"  Min:    {row['Min (ms)']:>8} | Max:  {row['Max (ms)']:>8}")

print("\n DECISION METRICS:")
for metric in decision_metrics[1:]:  \
    print(f"  {metric['Metric']}: {metric['Median (ms)']}")

print("\n" + "="*70)
print(f"Metrics saved to: {output_file}")
print("="*70)
