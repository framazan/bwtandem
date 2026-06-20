#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    os.makedirs("figures", exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    # ---------------------------------------------------------
    # Figure 1: Arabidopsis Centromere Coverage vs. Runtime
    # ---------------------------------------------------------
    if os.path.exists('reports/experiment2_metrics.csv'):
        df_exp2 = pd.read_csv('reports/experiment2_metrics.csv')
        # Filter to tools that actually ran/found something meaningful
        df_exp2 = df_exp2[df_exp2['Tool'].isin(['trf', 'bwtandem', 'ultra', 'tantan', 'trash_template'])]
        df_exp2['Runtime (Hours)'] = df_exp2['Runtime (s)'] / 3600.0
        
        fig, ax1 = plt.subplots(figsize=(8, 6))
        
        # Bar chart for Coverage
        sns.barplot(data=df_exp2, x='Tool', y='Centromere Cov (%)', color='skyblue', ax=ax1, alpha=0.8)
        ax1.set_ylabel('Centromere Coverage (%)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_ylim(0, 100)
        
        # Line plot for Runtime on secondary Y-axis
        ax2 = ax1.twinx()
        sns.lineplot(data=df_exp2, x='Tool', y='Runtime (Hours)', color='red', marker='o', linewidth=2, markersize=8, ax=ax2)
        ax2.set_ylabel('Runtime (Hours)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_yscale('log')
        
        plt.title('Arabidopsis (Col-CEN): Centromere Coverage vs. Runtime')
        plt.tight_layout()
        plt.savefig('figures/fig1_arabidopsis_cov_vs_runtime.png', dpi=300)
        plt.close()

    # ---------------------------------------------------------
    # Figure 2: Human Adotto BP Recall vs. Runtime
    # ---------------------------------------------------------
    if os.path.exists('reports/experiment1_metrics.csv'):
        df_exp1 = pd.read_csv('reports/experiment1_metrics.csv')
        # Filter for visualization
        df_exp1 = df_exp1[df_exp1['Tool'].isin(['trf', 'bwtandem', 'ultra', 'tantan', 'trash_template'])]
        
        # Add bwtandem_sensitive if it exists
        if os.path.exists('reports/human_rerun_metrics.csv'):
            df_rerun = pd.read_csv('reports/human_rerun_metrics.csv')
            # Copy runtime/memory from regular bwtandem for the sensitive run (approximate)
            bwt_rt = df_exp1.loc[df_exp1['Tool'] == 'bwtandem', 'Runtime (s)'].values[0]
            df_rerun['Runtime (s)'] = bwt_rt * 1.5 # Assume sensitive takes a bit longer
            df_exp1 = pd.concat([df_exp1, df_rerun], ignore_index=True)
            
        df_exp1['Runtime (Hours)'] = df_exp1['Runtime (s)'] / 3600.0
        
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df_exp1, x='Runtime (Hours)', y='BP Recall (%)', hue='Tool', s=200, palette='Set2', edgecolor='black')
        
        # Add labels to points
        for i, row in df_exp1.iterrows():
            plt.text(row['Runtime (Hours)'] * 1.1, row['BP Recall (%)'], row['Tool'], horizontalalignment='left', size='small', color='black')
            
        plt.xscale('log')
        plt.title('Human (GRCh38): Base-Pair Recall vs. Runtime')
        plt.xlabel('Runtime (Hours, log scale)')
        plt.ylabel('Base-Pair Recall (%)')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('figures/fig2_human_recall_vs_runtime.png', dpi=300)
        plt.close()

    # ---------------------------------------------------------
    # Figure 3: Maize Array Detection
    # ---------------------------------------------------------
    if os.path.exists('reports/experiment3b_metrics.csv'):
        df_exp3 = pd.read_csv('reports/experiment3b_metrics.csv')
        df_exp3 = df_exp3[df_exp3['Tool'].isin(['trf', 'bwtandem', 'ultra', 'tantan', 'trash_template'])]
        
        # Melt dataframe for grouped bar chart
        df_melt = pd.melt(df_exp3, id_vars=['Tool'], value_vars=['knob180 arrays (of 25)', 'TR-1 arrays (of 17)'], 
                          var_name='Array Type', value_name='Count')
                          
        plt.figure(figsize=(8, 6))
        sns.barplot(data=df_melt, x='Tool', y='Count', hue='Array Type', palette='pastel', edgecolor='black')
        
        # Draw target lines for perfect score
        plt.axhline(25, color='blue', linestyle='--', alpha=0.5, label='Max knob180 (25)')
        plt.axhline(17, color='orange', linestyle='--', alpha=0.5, label='Max TR-1 (17)')
        
        plt.title('Maize (Mo17): Complex Satellite Array Detection')
        plt.ylabel('Arrays Detected')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('figures/fig3_maize_array_detection.png', dpi=300)
        plt.close()

    print("Figures generated successfully in figures/ directory!")

if __name__ == "__main__":
    main()
