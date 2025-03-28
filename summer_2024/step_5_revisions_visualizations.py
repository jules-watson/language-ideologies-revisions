"""
Plot graphs representing the revision and justification statistics.

TODO: make this use the stacked_grouped_bar_graph function from visualizations.py

Author: Raymond Liu
Date: Aug 2024
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from common import load_json

from random_scripts.plot_gender_association_scores import plot_gender_association_scores

from matplotlib.colors import ListedColormap
from constants import EXPERIMENT_PATH, MODEL_NAMES

# Colors for revision bar graph
# base_colors = [
#     (0/255, 121/255, 255/255, 1),    # blue - alternative wording
#     (0/255, 223/255, 162/255, 1),    # green - neutral
#     (255/255, 165/255, 0/255, 1),  # orange - masculine
#     (255/255, 0/255, 96/255, 1)      # red - feminine
# ]

base_colors = [
    (.28, .51, .40, 1),  # green - alternattive wording
    (.37, .35, .62, 1),    # blue-purple - neutral
    (.92, .68, .32, 1),    # yellow - masculine
    (.74, .35, .27, 1)      # orange - feminine
]

model_names_shortened = {
    'gpt-3.5-turbo': 'GPT-3.5',
    'gpt-4-turbo': 'GPT-4',
    'gpt-4o': 'GPT',
    'llama-3.1-8B-Instruct': 'llama',
    'gemma-2-9b-it': 'gemma',
    'Mistral-Nemo-Instruct-2407': 'mistral'
}


def plot_revision_rates(data_frames, output_path, prompt_wording, axe=None, legend=True, **kwargs):
    """Create a clustered stacked bar plot.

    data_frames is a list of pandas dataframes. The dataframes should have
        identical columns and index

    Adapted from https://github.com/juliawatson/language-ideologies-2024/blob/ae9ddbeb2cb4c78dc8cbcd8f72f7de670b6675ab/fall_2023_main/exploratory/visualize_by_gender.py#L45
    """
    n_df = len(data_frames)
    n_col = len(data_frames[0].columns)-2  # = total number of columns - number of columns before removed_rate
    n_ind = len(data_frames[0].index)

    save_figure = False
    if axe is None:
        if n_ind > 2:
            figsize=[8, 5]
        else:
            figsize=[4.8, 6]

        plt.figure(figsize=figsize)
        axe = plt.subplot(111)
        save_figure = True

    alphas = [0.4, 0.6, 0.8, 1]
    for i, df in enumerate(data_frames):  # for each dataframe
        alpha = alphas[i]
        colormap = ListedColormap([
            (r, g, b, alpha) for r, g, b, _ in base_colors
        ])
        axe = df.plot(kind="bar",
                      linewidth=0,
                      stacked=True,
                      ax=axe,
                      legend=True,
                      grid=False,
                      colormap=colormap,
                      **kwargs)  # make bar plots

    h, l = axe.get_legend_handles_labels()  # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col):  # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i + n_col]):
            for rect in pa.patches:  # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_width(1 / float(n_df + 1))

    xtick_offset = 1 / float(n_df + 1) / 2
    xticks = (np.arange(0, 2 * n_ind, 2) + xtick_offset) / 2.

    model_ticks = []
    for item in xticks:
        if n_df == 2:
            model_ticks += [item - xtick_offset, item + xtick_offset]
        elif n_df == 3:
            model_ticks += [item - xtick_offset, item + xtick_offset, item + 3*xtick_offset]
        elif n_df == 4:
            # model_ticks += [item - 3*xtick_offset, item - xtick_offset, item + xtick_offset, item + 3*xtick_offset]
            model_ticks += [item - 2*xtick_offset, item, item + 2*xtick_offset, item + 4*xtick_offset]
    # xticks = sorted(list(xticks) + model_ticks)
    xticks = sorted(model_ticks)

    index_items = list(df.index)
    xtick_labels = []
    for item in index_items:
        labels = [model_names_shortened[model_name] for model_name in MODEL_NAMES]
        # labels = [1, 2, 3, 4]
        # labels.insert(1, '\n')
        xtick_labels += (labels)

    axe.set_xticks(xticks)
    axe.set_xticklabels(xtick_labels, rotation=45, fontsize=8)
    axe.tick_params(axis=u'both', which=u'both',length=0)

    # Create a new axis below the primary x-axis for gender labels
    axe2 = axe.secondary_xaxis('bottom')
    axe2.xaxis.set_ticks_position('none') 
    axe2.spines['bottom'].set_visible(False)  # Hides the horizontal line (spine)
    axe2.set_xticks((np.arange(0, 2 * n_ind, 2) + xtick_offset * 3) / 2.)
    axe2.set_xticklabels(['\n\nneutral', '\n\nmasculine', '\n\nfeminine'])
    axe2.spines['bottom'].set_position(('outward', 8))  # Adjust the distance of the secondary axis below

    axe.set_xlabel("\nRole noun gender in original sentence", labelpad=10)
    axe.set_ylabel("Proportion")

    axe.set_title(f"Revision rates for: {prompt_wording}", pad=60)
    axe.set_xlim(xticks[0] - xtick_offset * 4, xticks[-1] + xtick_offset * 4)
    axe.set_ylim(0, 1)

    if legend:
        legend_labels = ["Revised to alternative wording", "Revised to neutral variant", "Revised to masculine variant", "Revised to feminine variant"]
        handles, _ = axe.get_legend_handles_labels()
        handles = handles[:4]
        l1 = axe.legend(handles, legend_labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.2))
        plt.subplots_adjust(top=0.85)

    if save_figure:
        plt.savefig(output_path, #bbox_extra_artists=(l1,),
                    bbox_inches='tight',
                    dpi=300)


def plot_justification_words(justification_freqs_df, output_path, desired_words=None):
    """
    Plot the frequency of certain words in the justification.
    """
    if desired_words is None: # plot all the words
        desired_words = justification_freqs_df['word'].tolist()
    
    # Filter the dataframe for only the desired words
    df_filtered = justification_freqs_df[justification_freqs_df['word'].isin(desired_words)]

    # Prepare data for plotting
    words = df_filtered['word'].tolist()
    settings = MODEL_NAMES
    names = [f'frequency_{model_name}' for model_name in MODEL_NAMES]
    frequencies = df_filtered[names].values

    # Set the number of groups and bars
    n_groups = len(words)
    n_bars = len(settings)

    # Set up the bar width and positions
    bar_width = 0.2
    index = np.arange(n_groups)

    # Create the plot
    fig, ax = plt.subplots()

    # Generate a unique color for each word
    colors = plt.cm.get_cmap('tab10', n_groups)

    # Plot each word's frequencies as a group of bars
    alphas = [0.7, 0.4, 1]
    for i in range(n_groups):
        color = colors(i)
        for j in range(n_bars):
            ax.bar(index[i] + j * bar_width, frequencies[i][j], bar_width,
                label=f'{words[i]}',
                color=color, alpha=alphas[j])
            ax.text(index[i] + j * bar_width, frequencies[i][j] + 0 * max(frequencies[i]), settings[j],
                ha='center', va='top', rotation=90, fontsize=10)

    # Add labels, title, and legend
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(words)
    #ax.legend(loc='best')

    # Show the plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # adjust the layout to make room for the title and legend
    plt.savefig(output_path, #bbox_extra_artists=(l1,),
                bbox_inches='tight',
                dpi=700)





def main(config):
    dirname = "/".join(config.split("/")[:-1])  
    config_path = f"{dirname}/config.json"  # path to the config file
    config = load_json(config_path)  # load the config file

    df_paths = [f'{dirname}/{model_name}/revision_stats.csv' for model_name in MODEL_NAMES]  # for every model, retrieve the path to the revision_stats csv 
    dfs = [pd.read_csv(df_path) for df_path in df_paths]  # load the revision_stats csvs for every model as dataframes
    
    task_wording_dict = config['task_wording']  # get the task wording dict 
    for prompt_wording in task_wording_dict.keys():  # every task wording name
    
        filtered_revision_stats_dfs = [df[df['prompt_wording'] == prompt_wording] for df in dfs] # filter the dataframes to keep only the rows for the current prompt wording

        # Escape slashes in prompt_wording
        escaped_prompt_wording = prompt_wording.replace('/', ' ')        
        
        # revision rates plot for the current prompt
        output_path = f'{dirname}/revision_bar_graph_{escaped_prompt_wording}.png'
        plot_revision_rates(data_frames=filtered_revision_stats_dfs,
                            output_path=output_path,
                            prompt_wording=prompt_wording)

        # plot sentences' feminine and genderedness scores against their revision status
        if prompt_wording == 'unknown gender' and 'gender_association_scores' in config: 
            revision_df_paths = [f'{dirname}/{model_name}/revision.csv' for model_name in MODEL_NAMES]  # for every model, retrieve the path to the revision csv 
            revision_dfs = [pd.read_csv(revision_df_path) for revision_df_path in revision_df_paths]  # load the revision csvs for every model as dataframes
            filtered_revision_dfs = [df[df['task_wording'] == 'unknown gender'] for df in revision_dfs]

            gender_associations_df = pd.read_csv(config['gender_association_scores'])

            plot_gender_association_scores(
                data_frames=filtered_revision_dfs,
                output_dir=dirname,
                model_names=MODEL_NAMES, 
                gender_associations_df=gender_associations_df
            )

    # justification word counts plot
    # justification_freqs_df = pd.read_csv(f"{dirname}/justification_word_frequencies.csv")
    # output_path = f'{dirname}/justification_plot.png'
    # plot_justification_words(justification_freqs_df,
    #                          output_path = output_path,
    #                          desired_words=['professional', 'tone', 'clarity', 'inclusive', 'engaging'])

        


if __name__=="__main__":
    main(f'{EXPERIMENT_PATH}/config.json')
