import matplotlib.pyplot as plt
import numpy as np

from constants import EXPERIMENT_PATH, MODEL_NAMES

model_names_shortened = {
    'gpt-3.5-turbo': 'GPT-3.5',
    'gpt-4-turbo': 'GPT-4',
    'gpt-4o': 'GPT-4o',
    'llama-3.1-8B-Instruct': 'Llama-3.1-8B'
}


def stacked_grouped_bar_graph(data_frames, output_path, prompt_wording, label_col, axe=None, legend=True, **kwargs):
    """Create a clustered stacked bar plot.

    data_frames is a list of pandas dataframes. The dataframes should have
        identical columns and index

    Adapted from https://github.com/juliawatson/language-ideologies-2024/blob/ae9ddbeb2cb4c78dc8cbcd8f72f7de670b6675ab/fall_2023_main/exploratory/visualize_by_gender.py#L45
    """
    n_df = len(data_frames)
    n_col = len(data_frames[0].columns)-2  # = total number of columns - number of columns before removed_rate
    for data_frame in data_frames:
        assert len(data_frame.columns) == n_col + 2
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

    alphas = [0.7, 0.4, 1]
    for i, df in enumerate(data_frames):  # for each dataframe
        alpha = alphas[i]
        # colormap = ListedColormap([
        #     (r, g, b, alpha) for r, g, b, _ in base_colors
        # ])
        axe = df.plot(kind="bar",
                      linewidth=0,
                      stacked=True,
                      ax=axe,
                      legend=True,
                      grid=False,
                    #   colormap=colormap,
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
        else:
            model_ticks += [item - xtick_offset, item + xtick_offset, item + 3*xtick_offset]
    xticks = sorted(list(xticks) + model_ticks)

    index_items = list(df.index)
    xtick_labels = []
    for item in index_items:
        labels = [model_names_shortened[model_name] for model_name in MODEL_NAMES]
        labels.insert(1, '\n')
        xtick_labels += (labels)

    axe.set_xticks(xticks)
    axe.set_xticklabels(xtick_labels, rotation=0, fontsize=8)
    axe.tick_params(axis=u'both', which=u'both',length=0)

    # Create a new axis below the primary x-axis for gender labels
    axe2 = axe.secondary_xaxis('bottom')
    axe2.xaxis.set_ticks_position('none') 
    axe2.spines['bottom'].set_visible(False)  # Hides the horizontal line (spine)
    axe2.set_xticks((np.arange(0, 2 * n_ind, 2) + xtick_offset) / 2.)
    axe2.set_xticklabels(list(data_frames[0][label_col]))
    axe2.spines['bottom'].set_position(('outward', 8))  # Adjust the distance of the secondary axis below

    axe.set_xlabel("Role noun gender in original sentence", labelpad=10)
    axe.set_ylabel("Proportion")

    axe.set_title(f"Revision rates for {EXPERIMENT_PATH} - {prompt_wording}", pad=60)
    axe.set_xlim(xticks[0] - xtick_offset * 4, xticks[-1] + xtick_offset * 4)
    axe.set_ylim(0, 1)

    if legend:
        # legend_labels = ["Revised to alternative wording", "Revised to neutral variant", "Revised to masculine variant", "Revised to feminine variant"]
        handles, labels = axe.get_legend_handles_labels()
        handles = handles[:n_col]
        l1 = axe.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.2))
        plt.subplots_adjust(top=0.85)

    if save_figure:
        plt.savefig(output_path,
                    bbox_inches='tight',
                    dpi=700)
