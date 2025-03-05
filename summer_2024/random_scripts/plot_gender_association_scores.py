import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Colors for revision bar graph
base_colors = [
    (0/255, 121/255, 255/255, 1),    # blue
    (0/255, 223/255, 162/255, 1),    # green
    (255/255, 165/255, 0/255, 1),  # orange
    (255/255, 0/255, 96/255, 1)      # red
]

model_names_shortened = {
    'gpt-3.5-turbo': 'GPT-3.5',
    'gpt-4-turbo': 'GPT-4',
    'gpt-4o': 'GPT-4o',
    'llama-3.1-8B-Instruct': 'Llama-3.1-8B'
}

def plot_gender_association_scores(
    data_frames, 
    output_dir, 
    model_names, 
    gender_associations_df
):
    """
    Read a CSV (already loaded here as `gender_associations_df`) that contains
    sentence_format, feminine_score, and genderedness_score. Merge those score
    columns into each dataframe in `data_frames` by matching on 'sentence_format'.
    Then, generate a grouped bar plot for each dataframe:

      - 3 bar groups on the x-axis: neutral, masculine, feminine.
      - Each group has 2 bars: "revised" and "not revised" (distinct colors).
      - The bar height indicates the proportion of sentences in each category.
      - Overlay data points with y-values as the feminine_score or genderedness_score.
      - Add box plots to show the distribution of feminine and genderedness scores.
      - Save summary statistics of the feminine and genderedness scores in a CSV file.
      
    Parameters
    ----------
    data_frames : list of pd.DataFrame
        Each item is a dataframe representing a model's outputs.
    output_dir : str
        Directory path to save the plots.
    model_names : list
        Names of the models (parallel to data_frames).
    gender_associations_df : pd.DataFrame
        A dataframe that holds sentence_format and the associated
        feminine_score and genderedness_score columns of each sentence.
    """

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        print(f"The output directory {output_dir} for the genderedness bar graph doesn't exist.")
        return

    # Define distinct color palettes: one for bars, one for data points
    bar_palette = [base_colors[0], base_colors[2]]  # Blue for "not revised", Orange for "revised"
    point_palette = [base_colors[1], base_colors[3]]  # Red for "not revised", Green for "revised"

    # Remove duplicate sentence_format / feminine and genderedness scores in gender_associations_df 
    gender_associations_df = gender_associations_df.drop_duplicates(subset=['sentence_format'])

    for model_name, df in zip(model_names, data_frames):
        # Remove duplicates in df based on sentence - not sure if needed?
        df = df.drop_duplicates(subset=['sentence']) 

        # Attach feminine_score and genderedness_score to every sentence
        # Matching across dataframes will be based on the 'sentence_format' column
        df_merged = df.merge(
            gender_associations_df[["sentence_format", "feminine_score", "genderedness_score"]],
            on="sentence_format",
            how="left"
        )

        # Classify sentence based on whether the role noun variant was revised
        def classify_revision_status(row):
            if ((not row['variant_removed']) and (pd.isna(row['variant_added']))):
                return "not revised"
            return "revised"

        df_merged["revision_status"] = df_merged.apply(classify_revision_status, axis=1)

        # Group and compute proportions
        group_counts = (
            df_merged.groupby(["role_noun_gender", "revision_status"])
            .size()
            .reset_index(name="count")
        )
        # Calculate total count for each role_noun_gender group
        total_counts = df_merged["role_noun_gender"].value_counts().reset_index()
        total_counts.columns = ["role_noun_gender", "total_count"]

        # Merge total counts into group_counts
        group_counts = group_counts.merge(total_counts, on="role_noun_gender")

        # Calculate proportions based on total counts for each group
        group_counts["proportion"] = group_counts["count"] / group_counts["total_count"]

        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

        # y-axis starts at 0, ends at 1
        for ax in axes:
            ax.set_ylim(0, 1)

        # ----------------
        # Left subplot: feminine_score distribution
        # ----------------
        
        # since stripplot can also show the proportion of sentences whose role 
        # noun variants are revised vs not revised, the bar plot is optional 

        # sns.barplot(
        #     data=group_counts,
        #     x="role_noun_gender",
        #     y="proportion",
        #     hue="revision_status",
        #     dodge=True,
        #     errorbar=None,         # no confidence interval
        #     palette=bar_palette,  
        #     ax=axes[0],
        # )
        # Overlay data points for feminine_score
        sns.stripplot(
            data=df_merged,
            x="role_noun_gender",
            y="feminine_score",
            hue="revision_status",
            dodge=True,
            palette=point_palette, 
            alpha=0.6,
            edgecolor="black",      # outline for better visibility
            linewidth=0.5,
            ax=axes[0],
        )
        # Add box plot for feminine_score
        sns.boxplot(
            data=df_merged,
            x="role_noun_gender",
            y="feminine_score",
            hue="revision_status",
            dodge=True,
            palette=point_palette,
            ax=axes[0],
            showcaps=True,
            boxprops={'facecolor':'None'},
            flierprops={'marker':'x', 'color':'black', 'markersize':5, 'linestyle':'none'},  # Customize outliers
        )
        axes[0].set_title(f"{model_names_shortened[model_name]} - Feminine Score")
        axes[0].set_ylabel("Proportion (bars) / Feminine Score (points)")

        # ----------------
        # Right subplot: genderedness_score distribution
        # ----------------

        # since stripplot can also show the proportion of sentences whose role 
        # noun variants are revised vs not revised, the bar plot is optional 

        # sns.barplot(
        #     data=group_counts,
        #     x="role_noun_gender",
        #     y="proportion",
        #     hue="revision_status",
        #     dodge=True,
        #     errorbar=None,
        #     palette=bar_palette,
        #     ax=axes[1],
        # )

        sns.stripplot(
            data=df_merged,
            x="role_noun_gender",
            y="genderedness_score",
            hue="revision_status",
            dodge=True,
            palette=point_palette,
            alpha=0.6,
            edgecolor="black",
            linewidth=0.5,
            ax=axes[1],
        )
        # Add box plot for genderedness_score
        sns.boxplot(
            data=df_merged,
            x="role_noun_gender",
            y="genderedness_score",
            hue="revision_status",
            dodge=True,
            palette=point_palette,
            ax=axes[1],
            showcaps=True,
            boxprops={'facecolor':'None'},
            flierprops={'marker':'x', 'color':'black', 'markersize':5, 'linestyle':'none'},  # Customize outliers
        )
        axes[1].set_title(f"{model_names_shortened[model_name]} - Genderedness Score")
        axes[1].set_ylabel("Proportion (bars) / Genderedness Score (points)")

        for ax in axes:
            handles, labels = ax.get_legend_handles_labels()
            # We only need two labels: "not revised" and "revised"
            if "revised" in labels and "not revised" in labels:
                # Show first 2 handles/labels
                ax.legend(handles[:2], labels[:2], loc="upper right")
            else:
                ax.legend([], [], frameon=False)

        # ---- Set custom x-ticks and multi-level axis labels ----
        # Each x-axis has 3 groups (feminine, masculine, neutral), each with 2 bars
        # We want a row of gender labels, a row of "Revised" vs "Not Revised" subtitles
        # We'll remove the default x-axis ticks, then manually place them.

        for ax in axes:
            # Remove default x labels
            ax.set_xticks([0, 1, 2])  # Set the positions of the ticks
            ax.set_xticklabels([""] * 3)  # 3 is the number of variants (feminine, masculine, neutral)

            # Place gender labels at the center of each group
            # By default, Seaborn groups the bars at integer positions [0,1,2],
            # near-labeled as valid_genders in alphabetical or data order.
            # We'll place "feminine", "masculine", "neutral" near x=0,1,2. 
            # Then we place "Revised" / "Not Revised" at sub-locations for each group.
            ax.set_xlabel("")

            # Create a secondary axis for sub-labels
            secax = ax.secondary_xaxis('bottom')
            secax.spines["bottom"].set_visible(False)
            secax.set_xlabel("")
            # Hardcode the positions: 
            # bar 1 for group 0 is at ~ -0.2, bar 2 is at ~ +0.2, group 1 is at 0.8/1.2, group 2 is at 1.8/2.2, etc.
            # We'll approximate them below. 
            # "neutral" center ~ 0.0, "masculine" center ~ 1.0, "feminine" center ~ 2.0
            # We'll place "Revised" near -0.2, +0.8, +1.8, and "Not Revised" near +0.2, +1.2, +2.2
            secax.set_xticks([-0.2, 0.2, 0.8, 1.2, 1.8, 2.2])
            secax.set_xticklabels(["Not Revised","Revised","Not Revised","Revised","Not Revised","Revised"], rotation=0)

            # Another axis for the main labels (or reuse the same axis with more major ticks):
            # We'll simply use a second secondary_xaxis or manual annotation:
            ax2 = ax.secondary_xaxis('bottom')
            ax2.spines["bottom"].set_visible(False)
            ax2.set_xlabel("")
            # Put main group labels for x=0,1,2
            ax2.set_xticks([0,1,2])
            ax2.set_xticklabels(["Feminine","Masculine","Neutral"])
            ax2.tick_params(axis='x', pad=30)  # Move them down to avoid overlap

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{model_names_shortened[model_name]}_gender_score_grouped_bars.png")
        plt.savefig(save_path, dpi=300)
        plt.close(fig)

        # Calculate summary statistics for feminine_score and genderedness_score - optional 
        # summary_stats = df_merged.groupby(["role_noun_gender", "revision_status"]).agg(
        #     feminine_mean=("feminine_score", "mean"),
        #     feminine_std=("feminine_score", "std"),
        #     feminine_min=("feminine_score", "min"),
        #     feminine_q1=("feminine_score", lambda x: x.quantile(0.25)),
        #     feminine_median=("feminine_score", "median"),
        #     feminine_q3=("feminine_score", lambda x: x.quantile(0.75)),
        #     feminine_max=("feminine_score", "max"),
        #     genderedness_mean=("genderedness_score", "mean"),
        #     genderedness_std=("genderedness_score", "std"),
        #     genderedness_min=("genderedness_score", "min"),
        #     genderedness_q1=("genderedness_score", lambda x: x.quantile(0.25)),
        #     genderedness_median=("genderedness_score", "median"),
        #     genderedness_q3=("genderedness_score", lambda x: x.quantile(0.75)),
        #     genderedness_max=("genderedness_score", "max")
        # ).reset_index()

        # Save & print summary statistics to CSV 
        # summary_stats_path = os.path.join(output_dir, f"{model_names_shortened[model_name]}_gender_score_summary_stats.csv")
        # summary_stats.to_csv(summary_stats_path, index=False)

        # print(summary_stats)