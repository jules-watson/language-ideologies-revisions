"""Generate a TSNE plot of adjective embeddings by theme"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import manifold
import numpy as np  # Import numpy
import os  # Import os module to handle files and directories


def plot_tsne_for_themes(theme_to_embeddings, perplexity_values):
    """
    Generate t-SNE plots for each theme with specified perplexity values.
    
    Args:
        theme_to_embeddings (dict): A dictionary containing embeddings and words for each theme.
        perplexity_values (list): A list of perplexity values to try for t-SNE.
    """
    # Create a folder to store all t-SNE plots
    os.makedirs("tsne_plots", exist_ok=True)

    # Iterate over each theme
    for theme, data in theme_to_embeddings.items():
        # Create a folder for the current theme
        theme_folder = f"tsne_plots/{theme}"
        os.makedirs(theme_folder, exist_ok=True)  # Create the folder if it doesn't exist

        # Iterate over each perplexity value
        for perplexity in perplexity_values:
            all_embeddings = data["embeddings"]
            all_words = data["words"]

            if not all_embeddings:
                print(f"No embeddings to plot for theme: {theme}.")
                continue

            print(f"Number of samples: {len(all_embeddings)} for perplexity: {perplexity}")

            # Convert to numpy array
            S_points = np.array(all_embeddings)

            # Apply t-SNE with the current perplexity
            n_components = 2
            t_sne = manifold.TSNE(
                n_components=n_components,
                perplexity=perplexity,
                init="random",
                max_iter=1000,
                random_state=42,
            )
            S_t_sne = t_sne.fit_transform(S_points)

            # Create scatter plot for the current theme and perplexity
            plt.figure(figsize=(15, 10))
            plt.scatter(S_t_sne[:, 0], S_t_sne[:, 1], alpha=0.6)

            # Annotations each point with the adjective
            for i, txt in enumerate(all_words):
                plt.annotate(
                    txt,
                    (S_t_sne[i, 0], S_t_sne[i, 1]),
                    fontsize=8,
                    alpha=0.7
                )

            plt.title(f"TSNE Visualization of Adjective Embeddings for Theme: {theme} (Perplexity: {perplexity})")
            plt.xlabel("TSNE Dimension 1")
            plt.ylabel("TSNE Dimension 2")
            plt.tight_layout()
            plt.savefig(f"{theme_folder}/tsne_plot_{theme}_perplexity_{perplexity}.png", bbox_inches='tight', dpi=300)
            plt.close()

if __name__ == "__main__":
    # Load adjective embeddings
    adj_embedding_df = pd.read_csv("adj_embddings.csv")
    adj_embedding_df["embedding"] = [eval(item) for item in adj_embedding_df["embedding"]]
    adj_embedding_df = adj_embedding_df.set_index("adjective")

    # Load theme words
    themes_df = pd.read_csv("theme_words.csv")
    themes_df["name"] = [item.split("\t")[0] for item in themes_df["seed_set"]]
    themes_df = themes_df.set_index("name")
    theme_to_word_sets = {
        theme: list(set(row["theme_words"].split("\t") + row["seed_set"].split("\t")))
        for theme, row in themes_df.iterrows()
    }

    # Collect embeddings for each theme
    theme_to_embeddings = {}
    for theme, word_set in theme_to_word_sets.items():
        embeddings = []
        words = []
        for word in word_set:
            if word in adj_embedding_df.index:
                embeddings.append(adj_embedding_df.loc[word, "embedding"])
                words.append(word)
        theme_to_embeddings[theme] = {"embeddings": embeddings, "words": words}

    # Define a list of perplexity values to tune
    perplexity_values = list(range(5, 15))  # values from 5 to 14

    # Generate t-SNE plots for different themes and perplexity values
    plot_tsne_for_themes(theme_to_embeddings, perplexity_values)