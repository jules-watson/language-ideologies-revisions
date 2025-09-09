"""TODO - add description

Built on examples here: https://radimrehurek.com/gensim/models/ldamodel.html
"""

from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import pandas as pd
import warnings

from common import load_csv, load_json
from constants import EXPERIMENT_PATH, MODEL_NAMES

import nltk
nltk.download('punkt')
nltk.download('stopwords')

STOP_WORDS = set(stopwords.words('english'))
N_TOPICS = 10

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from common import load_json

from matplotlib.colors import ListedColormap
from constants import EXPERIMENT_PATH, MODEL_NAMES

# Colors for revision bar graph
# base_colors = [
#     (0/255, 121/255, 255/255, 1),    # blue
#     (0/255, 223/255, 162/255, 1),    # green
#     (255/255, 165/255, 0/255, 1),  # orange
#     (255/255, 0/255, 96/255, 1)      # red
# ]

model_names_shortened = {
    'gpt-3.5-turbo': 'GPT-3.5',
    'gpt-4-turbo': 'GPT-4',
    'gpt-4o': 'GPT-4o',
    'llama-3.1-8B-Instruct': 'Llama-3.1-8B'
}


def plot_justification_topics(data_frames, output_path, prompt_wording, axe=None, legend=True, **kwargs):
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
    axe2.set_xticklabels(['neutral', 'masculine', 'feminine'])
    axe2.spines['bottom'].set_position(('outward', 8))  # Adjust the distance of the secondary axis below

    axe.set_xlabel("Role noun gender in original sentence", labelpad=10)
    axe.set_ylabel("Proportion")

    axe.set_title(f"Revision rates for {EXPERIMENT_PATH} - {prompt_wording}", pad=60)
    axe.set_xlim(xticks[0] - xtick_offset * 4, xticks[-1] + xtick_offset * 4)
    axe.set_ylim(0, 1)

    if legend:
        # legend_labels = ["Revised to alternative wording", "Revised to neutral variant", "Revised to masculine variant", "Revised to feminine variant"]
        handles, labels = axe.get_legend_handles_labels()
        handles = handles[:4]
        l1 = axe.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.2))
        plt.subplots_adjust(top=0.85)

    if save_figure:
        plt.savefig(output_path,
                    bbox_inches='tight',
                    dpi=700)


def contains_any(sentence, role_noun_variants):
    for variant in role_noun_variants:
        if variant in sentence:
            return True
    return False


def select_words_for_lda(original, revised, justification, role_noun_variants):
    """Select words for lda from the justification"""
    
    # Select sentences from the justification that contain the role noun variants
    relevant_justification_sentences = [
        sent for sent in sent_tokenize(justification)
        if contains_any(sent, role_noun_variants)]

    # Tokenize sentences
    relevant_tokens = []
    for sent in relevant_justification_sentences:
        relevant_tokens.extend(word_tokenize(sent.lower()))

    generic_words = ["phrase", "sentence", "term"]

    # Filter out stop words
    relevant_tokens = [
        word for word in relevant_tokens 
        if word.isalnum() 
        and word not in STOP_WORDS 
        and not word.isdigit()]
    
    # Filter out words that occur in the original or revised sentences
    original_revised_tokens = word_tokenize(original.lower()) + word_tokenize(revised.lower())
    relevant_tokens = [
        word for word in relevant_tokens
        if word not in original_revised_tokens
        and word not in generic_words
        and word not in role_noun_variants
    ]
    return relevant_tokens


def load_and_process_revisions(config_path, model_names):
    dirname = "/".join(config_path.split("/")[:-1])

    revisions_dfs = []
    for model_name in model_names:
        curr_revision_path = f"{dirname}/{model_name}/revision.csv"
        curr_revision_df = load_csv(curr_revision_path)
        curr_revision_df["model_name"] = model_name

        # Filter to select rows that were revised
        curr_revision_df = curr_revision_df[(curr_revision_df['variant_removed'] == True)]

        if any(pd.isna(curr_revision_df["justification"])):
            warnings.warn(f"Dropping nan justifications for model = {model_name}")
            curr_revision_df = curr_revision_df.dropna(subset=["justification"])

        # Add column corresponding to the selected words per comment
        curr_revision_df["document_words"] = [
            select_words_for_lda(
                row["sentence"], row["revision"], row["justification"], eval(row["role_noun_set"]))
            for _, row in curr_revision_df.iterrows()]

        # Add curr_revision_df to the end of revisions_df
        revisions_dfs.append(curr_revision_df)

    return revisions_dfs


def train_lda_model(revisions_dfs):
    document_words = []
    for curr_df in revisions_dfs:
        document_words.extend(list(curr_df["document_words"]))

    lda_dictionary = Dictionary(document_words)
    revisions_corpus = [lda_dictionary.doc2bow(text) for text in document_words]
    lda_model = LdaModel(
        revisions_corpus,
        random_state=0,
        num_topics=N_TOPICS, 
        # num_topics=4,
        update_every=1,
        chunksize=100,
        passes=10,
        # alpha=50/4,
        eta = 0.1,     
        per_word_topics=True
    )
    return lda_dictionary, lda_model


def store_topic_summaries(config_path, lda_dictionary, lda_model):
    topic_terms_list = []
    for i in range(lda_model.num_topics):
        topic_terms = lda_model.get_topic_terms(i, topn=30)
        topic_terms_list.append({
            "topic": i,
            "top_terms": [lda_dictionary[token_id] for token_id, _ in topic_terms]
        })
    topic_summary_df = pd.DataFrame(topic_terms_list)
    output_path = config_path.replace("config.json", "topics_summary.csv")
    topic_summary_df.to_csv(output_path)


def get_visualization_df(df, lda_dictionary, lda_model, prompt_wording):
    def unwrap_distribution(dist):
        result = [0] * N_TOPICS
        for topic_i, p_topic in dist[0]:
            result[topic_i] = p_topic
        return result
        # dist = dist[0]
        # assert [item[0] for item in dist] == list(range(len(dist)))
        # return [item[1] for item in dist]
    
    
    df_list = []
    for gender_label, gender_df in df.groupby("role_noun_gender"):
        topic_distributions = np.array([
            unwrap_distribution(lda_model[lda_dictionary.doc2bow(document_words)])
            for document_words in list(gender_df["document_words"])])
        mean_dist = topic_distributions.mean(axis=0)
        curr_row_dict = {
            "prompt_wording": prompt_wording,
            "starting_variant": gender_label
        }
        for topic_i, p_topic in enumerate(mean_dist):
            curr_row_dict[f"topic_{topic_i}"] = p_topic
        df_list.append(curr_row_dict)
    
    result_df = pd.DataFrame(df_list)
    return result_df


def visualize_topics_by_llm(config_path, revisions_dfs, lda_dictionary, lda_model):
    config = load_json(config_path)
    dirname = "/".join(config_path.split("/")[:-1])

    task_wording_dict = config['task_wording']
    for prompt_wording in task_wording_dict.keys():
        filtered_dfs = [df[df['task_wording'] == prompt_wording] for df in revisions_dfs]
        visualization_dfs = [
            get_visualization_df(df, lda_dictionary, lda_model, prompt_wording) 
            for df in filtered_dfs]
        
        # revision rates plot for the current prompt
        escaped_prompt_wording = prompt_wording.replace('/', ' ')        
        output_path = f'{dirname}/justification_bar_graph_{escaped_prompt_wording}.png'
        plot_justification_topics(
            data_frames=visualization_dfs,
            output_path=output_path,
            prompt_wording=prompt_wording)


def main(config_path):
    revisions_dfs = load_and_process_revisions(config_path, MODEL_NAMES)
    lda_dictionary, lda_model = train_lda_model(revisions_dfs)
    store_topic_summaries(config_path, lda_dictionary, lda_model)
    visualize_topics_by_llm(config_path, revisions_dfs, lda_dictionary, lda_model)


if __name__ == "__main__":
    main(f"{EXPERIMENT_PATH}/config.json")