"""TODO - add description

Built on examples here: https://radimrehurek.com/gensim/models/ldamodel.html
"""

from gensim.corpora.dictionary import Dictionary
from gensim.test.utils import common_texts
from gensim.models import LdaModel


common_dictionary = Dictionary(common_texts)
common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]

lda = LdaModel(common_corpus, num_topics=10)


# Create a new corpus, made of previously unseen documents.
other_texts = [
    ['computer', 'time', 'graph'],
    ['survey', 'response', 'eps'],
    ['human', 'system', 'computer']
]
other_corpus = [common_dictionary.doc2bow(text) for text in other_texts]

# get topic probability distribution for a document
unseen_doc = other_corpus[0]
vector = lda[unseen_doc]  
# vector
# [(0, 0.025000056),
#  (1, 0.025000056),
#  (2, 0.5249726),
#  (3, 0.025000056),
#  (4, 0.02500005),
#  (5, 0.02500005),
#  (6, 0.025011875),
#  (7, 0.27500314),
#  (8, 0.025000056),
#  (9, 0.025012067)]

# Get words most strongly associated with a topic
topic_id = 0
topic_terms = lda.get_topic_terms(0)
# topic_terms
# [(9, 0.08336747),
#  (10, 0.08334456),
#  (7, 0.08333651),
#  (5, 0.08333604),
#  (11, 0.08333239),
#  (0, 0.08333012),
#  (1, 0.08333005),
#  (3, 0.08332667),
#  (8, 0.083325885),
#  (2, 0.08332581)]

topic_terms_decoded = [(common_dictionary[topic_id], score) for topic_id, score in topic_terms]
# topic_terms_decoded
# [('trees', 0.08336747),
#  ('graph', 0.08334456),
#  ('user', 0.08333651),
#  ('system', 0.08333604),
#  ('minors', 0.08333239),
#  ('computer', 0.08333012),
#  ('human', 0.08333005),
#  ('response', 0.08332667),
#  ('eps', 0.083325885),
#  ('interface', 0.08332581)]
