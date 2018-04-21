# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 17:08:31 2018

@author: Jesus
"""
def custom_tokenizer(document):
    import re
    import spacy
    
    # regexp used in CountVectorizer
    regexp = re.compile('(?u)\\b\\w\\w+\\b')

    # load spacy language model and save old tokenizer
    en_nlp = spacy.load('en')
    old_tokenizer = en_nlp.tokenizer
    
    # replace the tokenizer with the preceding regexp
    en_nlp.tokenizer = lambda string: old_tokenizer.tokens_from_list(regexp.findall(string))
    
#    doc_spacy = en_nlp(u""+document)
    from spacy.tokens import Doc
    words = document.split()
    spaces = [True] * len(words)
    spaces[-1] = False
    doc_spacy = Doc(en_nlp.vocab, words=words, spaces=spaces)
    return ' '.join([token.lemma_ for token in doc_spacy])