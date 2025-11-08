import spacy

# Carga el modelo una sola vez (global)
nlp = spacy.load('es_core_news_sm', disable=['parser', 'ner'])

def spacy_tokenizer(doc):
    return [t.lemma_ for t in nlp(doc) if not t.is_stop and not t.is_punct]
