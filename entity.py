from transformers import pipeline
from pydantic import BaseModel
from typing import List

token_classification = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    tokenizer="dslim/bert-base-NER"
)


class Entity(BaseModel):
    entity_type: str
    name: str
    score: float

    @classmethod
    def __get_entity_from_tokens(cls, tokens):
        if len(tokens) == 0:
            return None
        first = tokens.pop(0)
        assert type(first) is dict
        score = first['score']
        entity = first['entity']
        a, b = entity.split('-')
        out = {}
        if a == 'B':
            out['entity_type'] = b
            out['name'] = str(first['word']).replace('##', '')
            out['score'] = score
            while len(tokens) > 0 and tokens[0]['entity'] == 'I-' + b:
                next_token = tokens.pop(0)
                word = str(next_token['word'])
                if word.startswith('##'):
                    word = word.replace('##', '')
                else:
                    word = ' ' + word
                out['name'] += word
            out['name'] = out['name'].strip()
            return cls(**out)

    @classmethod
    def from_text(cls, text: str) -> List['Entity']:
        tokens = token_classification(text)
        assert type(tokens) is list
        out = []
        while len(tokens) > 0:
            entity = cls.__get_entity_from_tokens(tokens)
            if entity is not None:
                out.append(entity)
        return out
