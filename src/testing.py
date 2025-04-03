from STEL.similarity import Similarity, cosine_sim
from sentence_transformers import SentenceTransformer
import torch
from STEL import STEL


class SBERTSimilarity(Similarity):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = SentenceTransformer(model_name)
        self.model.to("cuda" if torch.cuda.is_available() else "mps")

    def similarities(self, sentences_1, sentences_2):
        with torch.no_grad():
            # maybe need to use like pooling here?
            sentence_emb_1 = self.model.encode(sentences_1)
            sentence_emb_2 = self.model.encode(sentences_2)
        return [
            cosine_sim(sentence_emb_1[i], sentence_emb_2[i])
            for i in range(len(sentences_1))
        ]


# STEL.eval_on_STEL(style_objects=[SBERTSimilarity("answerdotai/modernBERT-base")])
# STEL.eval_on_STEL(style_objects=[SBERTSimilarity("maxvb/glamorous-grass-190")])
# STEL.eval_on_STEL(style_objects=[SBERTSimilarity("AnnaWegmann/Style-Embedding")])
STEL.eval_on_STEL(style_objects=[SBERTSimilarity("FacebookAI/roberta-base")])
