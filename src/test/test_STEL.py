from unittest import TestCase
import STEL
from STEL.STEL import eval_on_STEL
from STEL.legacy_sim_classes import WordLengthSimilarity
from STEL.similarity import Similarity
from STEL.to_add_const import LOCAL_ANN_STEL_DIM_QUAD


class Test(TestCase):
    def test_eval_sim(self):
        eval_on_STEL(style_objects=[WordLengthSimilarity()])

    def test_eval_sbert(self):
        from STEL.similarity import SBERTSimilarity
        eval_on_STEL(style_objects=[SBERTSimilarity("AnnaWegmann/Style-Embedding")])

    def test_eval_full(self):
        # call on the unfiltered STEL data
        # STEL.STEL.eval_model(style_objects=[WordLengthSimilarity()], stel_dim_tsv=LOCAL_ANN_STEL_DIM_QUAD,
        #                      only_STEL=False)
        from STEL.similarity import SBERTSimilarity
        STEL.STEL.eval_model(style_objects=[SBERTSimilarity("FacebookAI/roberta-base")], stel_dim_tsv=LOCAL_ANN_STEL_DIM_QUAD,
                             only_STEL=False)

    def test_nsp(self):
        from transformers import BertTokenizer, BertForNextSentencePrediction
        import torch

        class BERTNSPSimilarity(Similarity):
            def __init__(self, cased=False):
                super().__init__()
                # Load pre-trained BERT model and tokenizer
                if cased:
                    self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
                    self.model = BertForNextSentencePrediction.from_pretrained('bert-base-cased')
                else:
                    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                    self.model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')

            def similarity(self, sentence_1: str, sentence_2: str) -> float:
                # Tokenize input sentences for BERT
                inputs = self.tokenizer(sentence_1, sentence_2, return_tensors='pt', truncation=True, max_length=512)

                # Get the model output for NSP (Next Sentence Prediction)
                with torch.no_grad():
                    outputs = self.model(**inputs)

                # Extract the logits for NSP (logits are of shape [batch_size, 2], where 2 corresponds to [IsNext, NotNext])
                logits = outputs.logits
                softmax_probs = torch.softmax(logits, dim=1)

                # The probability that sentence_2 follows sentence_1 is given by softmax_probs[0, 0]
                similarity_score = softmax_probs[0, 0].item()  # Convert to Python float

                return similarity_score

        # eval_on_STEL(style_objects=[BERTNSPSimilarity()])
        eval_on_STEL(style_objects=[BERTNSPSimilarity(cased=True)])
