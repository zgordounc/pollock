import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer
from nltk.translate import bleu_score
from nltk.translate.meteor_score import meteor_score
import sacrebleu
import pandas as pd

class Evaluator:

    r_scores = None
    b_scores = None
    m_scores = None
    p = None

    def __init__(self, model_name, source_texts, reference_summaries):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.generated_summaries = self.generate_summaries(source_texts)
        self.reference_summaries = reference_summaries

    def rouge_scores(self):
        if self.r_scores is None:    
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            self.r_scores = [scorer.score(ref, gen) for ref, gen in zip(self.reference_summaries, self.generated_summaries)]
        return self.r_scores

    def bleu_scores(self):
        if self.b_scores is None:
            self.b_scores = [bleu_score.sentence_bleu([ref.split()], gen.split()) for ref, gen in zip(self.reference_summaries, self.generated_summaries)]
        return self.b_scores

    def meteor_scores(self):
        if self.m_scores is None:
            self.m_scores = [meteor_score([ref], gen) for ref, gen in zip(self.reference_summaries, self.generated_summaries)]
        return self.m_scores

    def perplexities(self):
        if self.p is None:
            inputs_list = [self.tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True) for text in self.source_texts]
            inputs = torch.cat(inputs_list, dim=0).to(self.device)
            with torch.no_grad():
                outputs = self.model(inputs, labels=inputs)
                self.p = torch.exp(outputs.loss).tolist()
        return self.p

    def generate_summaries(self, source_texts):
        summaries = []
        for text in source_texts:
            inputs = self.tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
            inputs = inputs.to(self.device)
            summary_ids = self.model.generate(inputs)
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary)
        return summaries
    



    def get_generated_summaries(self):
        return self.generated_summaries
    
    def get_reference_summaries(self):
        return self.reference_summaries
    
    def get_source_texts(self):
        return self.source_texts
    
    def get_scores(self, rouge=False, bleu=False, meteor=False, perplexity=False):
        scores = {}
        # need to seperate rouge scores into rouge1, rouge2, rougeL 
        if rouge:
            scores["rouge"] = self.rouge_scores()
        if bleu:
            scores["bleu"] = self.bleu_scores()
        if meteor:
            scores["meteor"] = self.meteor_scores()
        if perplexity:
            scores["perplexity"] = self.perplexities()
        return scores
    
    def get_pandas_scores(self, include_texts=True, rouge=False, bleu=False, meteor=False, perplexity=False):
        scores = self.get_scores(rouge=rouge, bleu=bleu, meteor=meteor, perplexity=perplexity)
        scores_df = pd.DataFrame(scores)

        if include_texts:
            scores_df['reference'] = self.reference_summaries
            scores_df['generated'] = self.generated_summaries
            scores_df['source'] = self.source_texts

        return scores_df
        