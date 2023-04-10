import torch
from datasets import Dataset, DatasetDict, load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
import pandas as pd



class SummarizationModel:
    def __init__(
        self,
        model_checkpoint="allenai/led-base-16384",
        csv_file="train.csv",
        max_input_length=8192,
        max_output_length=512,
        batch_size=2,
        num_train_epochs=4,
    ):
        self.model_checkpoint = model_checkpoint
        self.csv_file = csv_file
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.batch_size = batch_size
        self.num_train_epochs = num_train_epochs

        self._prepare_data()
        self._initialize_tokenizer_and_model()
        self._initialize_trainer()

    def _prepare_data(self):
        df = pd.read_csv(self.csv_file)
        df = df.iloc[:, 1:]
        hf_dataset = Dataset.from_pandas(df)

        dataset_dict = DatasetDict(
            {
                "train": hf_dataset.train_test_split(test_size=0.1)["train"],
                "validation": hf_dataset.train_test_split(test_size=0.1)["test"],
            }
        )

        self.train_dataset = dataset_dict["train"]
        self.val_dataset = dataset_dict["validation"]

    def _initialize_tokenizer_and_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)

        self.led = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_checkpoint,
            gradient_checkpointing=True,
            use_cache=False,
        )

        self.led.config.num_beams = 2
        self.led.config.max_length = 512
        self.led.config.min_length = 100
        self.led.config.length_penalty = 2.0
        self.led.config.early_stopping = True
        self.led.config.no_repeat_ngram_size = 3

    def _initialize_trainer(self):
        self.training_args = Seq2SeqTrainingArguments(
            predict_with_generate=True,
            evaluation_strategy="steps",
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            fp16=True,
            output_dir="./",
            logging_steps=5,
            eval_steps=10,
            save_steps=10,
            save_total_limit=2,
            gradient_accumulation_steps=4,
            num_train_epochs=self.num_train_epochs,
        )

        self.trainer = Seq2SeqTrainer(
            model=self.led,
            tokenizer=self.tokenizer,
            args=self.training_args,
            compute_metrics=self.compute_metrics,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
        )

    # Function to preprocess the data
    # returns a dictionary with the input_ids, attention_mask, global_attention_mask, and labels
    def process_data_to_model_inputs(self,batch):
        # tokenize the inputs and labels
        inputs = self.tokenizer(
            batch["body"],
            padding="max_length",
            truncation=True,
            max_length=self.max_input_length,
        )
        outputs = self.selftokenizer(
            batch["abstract"],
            padding="max_length",
            truncation=True,
            max_length=self.max_output_length,
        )

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask

        # create 0 global_attention_mask lists
        batch["global_attention_mask"] = len(batch["input_ids"]) * [
            [0 for _ in range(len(batch["input_ids"][0]))]
        ]

        # since above lists are references, the following line changes the 0 index for all samples
        batch["global_attention_mask"][0][0] = 1
        batch["labels"] = outputs.input_ids

        # We have to make sure that the PAD token is ignored
        batch["labels"] = [
            [-100 if token == self.tokenizer.pad_token_id else token for token in labels]
            for labels in batch["labels"]
        ]

        return batch

    def compute_metrics(self,pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = self.tokenizer.pad_token_id
        label_str = self.tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        rouge_output = self.rouge.compute(
            predictions=pred_str, references=label_str, rouge_types=["rouge2"]
        )["rouge2"].mid

        return {
            "rouge2_precision": round(rouge_output.precision, 4),
            "rouge2_recall": round(rouge_output.recall, 4),
            "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
        }

    def preprocess_datasets(self):
        self.train_dataset = self.train_dataset.map(
            self.process_data_to_model_inputs,
            batched=True,
            batch_size=self.batch_size,
            remove_columns=["body", "abstract", "file"],
        )

        self.val_dataset = self.val_dataset.map(
            self.process_data_to_model_inputs,
            batched=True,
            batch_size=self.batch_size,
            remove_columns=["body", "abstract", "file"],
        )

        self.train_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
        )
        self.val_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
        )

    def train(self):
        self.trainer.train()


if __name__ == "__main__":
    summarizer = SummarizationModel()
    summarizer.preprocess_datasets()
    summarizer.train()
