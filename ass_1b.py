from datasets     import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer
import transformers
from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import AutoTokenizer, pipeline
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from typing import Dict, Optional, Sequence
import torch

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0]
                          for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

dataset = load_dataset("databricks/databricks-dolly-15k")
context_data = dataset['train']['context']
model_id = "google/flan-t5-small"

## Step 1. Tokenize Data
tokenizer = AutoTokenizer.from_pretrained(model_id)
model_encoder = transformers.T5EncoderModel.from_pretrained("t5-base")

inputs = _tokenize_fn(context_data, tokenizer)

print(embeddings_1)
print("stopped")












