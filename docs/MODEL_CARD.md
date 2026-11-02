---
language: ar
license: cc-by-sa-4.0
library_name: transformers
pipeline_tag: text-classification
base_model: aubmindlab/bert-base-arabertv02
tags:
  - arabic
  - arabic-dialects
  - intent-classification
  - banking
  - onnx
metrics:
  - f1
---

# AraBERT fine-tuned for Arabic banking intents (ArBanking77)

Classifies one Arabic banking customer message into one of the 77 ArBanking77 intents ("card
arrival", "exchange rate", "transfer fee charged", ...). It is
[`aubmindlab/bert-base-arabertv02`](https://huggingface.co/aubmindlab/bert-base-arabertv02),
fully fine-tuned on Modern Standard Arabic and Palestinian queries and tested on five varieties.
Built with a production-style system around it: versioned data, tracked runs, a registry with
promotion gates, and a served, monitored API (source: <!-- repository link -->).

Two formats are included:

| Folder | Format | Size | Runs with |
|---|---|---|---|
| `/` | PyTorch weights (safetensors, stored in fp16; 540 MB when loaded in fp32) | 272 MB | `transformers` |
| `onnx/` | ONNX graph in fp16, the version served | 272 MB | `onnxruntime` + `tokenizers`, no PyTorch |

On all 2,397 validation texts, the ONNX fp16 model gives exactly the same labels as the PyTorch one.

## How to use

```python
import torch
from transformers import pipeline

# The weights are stored in fp16; on a CPU, load them as fp32
classify = pipeline("text-classification", model="<this repository>", top_k=3, dtype=torch.float32)
classify("وين بطاقتي الجديدة؟ طلبتها من أسبوعين")
# [[{'label': 'card arrival', 'score': 0.998}, {'label': 'declined card payment', 'score': 0.0003}, ...]]
```

Without PyTorch, the ONNX version needs only `onnxruntime` and `tokenizers`. Feed it `input_ids`,
`attention_mask` and `token_type_ids` from `tokenizer.json`, truncated to 128 tokens, and apply a
softmax to the `logits`. The intent names are in `config.json` (`id2label`).

## Evaluation

Macro-F1, averaged over the intents present in each set; the 95% interval comes from a bootstrap of
whole source questions. The validation split is the project's own: the official one leaks, because
94% of its rows have a paraphrase of the same question in the training file.

| Set | This model | TF-IDF + logistic regression baseline |
|---|---|---|
| Validation, MSA + Palestinian (ours) | **0.919** (0.900–0.932) | 0.887 |
| Test, MSA | **0.919** | 0.881 |
| Test, Palestinian | **0.899** | 0.852 |
| Test, Saudi (not seen in training) | **0.843** | 0.719 |
| Test, Moroccan (not seen in training) | **0.779** | 0.612 |
| Test, Tunisian (not seen in training; 27 intents) | **0.599** | 0.499 |

- Trained on the official split instead, the same setup scores 0.921 on the MSA test, the number
  reported by the dataset's authors.
- Three seeds differed by 0.3 points on validation, but by up to 4 points on the dialects not seen in
  training. This model is the median seed (43).

## Training

- Data: ArBanking77, MSA and Palestinian, 21,571 training texts from 8,993 source questions; the
  validation split (2,397 texts, 999 questions) shares no question with training.
- Full fine-tuning on one T4 GPU (Kaggle) for 8 minutes: learning rate 4e-5, batch size 64, 10%
  warm-up, weight decay 0.01, fp16 mixed precision, maximum length 128 tokens. Early stopping on
  validation macro-F1 with patience 2; the best epoch, 8 of 10, was kept.

## Serving it on a CPU

Measured on a 2014 laptop CPU (4 cores, no AVX2), in a Docker container limited to 3 CPUs:

| | PyTorch | ONNX fp16 |
|---|---|---|
| Container image | 2.26 GB | 0.95 GB |
| Memory when idle | 1.23 GB | 0.86 GB |
| Median latency, 4 concurrent users | 670 ms | 530 ms |

int8 quantisation shrinks the model to 135 MB, but on a CPU without VNNI it either lost accuracy or
ran slower, so it was not adopted. With ONNX Runtime, set the threads per request to at most the
CPUs the process may use; left to choose, the runtime counted all the host's cores and latency
collapsed.

## Limitations

- **The domain is localised European fintech queries**, not real conversations with Arab banks.
- **Training covers MSA and Palestinian only.** The model is markedly weaker on Moroccan and Tunisian,
  and it was not tested on Egyptian, other Gulf or Levantine dialects, or Arabizi.
- **It always picks one of the 77 intents,** even for a message about something else; there is no
  "none of these" class. Check the top score before acting on it.

## Licence and citation

The fine-tuned weights are derived from ArBanking77 and so are released under **CC BY-SA 4.0**, as
the dataset requires. The base model's card declares no licence; see
[aub-mind/arabert](https://github.com/aub-mind/arabert) before commercial use.

```bibtex
@inproceedings{jarrar-etal-2023-arbanking77,
  title     = {{ArBanking77}: Intent Detection Neural Model and a New Dataset in Modern and Dialectical Arabic},
  author    = {Jarrar, Mustafa and Birim, Ahmet and Khalilia, Mohammed and Erden, Mustafa and Ghanem, Sana},
  booktitle = {Proceedings of ArabicNLP 2023},
  pages     = {276--287},
  year      = {2023},
  url       = {https://aclanthology.org/2023.arabicnlp-1.22}
}
@inproceedings{antoun-etal-2020-arabert,
  title     = {{AraBERT}: Transformer-based Model for Arabic Language Understanding},
  author    = {Antoun, Wissam and Baly, Fady and Hajj, Hazem},
  booktitle = {Proceedings of the 4th Workshop on Open-Source Arabic Corpora and Processing Tools},
  year      = {2020},
  url       = {https://arxiv.org/abs/2003.00104}
}
```
