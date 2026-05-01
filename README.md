# Transfer Learning Strategies for Sentiment Analysis

Comparing **8 ways to train a model** on the Rotten Tomatoes movie review dataset, with two backbones: **RoBERTa** (encoder) and **GPT-2** (decoder).

**Main takeaway:** LoRA reaches **84.9%** accuracy by training only **0.71%** of the parameters — almost as good as full fine-tuning (87.2%) at a fraction of the cost.

---

## Results

| Model | Strategy | Trainable Params | Accuracy |
|---|---|---|---|
| RoBERTa | Zero-shot | 0% | 75.1% |
| RoBERTa | Linear Probing | 0.5% | 82.4% |
| RoBERTa | **Full Fine-tuning** | 100% | **87.2%** |
| RoBERTa | Partial FT (6 layers frozen) | 65.9% | 86.9% |
| RoBERTa | **LoRA (r=8)** | **0.71%** | **84.9%** |
| GPT-2 | Zero-shot prompting | 0% | 49.6% |
| GPT-2 | Linear Probing | ~0% | 70.1% |
| GPT-2 | Full Fine-tuning | 100% | 86.5% |

---

## Key findings

- **LoRA is the winner on efficiency**: 0.71% of parameters → 97% of full fine-tuning performance.
- **GPT-2 zero-shot fails** (49.6% — random) because it's not instruction-tuned. After fine-tuning, it catches up with RoBERTa.
- **Domain match helps a lot**: zero-shot RoBERTa already gets 75% because it was pretrained on Twitter sentiment.
- **Partial fine-tuning** (freezing half the layers) loses almost nothing vs full FT — useful when memory is tight.



## Stack

- **Models**: RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment-latest`), GPT-2
- **Dataset**: Rotten Tomatoes (8,530 train / 1,066 test)
- **Libraries**: `transformers`, `peft`, `datasets`, `evaluate`
- **Training**: AdamW, `lr=2e-5`, batch 32, 5 epochs (same for all strategies)

---

## What's next

- [ ] LoRA on GPT-2 (only encoder is covered for now)
- [ ] Sweep LoRA rank `r ∈ {2, 4, 8, 16, 32}`
- [ ] Multi-seed runs for mean ± std
- [ ] QLoRA (4-bit) for low-memory training

---

## License

MIT
