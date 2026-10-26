# Multimodal Question Answering with a Custom LLaVA Architecture

A LLaVA-style vision-language model that answers multiple-choice questions about images. It connects a **LongCLIP ViT-B/32** vision encoder to the **Qwen3-0.6B** language model through a trainable projector, and compares three ways of improving it: a deeper projector, image augmentation and LoRA fine-tuning.

Built for the ECS7001P Neural Networks & NLP module at Queen Mary University of London (2026).

## Data
11,600 images, each with 5–6 multiple-choice questions (~65k questions): 56k training questions on 10k images, and 3.3k validation questions on 600 images. The dataset was provided by the module and is **not included** in this repository.

## Approach
**1. Recovering image–question mappings.** Validation images arrive without their question mapping. Each caption is matched to its most similar image using LongCLIP image and text embeddings, then linked to its questions through `caption_ids`. Alignment accuracy: **99.83%**.

**2. Building the model.**
```
image → LongCLIP ViT-B/32 (frozen) → visual tokens
      → projector (MLP) → Qwen3 embedding space
      → [visual tokens + question + options] → Qwen3-0.6B → answer
```
Answers are scored from the logits of the option letters (A/B/C/D) rather than by free-form generation, which is faster and avoids unparseable outputs.

**3. Improving performance.** One parameterised `run_experiment()` helper runs every experiment, so only one setting changes per run.

## Results (validation set)
| Experiment | Training setup | Accuracy |
|---|---|---|
| Baseline: 2-layer MLP projector | 8k samples, 500 steps, projector only | 33.39% |
| Deeper projector (LayerNorm + residual) | same as baseline | 40.82% |
| Image augmentation (flip, colour jitter, crop) | same as baseline | 42.55%* |
| LoRA on Qwen3 attention (`q_proj`, `v_proj`, r=16, α=32) | full 56k samples, 2 epochs (7,008 steps) | 70.00%* |

\* Reported in the notebook's write-up. The saved output of these two training cells is incomplete, so the figures do not appear in the cell outputs.

**Reading the LoRA result fairly:** the LoRA run also used about 7× more data and many more training steps than the other experiments, so its gain reflects the combined effect, not LoRA alone. A controlled comparison would rerun LoRA on the 8k-sample, 500-step budget.

## Engineering notes
- Trained on a shared 8 GB GPU using **gradient accumulation** (micro-batch 2 × 8 steps = effective batch 16), so the training dynamics match a true batch-16 run.
- Image augmentation is applied to training images only, never at evaluation time.

## How to run
1. Place the dataset (`images/`, `vqa_*.jsonl`, `caption_*.jsonl`) in the working directory.
2. `pip install -r requirements.txt`
3. Run the notebook top to bottom. A GPU is required. The LoRA run takes about 3 hours on an 8 GB GPU.

## Tech stack
Python · PyTorch · Hugging Face Transformers · PEFT (LoRA) · torchvision

---
**Author:** Sude Kumbasar
