# LLM Fine-Tuning (LoRA), RAG Pipeline & Serving Optimization

This repository contains the notebooks and implementation code for adapting `Qwen2.5-0.5B-Instruct` for structured JSON extraction using LoRA, building a grounded Retrieval-Augmented Generation (RAG) pipeline, and analyzing quantization/serving performance for 7B parameter models.

## Repository Structure

Yes, keeping both notebooks as separate files in the repository is the cleanest structure:

- `A1_LORA_QLORA_FineTuning.ipynb` — Dataset formatting (ChatML), LoRA fine-tuning on `Qwen2.5-0.5B-Instruct`, base vs. fine-tuned accuracy evaluation (`0.00%` vs. `92.31%`), and adversarial failure testing.
- `A2_RAG_Assignment.ipynb` — Document chunking, embedding retrieval, and context-grounded question answering over the National Violence Against Women Survey (NVAWS) dataset.
- `Assignment_Report.pdf` — Complete technical report covering experimental results, failure case analysis, 7B VRAM calculations, and 20-user serving capacity planning.

## Key Results Summary

### 1. LoRA Fine-Tuning (`Qwen2.5-0.5B-Instruct`)
- **Base Model Exact JSON Accuracy:** `0.00%` (outputs conversational text around JSON payloads)
- **Fine-Tuned Model Exact JSON Accuracy:** `92.31%` (tested across standard booking queries and adversarial edge cases)
- **LoRA Configuration:** `r=16`, `lora_alpha=32`, `lora_dropout=0.05`, trained with `SFTTrainer` (`lr=1e-4`, `max_grad_norm=0.3`).

### 2. 7B Fine-Tuning & Serving VRAM Analysis
- **Full Fine-Tuning (7B):** ~164.0 GB VRAM (multi-GPU setup required)
- **LoRA Fine-Tuning (7B):** ~20.7 GB VRAM (fits on a single 24GB GPU)
- **QLoRA Fine-Tuning (7B):** ~9.8 GB VRAM (runs on 12GB–16GB consumer GPUs)
- **Production Serving (20 Concurrent Users on 1x 24GB GPU):** ~7.02 GB peak VRAM using 4-bit quantization (`Q4_K_M`) and `vLLM` PagedAttention.

## How to Run
1. Open either `.ipynb` notebook in **Google Colab** with a **T4 GPU** runtime enabled.
2. Install the required dependencies (`transformers`, `peft`, `trl`, `datasets`, `accelerate`, `torchao`) as specified in the first cell.
3. Run all cells sequentially to reproduce the training logs, RAG retrieval outputs, and evaluation metrics.
