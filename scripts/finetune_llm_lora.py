#!/usr/bin/env python3
"""
LLM LoRA Fine-tuning + Test + Merge Script
============================================
Features:
  - FP16 LoRA fine-tuning (max accuracy)
  - รองรับ dataset หลายรูปแบบ: messages (chat), instruction (alpaca), text completion
  - Auto-detect dataset format
  - Interactive chat test เปรียบเทียบ base model vs LoRA
  - Merge LoRA weights กลับเข้า base model
  - Export merged model ในรูปแบบ safetensors

Usage:
  # Train LoRA
  python scripts/finetune_llm_lora.py --train \\
    --model AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16 \\
    --dataset your-username/your-dataset \\
    --output ./lora_output

  # Interactive chat (base vs LoRA)
  python scripts/finetune_llm_lora.py --chat \\
    --model AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16 \\
    --adapter ./lora_output

  # Merge LoRA กลับเข้า model
  python scripts/finetune_llm_lora.py --merge \\
    --model AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16 \\
    --adapter ./lora_output \\
    --output ./model-merged

  # Train + auto chat + merge (full pipeline)
  python scripts/finetune_llm_lora.py --all \\
    --model AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16 \\
    --dataset your-username/your-dataset \\
    --output ./lora_output

Recommended datasets for fixing Thai/ language issues:
  - ptangwee/chatgpt-thai-prompts
  - pitpong/thai_instruction_data
  - scb10x/typhoon-conversations
  - airesearch/WangchanX-sft-data
  - tigeralgorithm/aya_dataset_thai
"""

import argparse
import json
import os
import sys
from typing import Optional, List, Dict

# ============================================================
# ARGS
# ============================================================
parser = argparse.ArgumentParser(description="LLM LoRA Fine-tuning + Test + Merge")
parser.add_argument("--model", type=str, default="AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16",
                    help="Base model name or path")
parser.add_argument("--dataset", type=str, default=None,
                    help="Hugging Face dataset name (e.g. 'user/dataset')")
parser.add_argument("--dataset_split", type=str, default="train",
                    help="Dataset split to use (default: train)")
parser.add_argument("--dataset_text_column", type=str, default=None,
                    help="Column name for text completion datasets (auto-detect if not set)")
parser.add_argument("--dataset_max_samples", type=int, default=None,
                    help="Limit number of samples (for testing)")
parser.add_argument("--output", type=str, default="./lora_output",
                    help="Output directory for LoRA adapter / merged model")
parser.add_argument("--adapter", type=str, default=None,
                    help="Path to LoRA adapter (for --chat or --merge)")

# Training hyperparams
parser.add_argument("--lr", type=float, default=2e-4,
                    help="Learning rate (default: 2e-4)")
parser.add_argument("--epochs", type=int, default=3,
                    help="Number of training epochs (default: 3)")
parser.add_argument("--batch_size", type=int, default=1,
                    help="Per-device batch size (default: 1)")
parser.add_argument("--grad_accum", type=int, default=4,
                    help="Gradient accumulation steps (default: 4)")
parser.add_argument("--max_length", type=int, default=2048,
                    help="Max sequence length (default: 2048)")
parser.add_argument("--lora_r", type=int, default=64,
                    help="LoRA rank (default: 64, higher = more capacity)")
parser.add_argument("--lora_alpha", type=int, default=128,
                    help="LoRA alpha (default: 128)")
parser.add_argument("--lora_dropout", type=float, default=0.05,
                    help="LoRA dropout (default: 0.05)")
parser.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                    help="Target modules for LoRA (comma-separated)")

# Data filtering
parser.add_argument("--filter_refusals", action="store_true",
                    help="Remove training examples where the assistant refuses to answer (preserves uncensored behavior)")

# Modes
parser.add_argument("--train", action="store_true", help="Run training")
parser.add_argument("--chat", action="store_true", help="Chat test mode (compare base model vs LoRA)")
parser.add_argument("--merge", action="store_true", help="Merge LoRA weights back into base model")
parser.add_argument("--all", action="store_true", help="Full pipeline: train -> chat -> merge")
parser.add_argument("--merge_dtype", type=str, default="fp16",
                    choices=["fp16", "bf16", "fp32"],
                    help="Dtype for merged model (default: fp16)")

args = parser.parse_args()


# ============================================================
# IMPORTS (lazy to allow --help without deps)
# ============================================================
def _imports():
    global AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, \
        DataCollatorForSeq2Seq, PeftModel, LoraConfig, get_peft_model, \
        load_dataset, torch
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForSeq2Seq,
    )
    from peft import (
        PeftModel,
        LoraConfig,
        get_peft_model,
    )
    from datasets import load_dataset


# ============================================================
# HELPERS
# ============================================================
def find_all_linear_names(model):
    """Find all linear module names for LoRA targeting."""
    import torch
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            name = name.split(".")[-1]
            lora_module_names.add(name)
    # Remove lm_head if present
    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    return list(sorted(lora_module_names))


def auto_detect_dataset_format(dataset, sample_idx=0):
    """Try to auto-detect dataset format from structure."""
    sample = dataset[sample_idx]
    keys = list(sample.keys())

    # Check for messages/conversations format (ShareGPT style)
    if "messages" in keys or "conversations" in keys or "conversation" in keys:
        return "messages"

    # Check for instruction format (Alpaca style)
    if "instruction" in keys:
        return "instruction"

    # Check for plain text
    if "text" in keys or "content" in keys or "sentence" in keys:
        return "text"

    # Check for single string column
    for k in keys:
        if isinstance(sample[k], str) and len(sample[k]) > 50:
            return "text"

    # If we have exactly 2 columns and one looks like a response
    if len(keys) >= 2:
        potential = [k for k in keys if k.lower() in ("input", "prompt", "question", "query")]
        potential_out = [k for k in keys if k.lower() in ("output", "response", "answer", "completion", "target")]
        if potential and potential_out:
            return "instruction"

    return "text"


def format_messages_to_text(messages, tokenizer):
    """Convert messages/conversations format to text with chat template."""
    # messages format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return text


def format_instruction_to_text(example):
    """Convert instruction format to text."""
    instruction = example.get("instruction", "")
    inp = example.get("input", "")
    output = example.get("output", "")

    if inp:
        text = f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n{output}"
    else:
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    return text


# Refusal patterns (English + Thai) — messages where assistant refuses to comply
_REFUSAL_PATTERNS = [
    "i cannot", "i can't", "i'm sorry", "i apologize", "i am sorry",
    "i am unable", "i'm not able", "i will not", "i won't",
    "cannot assist", "cannot provide", "cannot fulfill",
    "against policy", "against my policy", "against ethical",
    "not appropriate", "not comfortable",
    "sorry, but", "sorry, i",
    "as an ai", "as a language model",
    "it is not appropriate", "that is not appropriate",
    "i cannot fulfill", "i cannot complete",
    "i cannot generate", "i cannot create",
    "i cannot answer", "i cannot respond",
    "i cannot help", "i'm unable to",
    "ไม่สามารถ", "ขอโทษ", "ฉันไม่สามารถ",
    "ไม่เหมาะสม", "ทำไม่ได้", "ช่วยไม่ได้",
    "ไม่สามารถตอบ", "ไม่สามารถให้",
]

def contains_refusal(text: str) -> bool:
    """Check if text contains refusal/censorship patterns."""
    text_lower = text.lower()
    for pattern in _REFUSAL_PATTERNS:
        if pattern in text_lower:
            return True
    return False

def filter_refusal_examples(dataset, format_type: str):
    """Filter out dataset examples where the assistant refuses to answer."""
    import json
    keep_indices = []
    removed_count = 0

    for i in range(len(dataset)):
        example = dataset[i]
        is_refusal = False

        if format_type == "messages":
            msg_key = None
            for k in ("messages", "conversations", "conversation"):
                if k in example:
                    msg_key = k
                    break
            if msg_key:
                raw = example[msg_key]
                messages = raw if isinstance(raw, list) else json.loads(raw) if isinstance(raw, str) else []
                for msg in messages:
                    if msg.get("role") in ("assistant", "bot", "model"):
                        if contains_refusal(msg.get("content", "")):
                            is_refusal = True
                            break
        elif format_type == "instruction":
            output = example.get("output", "")
            if contains_refusal(output):
                is_refusal = True
        else:
            for col in ("text", "content", "sentence"):
                if col in example and isinstance(example[col], str) and contains_refusal(example[col]):
                    is_refusal = True
                    break

        if is_refusal:
            removed_count += 1
        else:
            keep_indices.append(i)

    if removed_count > 0:
        print(f"  Filtered out {removed_count} refusal examples ({len(keep_indices)} kept)")
        dataset = dataset.select(keep_indices)
    else:
        print("  No refusal examples found")

    return dataset


def tokenize_function(examples, tokenizer, max_length, format_type):
    """Tokenize dataset examples based on format type."""
    texts = []
    for i in range(len(examples[list(examples.keys())[0]])):
        example = {k: examples[k][i] for k in examples.keys()}

        if format_type == "messages":
            msg_key = None
            for k in ("messages", "conversations", "conversation"):
                if k in example:
                    msg_key = k
                    break
            messages = example[msg_key]
            if isinstance(messages, str):
                messages = json.loads(messages)
            text = format_messages_to_text(messages, tokenizer)
        elif format_type == "instruction":
            text = format_instruction_to_text(example)
        else:
            text_col = None
            for k in ("text", "content", "sentence"):
                if k in example:
                    text_col = k
                    break
            if text_col is None:
                for k, v in example.items():
                    if isinstance(v, str) and len(v) > 50:
                        text_col = k
                        break
            if text_col is None:
                text_col = list(example.keys())[0]
            text = example[text_col]

        texts.append(text)

    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=False,
        max_length=max_length,
        return_tensors=None,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def load_model_with_lora_peft(model_name, adapter_path, torch_dtype="fp16"):
    """Load base model and apply LoRA adapter without merging (for comparison)."""
    _imports()
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map.get(torch_dtype, torch.float16)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    base.config.use_cache = False

    lora_model = PeftModel.from_pretrained(base, adapter_path)
    lora_model.eval()
    return base, lora_model, tokenizer


# ============================================================
# TRAIN
# ============================================================
def train():
    _imports()

    if not args.dataset:
        print("ERROR: --dataset is required for training")
        sys.exit(1)

    dtype = torch.float16

    print(f"\n{'='*60}")
    print(f"LoRA Fine-tuning (FP16)")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print(f"LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
    print(f"{'='*60}")

    # 1. Load model
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # 2. Prepare LoRA config
    target_modules = args.target_modules.split(",") if args.target_modules else find_all_linear_names(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # 3. Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split=args.dataset_split)

    if args.dataset_max_samples:
        dataset = dataset.select(range(min(args.dataset_max_samples, len(dataset))))

    # Auto-detect format
    format_type = auto_detect_dataset_format(dataset)
    print(f"Detected dataset format: {format_type}")

    # Filter refusal/censorship examples (for uncensored models)
    if args.filter_refusals:
        print("\nFiltering out assistant refusal examples...")
        dataset = filter_refusal_examples(dataset, format_type)
    else:
        print("\n⚠️  WARNING: --filter_refusals not set — dataset may contain "
              "refusal/censorship patterns")
        print("   that could teach the model to refuse answers.")
        print("   Add --filter_refusals to remove them.\n")

    # Tokenize
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, args.max_length, format_type),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    # 4. Training arguments
    output_dir = args.output
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=True,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        gradient_checkpointing=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_steps=100,
        report_to="none",
        ddp_find_unused_parameters=False,
    )

    # 5. Train
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print(f"\n{'='*60}")
    print(f"Starting training...")
    print(f"{'='*60}")
    trainer.train()

    # 6. Save adapter
    print(f"\nSaving LoRA adapter to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done!")

    return output_dir


# ============================================================
# CHAT TEST
# ============================================================
def chat():
    _imports()

    adapter_path = args.adapter or args.output
    if not adapter_path or not os.path.exists(adapter_path):
        print(f"ERROR: LoRA adapter not found at: {adapter_path}")
        print("Use --adapter or --output to specify the path")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Interactive Chat Test: Base Model vs LoRA")
    print(f"Model: {args.model}")
    print(f"LoRA: {adapter_path}")
    print(f"{'='*60}")
    print("\nLoading models (this may take a while)...")

    base_model, lora_model, tokenizer = load_model_with_lora_peft(
        args.model, adapter_path, torch_dtype="fp16"
    )

    print("\n" + "=" * 60)
    print("Chat ready! Type your messages below.")
    print("Commands: /switch - toggle base/LoRA, /clear - clear history, /quit - exit")
    print("=" * 60)

    base_history = []
    lora_history = []
    using_lora = True

    while True:
        prefix = "LoRA" if using_lora else "BASE"
        try:
            user_input = input(f"\n\033[92mYou [{prefix}]>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            break
        elif user_input == "/switch":
            using_lora = not using_lora
            print(f"Switched to: \033[93m{'LoRA' if using_lora else 'BASE (no adapter)'}\033[0m")
            continue
        elif user_input == "/clear":
            base_history = []
            lora_history = []
            print("History cleared!")
            continue

        # Run both models for comparison
        print("\n\033[94mBASE model generating...\033[0m")
        base_response = generate_response(base_model, tokenizer, user_input, base_history)
        base_history.append({"role": "user", "content": user_input})
        base_history.append({"role": "assistant", "content": base_response})

        print(f"\n\033[93mLoRA model generating...\033[0m")
        lora_response = generate_response(lora_model, tokenizer, user_input, lora_history)
        lora_history.append({"role": "user", "content": user_input})
        lora_history.append({"role": "assistant", "content": lora_response})

        print(f"\n\033[94m{'─'*50}")
        print(f"BASE MODEL response:")
        print(f"{'─'*50}\033[0m")
        print(base_response)

        print(f"\n\033[93m{'─'*50}")
        print(f"LoRA MODEL response:")
        print(f"{'─'*50}\033[0m")
        print(lora_response)


def generate_response(model, tokenizer, prompt, history=None):
    """Generate a response from the model."""
    if history:
        messages = history + [{"role": "user", "content": prompt}]
    else:
        messages = [{"role": "user", "content": prompt}]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


# ============================================================
# MERGE
# ============================================================
def merge():
    _imports()

    adapter_path = args.adapter or args.output
    output_path = args.output if args.output != "./lora_output" else args.output + "-merged"

    if not adapter_path or not os.path.exists(adapter_path):
        print(f"ERROR: LoRA adapter not found at: {adapter_path}")
        sys.exit(1)

    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    merge_dtype = dtype_map[args.merge_dtype]

    print(f"\n{'='*60}")
    print(f"Merging LoRA into Base Model")
    print(f"Base model: {args.model}")
    print(f"LoRA adapter: {adapter_path}")
    print(f"Output: {output_path}")
    print(f"Dtype: {args.merge_dtype}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=merge_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    print("Loading and merging LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True, max_shard_size="10GB")
    tokenizer.save_pretrained(output_path)

    print(f"\nDone! Merged model saved to: {output_path}")
    return output_path


# ============================================================
# MAIN
# ============================================================
def main():
    if args.all:
        print("\n>>> FULL PIPELINE: Train -> Chat -> Merge <<<\n")
        adapter = train()

        proceed = input(f"\nTraining done! LoRA saved to: {adapter}\n"
                        f"Run chat test now? (y/N): ").strip().lower()
        if proceed == "y":
            args.adapter = adapter
            chat()

            merge_confirm = input(f"\nMerge LoRA back into base model? (y/N): ").strip().lower()
            if merge_confirm == "y":
                merge()
        else:
            merge_confirm = input(f"\nMerge LoRA back into base model without testing? (y/N): ").strip().lower()
            if merge_confirm == "y":
                merge()
    else:
        if args.train:
            train()
        if args.chat:
            chat()
        if args.merge:
            merge()

    if not any([args.train, args.chat, args.merge, args.all]):
        parser.print_help()


if __name__ == "__main__":
    main()
