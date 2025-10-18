# import argparse
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from src.utils import read_jsonl, write_jsonl, extract_final, exact_match
# import torch
# from tqdm import tqdm

# DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
# FORWARD_TMPL = open("prompts/forward.txt").read()

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--model", default=DEFAULT_MODEL)
#     ap.add_argument("--train", required=True)
#     ap.add_argument("--out", default="run/star_forward.jsonl")
#     ap.add_argument("--max_new_tokens", type=int, default=256)
#     ap.add_argument("--temperature", type=float, default=0.2)
#     ap.add_argument("--top_p", type=float, default=0.95)
#     ap.add_argument("--batch_size", type=int, default=16)
#     args = ap.parse_args()

#     tok = AutoTokenizer.from_pretrained(args.model, token=True)
#     tok.pad_token = tok.eos_token
#     tok.padding_side = "left" 
#     model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto", device_map="auto", token=True)

#     examples = list(read_jsonl(args.train))
#     kept = []
#     batch_size = args.batch_size

#     for i in tqdm(range(0, len(examples), batch_size), desc="Generating", ncols=80):
#         batch = examples[i:i+batch_size]
#         prompts = [FORWARD_TMPL.format(question=ex["question"]) for ex in batch]
#         inputs = tok(prompts, return_tensors="pt", padding=True).to(model.device)
#         out_ids = model.generate(
#             **inputs,
#             max_new_tokens=args.max_new_tokens,
#             do_sample=True,
#             temperature=args.temperature,
#             top_p=args.top_p,
#             pad_token_id=tok.eos_token_id
#         )
#         texts = tok.batch_decode(out_ids, skip_special_tokens=True)
#         for ex, text in zip(batch, texts):
#             pred = extract_final(text)
#             # after decoding per candidate 'text'
#             if exact_match_from_text(ex["answer"], text):
#                 kept.append({"id": ex["id"], "question": ex["question"], "rationale": text, "final": ex["answer"]})

#     write_jsonl(args.out, kept)
#     print(f"[star_generate] kept {len(kept)} items")

# if __name__ == "__main__":
#     main()

# src/star_generate.py
# Generate forward chain-of-thoughts and keep only examples with correct final.
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils import read_jsonl, write_jsonl, exact_match_from_text
from tqdm import tqdm

DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
FORWARD_TMPL = open("prompts/forward.txt").read()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", default="run/star_forward.jsonl")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--n_candidates", type=int, default=4)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto", device_map="auto")

    examples = list(read_jsonl(args.train))
    kept = []
    B, K = args.batch_size, args.n_candidates

    for i in tqdm(range(0, len(examples), B), desc="Generating", ncols=80):
        batch = examples[i:i+B]
        prompts = [FORWARD_TMPL.format(question=ex["question"]) for ex in batch]
        inputs = tok(prompts, return_tensors="pt", padding=True).to(model.device)

        # out_ids = model.generate(
        #     **inputs,
        #     max_new_tokens=args.max_new_tokens,
        #     do_sample=False,
        #     temperature=0.0,
        #     top_p=1.0,
        #     num_return_sequences=K,
        #     pad_token_id=tok.eos_token_id,
        # )

        out_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,              # deterministic
            temperature=0.0,
            top_p=1.0,
            num_beams=args.n_candidates,  # beams must be >= returns
            num_return_sequences=args.n_candidates,
            pad_token_id=tok.eos_token_id,
        )
        texts = tok.batch_decode(out_ids, skip_special_tokens=True)

        for j, ex in enumerate(batch):
            candidates = texts[j*K:(j+1)*K]
            for text in candidates:
                if exact_match_from_text(ex["answer"], text):
                    kept.append({
                        "id": ex["id"],
                        "question": ex["question"],
                        "rationale": text,
                        "final": ex["answer"]
                    })
                    break

    write_jsonl(args.out, kept)
    print(f"[star_generate] kept {len(kept)} items")

if __name__ == "__main__":
    main()
