# LITE-008I Upstream Training Audit

## Learning path

`train.py` constructs `MemoryTrainingModule`, infers encoder/decoder dimensions, constructs `MemoryModule`, then constructs `MemoryLLM`. The decoder is frozen explicitly by `self.llm.text_model.requires_grad_(False).eval()` after the memory module is converted to bfloat16 and put in training mode. The optimizer selects only parameters with `requires_grad=True`, so it receives memory-module parameters and not decoder parameters.

`MemoryLLM.forward` computes `memory_tokens = self.memory(doc_embeds, doc_padding_mask, question_embeds)`, obtains token embeddings from the decoder, and calls `inject_memory`. `inject_memory` replaces embeddings at `<|memory|>` positions using a differentiable scatter update. The resulting `inputs_embeds` are passed to the causal decoder with labels.

`MemoryCollator` renders the answer-bearing chat, records the prompt token length, and creates labels filled with `IGNORE_INDEX=-100`. Only tokens after the prompt are copied into labels; left padding and prompt positions remain ignored. Hugging Face causal-LM loss then shifts logits and labels internally, so the answer-token loss propagates through decoder activations, the injected embeddings, and the memory module.

## Serialization

`MemoryModule.save` writes `memory.pt` and `config.json`. Lightning checkpoints are filtered in `on_save_checkpoint` to keys beginning with `memory.` because the frozen decoder is reloaded from its source. This preserves the intended lightweight artifact boundary.

## Fidelity requirements

A faithful lightweight implementation must preserve: decoder parameter freezing, external embedding injection at the exact placeholder positions, answer-only label masking, causal next-token alignment, gradients through the injected embedding tensor, memory-only optimizer selection, and memory-only checkpoint serialization. Replacing answer loss with embedding MSE or candidate ranking is not equivalent to the original mechanism.

## Verified versus unverified

The source path is verified statically. Runtime full-model gradient propagation is unverified because the official model exceeds the existing resource ceiling before activations and backward storage are included.
