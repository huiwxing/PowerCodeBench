# --------------------------------------------------------------------------
# Repository copy of the pipeline module llm_backend.py, unmodified apart from
# this header. Requires the GPU serving stack (vLLM/torch) and/or provider API
# keys supplied via environment variables; archived to document the frozen
# runs. See code/README.md for the module map.
# --------------------------------------------------------------------------
"""Unified LLM inference backend.

Supported backend types:
  'hf'       : HuggingFace transformers
  'vllm'     : vLLM offline inference (recommended for local panel)
  'gpt'      : OpenAI API
  'anthropic': Anthropic Claude API
  'gemini'   : Google Gemini API

The offline vLLM path runs inside the same Slurm job as the rest of the
pipeline, so no separate model server is required.
"""

import gc
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from typing import List, Dict, Any, Optional, Union
import torch

class LLMBackend:
    """Unified inference backend.

    All backends share the same interface:
        backend = LLMBackend('vllm', config)
        backend.setup()
        outputs = backend.generate_batch(msgs)
        backend.cleanup()
    """

    SUPPORTED_BACKENDS = ('hf', 'vllm', 'gpt', 'anthropic', 'gemini')

    def __init__(self, backend_type: str, config: Dict):
        if backend_type not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported backend: '{backend_type}'. "
                f"Choose from: {self.SUPPORTED_BACKENDS}"
            )
        self.backend_type = backend_type
        self.config = config
        self._ready = False

        self._hf_model = None
        self._hf_tokenizer = None
        self._vllm_engine = None
        self._vllm_SamplingParams = None
        self._openai_client = None
        self._text_token_cache = {}
        self._last_prompt_token_counts = []
        self._last_completion_token_counts = []

        print(f"\n🔧 LLMBackend: backend='{backend_type}'")

    # ============================================================
    # Initialization
    # ============================================================

    def setup(self):
        """Load the model or initialize the API client."""
        if self._ready:
            print("  ℹ️  Backend already initialized.")
            return

        if self.backend_type == 'hf':
            self._setup_hf()
        elif self.backend_type == 'vllm':
            self._setup_vllm_offline()
        elif self.backend_type == 'gpt':
            self._setup_gpt()
        elif self.backend_type == 'anthropic':
            self._setup_anthropic()
        elif self.backend_type == 'gemini':
            self._setup_gemini()

        self._ready = True

    # ── HuggingFace ──────────────────────────────────────────────

    def _setup_hf(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        model_name = self.config['model_name']
        max_seq_len = self.config['max_seq_len']
        print(f"  📦 Loading HF model: {model_name} (max_seq_len={max_seq_len})")

        self._hf_tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side='right',
        )
        if self._hf_tokenizer.pad_token is None:
            self._hf_tokenizer.pad_token = self._hf_tokenizer.eos_token

        if self.config.get('load_in_4bit', False):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config['bnb_4bit_quant_type'],
                bnb_4bit_compute_dtype=self.config['bnb_4bit_compute_dtype'],
                bnb_4bit_use_double_quant=self.config['bnb_4bit_use_double_quant'],
            )
            self._hf_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=self.config['bnb_4bit_compute_dtype'],
            )
        elif self.config.get('load_in_8bit', False):
            self._hf_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            self._hf_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )

        self._hf_model.eval()
        print(f"  ✓ HF model loaded")

    # ── vLLM offline ─────────────────────────────────────────────

    def _setup_vllm_offline(self):
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("vLLM not installed.")

        model_name  = self.config['model_name']
        max_seq_len = self.config['max_seq_len']
        available_gpus = torch.cuda.device_count()

        # Tensor- and pipeline-parallel sizing.  Explicit values in config
        # win; otherwise fall back to required_gpus or visible GPU count.
        tp_size = self.config.get('tp_size', None)
        pp_size = self.config.get('pp_size', None) or 1

        if tp_size is None:
            if 'vllm_tensor_parallel_size' in self.config:
                tp_size = self.config['vllm_tensor_parallel_size']
            elif 'required_gpus' in self.config:
                tp_size = self.config['required_gpus'] // pp_size
            else:
                tp_size = available_gpus

        total_gpus = tp_size * pp_size
        gpu_util = self.config.get('vllm_gpu_memory_utilization', 0.90)
        quantization = self.config.get('quantization', None)
        enable_ep = self.config.get('enable_expert_parallel', False)
        extra_kwargs = {}
        if total_gpus > available_gpus:
            extra_kwargs["distributed_executor_backend"] = "ray"
            print(f"     ⚡ Multi-node: using Ray backend (tp×pp={tp_size}×{pp_size}={total_gpus} > local={available_gpus})")

        print(f"  📦 Loading vLLM engine: {model_name}")
        print(f"     Local GPU count       = {available_gpus}")
        print(f"     tensor_parallel_size   = {tp_size}")
        print(f"     pipeline_parallel_size = {pp_size}")
        print(f"     max_model_len          = {max_seq_len}")
        print(f"     quantization           = {quantization}")
        print(f"     enable_expert_parallel  = {enable_ep}")

        import os
        max_num_seqs = self.config.get("vllm_max_num_seqs", 256)
        self._vllm_engine = LLM(
            model=model_name,
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
            max_model_len=max_seq_len,
            dtype="bfloat16",
            trust_remote_code=True,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_util,
            quantization=quantization,
            allow_deprecated_quantization=True,
            max_num_seqs=max_num_seqs,
            enable_expert_parallel=enable_ep,
            **extra_kwargs,
        )
        print(f"     max_num_seqs           = {max_num_seqs}")
        self._vllm_SamplingParams = SamplingParams
        print(f"  ✓ vLLM engine ready (tp={tp_size}, pp={pp_size})")

    def _normalize_model_output(self, text: str) -> str:
        """Apply narrow model-specific cleanup for known output protocols."""
        model_name = str(self.config.get("model_name", ""))
        if model_name == "openai/gpt-oss-120b":
            return self._normalize_openai_gptoss_output(text)
        if model_name == "Qwen/Qwen3-Coder-Next":
            return self._normalize_qwen3_next_output(text)
        return text

    @staticmethod
    def _normalize_openai_gptoss_output(text: str) -> str:
        """Strip Harmony-style analysis/final channel leakage from gpt-oss."""
        if not text:
            return text
        s = text.strip()
        lowered = s.lower()

        # vLLM can expose compact Harmony channel labels in offline mode. Keep
        # the assistant final payload when present; analysis text is not code.
        final_patterns = [
            r"<\|channel\|>\s*final\s*<\|message\|>",
            r"<\|start\|>\s*assistant\s*<\|channel\|>\s*final\s*<\|message\|>",
            r"\bassistant\s*final",
            r"\bassistantfinal",
        ]
        final_matches = []
        for pattern in final_patterns:
            final_matches.extend(list(re.finditer(pattern, s, flags=re.IGNORECASE)))
        if final_matches:
            marker = max(final_matches, key=lambda m: m.start())
            s = s[marker.end():].strip()

        # If only an analysis channel leaked and it contains fenced code, prefer
        # the last code block. That is usually the final answer after reasoning.
        if not final_matches and lowered.startswith("analysis"):
            blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)(?:```|$)", s, flags=re.DOTALL | re.IGNORECASE)
            if blocks:
                s = blocks[-1].strip()

        s = re.sub(r"^\s*(?:assistant)?\s*final\s*", "", s, flags=re.IGNORECASE)
        return s.strip()

    @staticmethod
    def _normalize_qwen3_next_output(text: str) -> str:
        """Keep Qwen3-Next cleanup to output formatting, not API repair."""
        if not text:
            return text
        s = text.strip()

        # If Qwen emits extra prose after a complete code fence, keep the first
        # complete fenced block. clean_generated_code will extract its contents.
        match = re.search(r"```(?:python|py)?\s*\n(.*?)```", s, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return f"```python\n{match.group(1).strip()}\n```"

        # When generation runs to max tokens inside an unterminated fence, the
        # benchmark answer is normally complete once the final print appears.
        lines = s.splitlines()
        if s.startswith("```") and len(lines) > 120:
            last_print = None
            for idx, line in enumerate(lines):
                if re.match(r"\s*print\s*\(", line):
                    last_print = idx
            if last_print is not None:
                return "\n".join(lines[:last_print + 1]).strip()

        return s

    # ── OpenAI GPT ───────────────────────────────────────────────

    def _setup_gpt(self):
        from openai import OpenAI
        client_kwargs = {"api_key": self.config['gpt_api_key']}
        if self.config.get("gpt_base_url"):
            client_kwargs["base_url"] = self.config["gpt_base_url"]
        self._openai_client = OpenAI(**client_kwargs)
        base_url = self.config.get("gpt_base_url") or "https://api.openai.com/v1"
        print(
            f"  ✓ OpenAI-compatible client ready, "
            f"model={self.config.get('gpt_model', 'gpt-4o-mini')}, base_url={base_url}"
        )

    # ── Anthropic Claude ────────────────────────────────────────

    def _setup_anthropic(self):
        if not self.config.get("anthropic_api_key"):
            raise RuntimeError("ANTHROPIC_API_KEY is required for Anthropic models.")
        print(
            "  ✓ Anthropic client ready, "
            f"model={self.config.get('anthropic_model', 'claude-sonnet-4-5-20250929')}"
        )

    # ── Google Gemini ───────────────────────────────────────────

    def _setup_gemini(self):
        if not self.config.get("gemini_api_key"):
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required for Gemini models.")
        print(f"  ✓ Gemini client ready, model={self.config.get('gemini_model', 'gemini-2.5-pro')}")

    # ============================================================
    # Batch generation
    # ============================================================

    def generate_batch(
        self,
        messages_list: List[List[Dict]],
        max_new_tokens: int = 1500,
        temperature: float = 0.0,
        batch_size: int = 8,
    ) -> List[str]:
        """Batch chat-completion.  Input is a list of OpenAI-style messages.

        ``batch_size`` only applies to the HF backend; vLLM and the API
        backends do their own scheduling.
        """
        if not self._ready:
            self.setup()

        self._last_prompt_token_counts = []
        self._last_completion_token_counts = []

        if self.backend_type == 'hf':
            return self._generate_hf(messages_list, max_new_tokens, temperature, batch_size)
        elif self.backend_type == 'vllm':
            return self._generate_vllm(messages_list, max_new_tokens, temperature)
        elif self.backend_type == 'gpt':
            return self._generate_gpt(messages_list, max_new_tokens, temperature)
        elif self.backend_type == 'anthropic':
            return self._generate_anthropic(messages_list, max_new_tokens, temperature)
        elif self.backend_type == 'gemini':
            return self._generate_gemini(messages_list, max_new_tokens, temperature)

    # ── Token accounting ───────────────────────────────────────

    @property
    def last_prompt_token_counts(self) -> List[Optional[int]]:
        """Exact prompt token counts from the most recent generate_batch call."""
        return list(self._last_prompt_token_counts or [])

    @property
    def last_completion_token_counts(self) -> List[Optional[int]]:
        """Exact completion token counts from the most recent generate_batch call."""
        return list(self._last_completion_token_counts or [])

    def _get_tokenizer_for_counting(self):
        if not self._ready:
            self.setup()

        if self.backend_type == 'hf':
            if self._hf_tokenizer is None:
                raise RuntimeError("HF tokenizer is not initialized.")
            return self._hf_tokenizer

        if self.backend_type == 'vllm':
            if self._vllm_engine is None:
                raise RuntimeError("vLLM engine is not initialized.")
            if hasattr(self._vllm_engine, "get_tokenizer"):
                return self._vllm_engine.get_tokenizer()
            raise RuntimeError("This vLLM version does not expose get_tokenizer().")

        if self.backend_type == 'gpt':
            try:
                import tiktoken
            except ImportError as e:
                raise RuntimeError(
                    "Exact GPT text token counting requires tiktoken; "
                    "prompt tokens are still read exactly from API usage after generation."
                ) from e
            model_name = self.config.get('gpt_model', 'gpt-4o-mini')
            try:
                return tiktoken.encoding_for_model(model_name)
            except KeyError:
                return tiktoken.get_encoding("o200k_base")

        if self.backend_type in ("anthropic", "gemini"):
            raise RuntimeError(
                f"{self.backend_type} text token counting uses provider count-token APIs, "
                "not a local tokenizer."
            )

        raise RuntimeError(f"Unsupported backend for token counting: {self.backend_type}")

    def count_text_tokens(self, text: str) -> int:
        """Count text tokens with the active model tokenizer. No chars/4 fallback."""
        if not text:
            return 0
        if self.backend_type == 'anthropic':
            return self._count_anthropic_text_tokens(text)
        if self.backend_type == 'gemini':
            return self._count_gemini_text_tokens(text)
        tokenizer = self._get_tokenizer_for_counting()
        if self.backend_type == 'gpt':
            return len(tokenizer.encode(text))
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            return len(tokenizer.encode(text))

    def count_chat_prompt_tokens(self, messages: List[Dict]) -> int:
        """Count the exact chat prompt tokens sent to HF/vLLM generation."""
        tokenizer = self._get_tokenizer_for_counting()

        if self.backend_type in ('gpt', 'anthropic', 'gemini'):
            raise RuntimeError(
                f"Exact {self.backend_type} chat prompt token counts are obtained from API usage "
                "after generation, not by pre-generation local estimation."
            )

        if self.backend_type == 'hf':
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            encoded = tokenizer(
                prompt,
                truncation=True,
                max_length=self.config['max_seq_len'],
            )
            return len(encoded["input_ids"])

        token_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        return len(token_ids)

    def count_chat_prompt_tokens_batch(
        self, messages_list: List[List[Dict]]
    ) -> List[int]:
        return [self.count_chat_prompt_tokens(msgs) for msgs in messages_list]

    # ── HF generation ────────────────────────────────────────────

    def _generate_hf(self, messages_list, max_new_tokens, temperature, batch_size):
        from tqdm import tqdm

        all_outputs = []
        prompt_counts = []
        completion_counts = []
        orig_padding = self._hf_tokenizer.padding_side
        self._hf_tokenizer.padding_side = 'left'

        try:
            for i in tqdm(
                range(0, len(messages_list), batch_size),
                desc="  🔮 HF Generating", ncols=100
            ):
                batch = messages_list[i:i + batch_size]
                prompts = [
                    self._hf_tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True
                    )
                    for msgs in batch
                ]

                inputs = self._hf_tokenizer(
                    prompts, return_tensors="pt", padding=True,
                    truncation=True,
                    max_length=self.config['max_seq_len'],
                ).to(self._hf_model.device)
                batch_prompt_counts = inputs["attention_mask"].sum(dim=1).tolist()

                with torch.no_grad():
                    outputs = self._hf_model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=(temperature > 0),
                        pad_token_id=self._hf_tokenizer.pad_token_id,
                        eos_token_id=self._hf_tokenizer.eos_token_id,
                    )

                for j, out in enumerate(outputs):
                    completion_ids = out[inputs['input_ids'][j].shape[0]:]
                    text = self._hf_tokenizer.decode(
                        completion_ids,
                        skip_special_tokens=True,
                    )
                    all_outputs.append(text.strip())
                    prompt_counts.append(int(batch_prompt_counts[j]))
                    completion_counts.append(int(completion_ids.shape[0]))

                torch.cuda.empty_cache()
                gc.collect()

        finally:
            self._hf_tokenizer.padding_side = orig_padding

        self._last_prompt_token_counts = prompt_counts
        self._last_completion_token_counts = completion_counts
        return all_outputs

    # ── vLLM generation ──────────────────────────────────────────

    def _generate_vllm(self, messages_list, max_new_tokens, temperature):
        """Submit all prompts to the engine and let vLLM's continuous
        batching schedule them; far higher GPU utilisation than the HF
        per-batch loop."""
        sampling_kwargs = {
            "temperature": temperature,
            "max_tokens": max_new_tokens,
        }
        if self.config.get("generation_seed") is not None:
            sampling_kwargs["seed"] = int(self.config.get("generation_seed", 22))
        try:
            sampling_params = self._vllm_SamplingParams(**sampling_kwargs)
        except TypeError:
            # Older vLLM versions may not expose SamplingParams.seed.
            sampling_kwargs.pop("seed", None)
            sampling_params = self._vllm_SamplingParams(**sampling_kwargs)

        print(f"\n  🚀 vLLM generating {len(messages_list)} prompts...")
        t0 = time.time()

        # vLLM's tqdm is very noisy in Slurm logs. Run in coarse chunks and
        # print one progress line per roughly 10% instead.
        outputs = []
        total_prompts = len(messages_list)
        n_chunks = min(10, max(1, total_prompts))
        chunk_size = max(1, (total_prompts + n_chunks - 1) // n_chunks)
        for start in range(0, total_prompts, chunk_size):
            end = min(start + chunk_size, total_prompts)
            outputs.extend(
                self._vllm_engine.chat(
                    messages=messages_list[start:end],
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )
            )
            done = len(outputs)
            elapsed_so_far = time.time() - t0
            rate = done / elapsed_so_far if elapsed_so_far > 0 else 0.0
            eta = (total_prompts - done) / rate if rate > 0 else 0.0
            print(
                f"  ⏳ vLLM generation: {done}/{total_prompts} "
                f"({done / total_prompts:.0%}) elapsed={elapsed_so_far:.1f}s eta={eta:.1f}s"
            )

        elapsed = time.time() - t0
        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        self._last_prompt_token_counts = [
            len(getattr(o, "prompt_token_ids", []) or [])
            if getattr(o, "prompt_token_ids", None) is not None
            else None
            for o in outputs
        ]
        if any(v is None for v in self._last_prompt_token_counts):
            try:
                self._last_prompt_token_counts = self.count_chat_prompt_tokens_batch(messages_list)
            except Exception:
                self._last_prompt_token_counts = [None for _ in outputs]
        self._last_completion_token_counts = [len(o.outputs[0].token_ids) for o in outputs]
        print(f"  ✓ {elapsed:.1f}s | {total_tokens/elapsed:.0f} tok/s | "
              f"{len(outputs)/elapsed:.1f} samples/s")

        return [self._normalize_model_output(o.outputs[0].text) for o in outputs]

    # ── Shared HTTP helpers for API backends ─────────────────────

    def _post_json(self, url: str, payload: Dict, headers: Dict = None, timeout: int = 120) -> Dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                **(headers or {}),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {e.code} from {url}: {body}") from e

    @staticmethod
    def _split_system_messages(messages: List[Dict]) -> tuple:
        system_parts = []
        chat = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_parts.append(str(content))
            else:
                chat.append({"role": role, "content": str(content)})
        return "\n\n".join(system_parts), chat

    @staticmethod
    def _anthropic_messages(messages: List[Dict]) -> tuple:
        system, chat = LLMBackend._split_system_messages(messages)
        converted = []
        for msg in chat:
            role = "assistant" if msg["role"] == "assistant" else "user"
            converted.append({"role": role, "content": msg["content"]})
        if not converted:
            converted = [{"role": "user", "content": ""}]
        return system, converted

    @staticmethod
    def _gemini_contents(messages: List[Dict]) -> tuple:
        system, chat = LLMBackend._split_system_messages(messages)
        contents = []
        for msg in chat:
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})
        if not contents:
            contents = [{"role": "user", "parts": [{"text": ""}]}]
        return system, contents

    # ── OpenAI generation ───────────────────────────────────────

    def _generate_gpt(self, messages_list, max_new_tokens, temperature):
        from tqdm import tqdm

        model_name = self.config.get('gpt_model', 'gpt-4o-mini')
        all_outputs = []
        prompt_counts = []
        completion_counts = []
        for msgs in tqdm(messages_list, desc="  🔮 GPT Generating", ncols=100):
            try:
                resp = self._openai_client.chat.completions.create(
                    model=model_name, messages=msgs,
                    temperature=temperature, max_tokens=max_new_tokens,
                )
                all_outputs.append(resp.choices[0].message.content or "")
                usage = getattr(resp, "usage", None)
                prompt_counts.append(getattr(usage, "prompt_tokens", None))
                completion_counts.append(getattr(usage, "completion_tokens", None))
            except Exception as e:
                print(f"\n  ⚠️  GPT error: {e}")
                all_outputs.append("")
                prompt_counts.append(None)
                completion_counts.append(None)
        self._last_prompt_token_counts = prompt_counts
        self._last_completion_token_counts = completion_counts
        return all_outputs

    # ── Anthropic generation ─────────────────────────────────────

    def _generate_anthropic(self, messages_list, max_new_tokens, temperature):
        from tqdm import tqdm

        model_name = self.config.get("anthropic_model", "claude-sonnet-4-5-20250929")
        api_key = self.config["anthropic_api_key"]
        headers = {
            "x-api-key": api_key,
            "anthropic-version": self.config.get("anthropic_version", "2023-06-01"),
        }
        all_outputs = []
        prompt_counts = []
        completion_counts = []
        for msgs in tqdm(messages_list, desc="  🔮 Claude Generating", ncols=100):
            try:
                system, converted = self._anthropic_messages(msgs)
                payload = {
                    "model": model_name,
                    "messages": converted,
                    "max_tokens": max_new_tokens,
                    "temperature": temperature,
                }
                if system:
                    payload["system"] = system
                resp = self._post_json(
                    "https://api.anthropic.com/v1/messages",
                    payload,
                    headers=headers,
                )
                text_parts = [
                    part.get("text", "")
                    for part in resp.get("content", [])
                    if part.get("type") == "text"
                ]
                all_outputs.append("".join(text_parts))
                usage = resp.get("usage") or {}
                prompt_counts.append(usage.get("input_tokens"))
                completion_counts.append(usage.get("output_tokens"))
            except Exception as e:
                print(f"\n  ⚠️  Anthropic error: {e}")
                all_outputs.append("")
                prompt_counts.append(None)
                completion_counts.append(None)
        self._last_prompt_token_counts = prompt_counts
        self._last_completion_token_counts = completion_counts
        return all_outputs

    # ── Gemini generation ────────────────────────────────────────

    def _generate_gemini(self, messages_list, max_new_tokens, temperature):
        from tqdm import tqdm

        model_name = self.config.get("gemini_model", "gemini-2.5-pro")
        api_key = self.config["gemini_api_key"]
        encoded_model = urllib.parse.quote(model_name, safe="")
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{encoded_model}:generateContent?key={urllib.parse.quote(api_key, safe='')}"
        )
        all_outputs = []
        prompt_counts = []
        completion_counts = []
        for msgs in tqdm(messages_list, desc="  🔮 Gemini Generating", ncols=100):
            try:
                system, contents = self._gemini_contents(msgs)
                payload = {
                    "contents": contents,
                    "generationConfig": {
                        "temperature": temperature,
                        "maxOutputTokens": max_new_tokens,
                    },
                }
                if system:
                    payload["systemInstruction"] = {"parts": [{"text": system}]}
                resp = self._post_json(url, payload)
                candidates = resp.get("candidates") or []
                parts = []
                if candidates:
                    for part in candidates[0].get("content", {}).get("parts", []):
                        if "text" in part:
                            parts.append(part["text"])
                all_outputs.append("".join(parts))
                usage = resp.get("usageMetadata") or {}
                prompt_counts.append(usage.get("promptTokenCount"))
                completion = (usage.get("candidatesTokenCount") or 0) + (
                    usage.get("thoughtsTokenCount") or 0
                )
                completion_counts.append(completion or None)
            except Exception as e:
                print(f"\n  ⚠️  Gemini error: {e}")
                all_outputs.append("")
                prompt_counts.append(None)
                completion_counts.append(None)
        self._last_prompt_token_counts = prompt_counts
        self._last_completion_token_counts = completion_counts
        return all_outputs

    # ── API count-token helpers ─────────────────────────────────

    def _count_anthropic_text_tokens(self, text: str) -> int:
        key = ("anthropic", self.config.get("anthropic_model"), text)
        if key in self._text_token_cache:
            return self._text_token_cache[key]
        model_name = self.config.get("anthropic_model", "claude-sonnet-4-5-20250929")
        headers = {
            "x-api-key": self.config["anthropic_api_key"],
            "anthropic-version": self.config.get("anthropic_version", "2023-06-01"),
        }
        resp = self._post_json(
            "https://api.anthropic.com/v1/messages/count_tokens",
            {"model": model_name, "messages": [{"role": "user", "content": text}]},
            headers=headers,
        )
        value = int(resp.get("input_tokens", 0))
        self._text_token_cache[key] = value
        return value

    def _count_gemini_text_tokens(self, text: str) -> int:
        key = ("gemini", self.config.get("gemini_model"), text)
        if key in self._text_token_cache:
            return self._text_token_cache[key]
        model_name = self.config.get("gemini_model", "gemini-2.5-pro")
        api_key = self.config["gemini_api_key"]
        encoded_model = urllib.parse.quote(model_name, safe="")
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{encoded_model}:countTokens?key={urllib.parse.quote(api_key, safe='')}"
        )
        resp = self._post_json(
            url,
            {"contents": [{"role": "user", "parts": [{"text": text}]}]},
        )
        value = int(resp.get("totalTokens", 0))
        self._text_token_cache[key] = value
        return value

    # ============================================================
    # Resource release
    # ============================================================

    def cleanup(self):
        """Release model GPU memory. vLLM needs explicit worker teardown."""
        if self.backend_type == 'vllm' and self._vllm_engine is not None:
            try:
                from vllm.distributed.parallel_state import destroy_model_parallel
                destroy_model_parallel()
            except Exception:
                pass

            del self._vllm_engine
            self._vllm_engine = None

            import gc
            gc.collect()

            import torch
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            # vLLM spawns worker subprocesses that need a moment to exit
            # before the next stage's CUDA contexts can be created cleanly.
            import time
            time.sleep(5)

            print("  ✅ vLLM engine released")

        elif self.backend_type == 'hf':
            if hasattr(self, '_hf_model') and self._hf_model is not None:
                del self._hf_model
                self._hf_model = None
            import torch, gc
            gc.collect()
            torch.cuda.empty_cache()

    @property
    def backend(self) -> str:
        return self.backend_type

    def is_ready(self) -> bool:
        return self._ready
