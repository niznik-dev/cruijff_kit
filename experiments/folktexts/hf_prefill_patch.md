# inspect-ai HF Backend Prefill Patch

## Problem

The inspect-ai HuggingFace backend (`inspect_ai/model/_providers/hf.py`) always sets
`add_generation_prompt=True` when applying chat templates, even when the last message
in the conversation is an assistant message (i.e., a prefill). This causes the model to
start a new assistant turn rather than continuing the prefill text.

Other backends (vLLM, SGLang) handle this correctly by detecting assistant prefill and
using `continue_final_message=True` instead.

## Impact

Without this patch, the `assistant_message()` solver in inspect-ai has no effect when
using the HF backend — the model ignores the prefill and generates a fresh response.
This breaks the Folktexts paper methodology, which relies on prefilling with
"If I had to select one of the options, my answer would be " to constrain model output.

## The Patch

**File:** `inspect_ai/model/_providers/hf.py` (around line 280)

**What it does:** Before applying the chat template, check if the last message is an
assistant message. If so, set `continue_final_message=True` and `add_generation_prompt=False`.
Also convert messages to plain dicts, which HF's `continue_final_message` codepath requires.

**Diff against inspect-ai 0.3.163:**

```diff
--- hf.py.orig
+++ hf.py.patch
@@ -280,7 +280,18 @@
+        # If the last message is an assistant message (prefill), continue it
+        # instead of adding a new generation prompt (matches vLLM/SGLang behavior)
+        if hf_messages and getattr(hf_messages[-1], "role", None) == "assistant":
+            add_gen = False
+            continue_final = True
+            # HF's continue_final_message codepath expects plain dicts
+            hf_messages = [{"role": m.role, "content": m.content} for m in hf_messages]
+        else:
+            add_gen = True
+            continue_final = False
+
         prompt = self._tokenizer.apply_chat_template(
             hf_messages,
-            add_generation_prompt=True,
+            add_generation_prompt=add_gen,
+            continue_final_message=continue_final,
             tokenize=False,
```

## Applying the Patch

The patched and original files are stored alongside the installed package:

```
~/.conda/envs/cruijff/lib/python3.13/site-packages/inspect_ai/model/_providers/
├── hf.py          # currently-active version
├── hf.py.orig     # clean original from pip install
└── hf.py.patch    # our patched version
```

**Activate:** `cp hf.py.patch hf.py`

**Deactivate:** `cp hf.py.orig hf.py` (or `pip install inspect-ai==0.3.163`)

## Failure Modes Encountered During Development

1. **Pydantic `.get()` error** — `ChatMessageAssistant` is a Pydantic model, not a dict. Used `getattr()` instead.
2. **`render_jinja_template` expects dicts** — HF's `continue_final_message` codepath calls `render_jinja_template`, which iterates messages as dicts. Solution: convert Pydantic message objects to `{"role": ..., "content": ...}` dicts before passing to `apply_chat_template`.

## Status

This patch is not upstreamed. Consider filing an issue or PR against inspect-ai if
the HF backend prefill gap persists in future releases.
