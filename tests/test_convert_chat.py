#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""test_convert_chat.py — the chat.json the converter installs must be the
one whose markup the container's tokenizer actually has.

Written after a Kimi-Linear container was served and chatted with for a
week without one: `waste chat` said so and fell back to raw continuation,
which is the designed behaviour, but the obvious fix — copy the ChatML
examples/chat.json — is wrong in a way nothing reports. `<|im_start|>` is
not in Kimi's vocabulary, so it encodes as six ordinary tokens and the
model reads its own turn structure as prose. The reply still looks like a
reply. That is the failure this file exists to make loud.

So two claims, per architecture: the right template is installed verbatim,
and a template whose markers the release does not carry is *not* installed
at all. The second is what stops the map in convert.py from being a
liability as it grows past two entries.

No torch, no source weights: the stubs come from test_convert_resume.py,
which already replaces the quantizer with one that writes the bank a real
one would. Everything deciding *which* chat.json is convert.py's own.

  python3 tests/test_convert_chat.py
"""
import io
import json
import os
import shutil
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tests"))

import test_convert_resume as H          # noqa: E402  — installs the stubs

# The markup each release actually ships, read out of its
# tokenizer_config.json added_tokens_decoder. Hardcoded rather than derived
# from the templates: this is the ground truth the templates are checked
# against, so deriving it from them would check nothing.
K3_MARKERS = ["<|open|>", "<|sep|>", "<|close|>", "<|end_of_msg|>"]
KL_MARKERS = ["<|im_system|>", "<|im_user|>", "<|im_assistant|>",
              "<|im_middle|>", "<|im_end|>"]

fails = 0


def ck(cond, what):
    global fails
    print(f"  {'ok  ' if cond else 'FAIL'}  {what}")
    if not cond:
        fails += 1


def shipped(name):
    p = os.path.join(REPO, "examples", name)
    return io.open(p, encoding="utf-8").read()


def uses_only(name, markers):
    """True if every <|…|> in the template is one the release carries.

    Deliberately not the regex convert.py uses — deleting the known
    markers and looking for a leftover `<|` is the same question asked a
    different way, so a bug in one does not hide in the other.
    """
    text = shipped(name)
    for m in markers:
        text = text.replace(m, "")
    return "<|" not in text


def build(tmp, name, arch, markers, preexisting=None):
    """Convert a stub checkpoint and return the chat.json it ended up with."""
    src = H.make_src(os.path.join(tmp, name + "-src"))
    p_cfg = os.path.join(src, "config.json")
    cfg = json.load(open(p_cfg))
    if arch:
        cfg["architectures"] = [arch]
    json.dump(cfg, open(p_cfg, "w"))
    if markers is not None:
        # Only added_tokens_decoder matters here; the ids are arbitrary,
        # since the check is on the text and not on where it landed.
        json.dump({"added_tokens_decoder":
                   {str(163584 + i): {"content": m}
                    for i, m in enumerate(markers)}},
                  open(os.path.join(src, "tokenizer_config.json"), "w"))
    out = os.path.join(tmp, name + ".waste")
    os.makedirs(out)
    if preexisting is not None:
        io.open(os.path.join(out, "chat.json"), "w").write(preexisting)
    H.run(out, src)
    p = os.path.join(out, "chat.json")
    return io.open(p, encoding="utf-8").read() if os.path.exists(p) else None


def main():
    tmp = tempfile.mkdtemp(prefix="chatjson-")
    try:
        print("the shipped templates against the vocabularies they target")
        ck(uses_only("chat-k3.json", K3_MARKERS),
           "chat-k3.json uses only markup K3 carries")
        ck(uses_only("chat-kimi-linear.json", KL_MARKERS),
           "chat-kimi-linear.json uses only markup Kimi-Linear carries")

        print("each architecture gets its own")
        ck(build(tmp, "kl", "KimiLinearForCausalLM", KL_MARKERS)
           == shipped("chat-kimi-linear.json"),
           "KimiLinear installs chat-kimi-linear.json verbatim")
        ck(build(tmp, "k3", "KimiK3ForCausalLM", K3_MARKERS)
           == shipped("chat-k3.json"),
           "KimiK3 installs chat-k3.json verbatim")

        print("and never one the tokenizer cannot spell")
        ck(build(tmp, "cross", "KimiLinearForCausalLM", K3_MARKERS) is None,
           "a release without <|im_*|> gets no chat.json rather than a mute one")

        print("the cases that must not change")
        ck(build(tmp, "notok", "KimiLinearForCausalLM", None)
           == shipped("chat-kimi-linear.json"),
           "no tokenizer_config.json is no evidence, so it still installs")
        ck(build(tmp, "keep", "KimiLinearForCausalLM", KL_MARKERS,
                 preexisting='{"open": "mine"}') == '{"open": "mine"}',
           "a hand-edited chat.json outranks the shipped one")
        ck(build(tmp, "other", "LlamaForCausalLM", KL_MARKERS) is None,
           "nothing is known for another architecture, so nothing is guessed")

        print("chat_template.jinja")
        src = H.make_src(os.path.join(tmp, "kimi-jinja-src"))
        io.open(os.path.join(src, "chat_template.jinja"), "w",
                encoding="utf-8").write("KIMI-TPL")
        out = os.path.join(tmp, "kimi-jinja.waste")
        os.makedirs(out)
        H.run(out, src)
        got = io.open(os.path.join(out, "chat_template.jinja"),
                      encoding="utf-8").read()
        ck(got == "KIMI-TPL", "Kimi still copies chat_template.jinja")

        src = H.make_src(os.path.join(tmp, "qwen-jinja-src"))
        cfg = json.load(open(os.path.join(src, "config.json")))
        cfg["architectures"] = ["Qwen4ExpForConditionalGeneration"]
        cfg["model_type"] = "qwen4_exp"
        cfg["text_config"] = {
            "model_type": "qwen4_exp_text",
            "num_hidden_layers": cfg["num_hidden_layers"],
            "num_experts": cfg["num_experts"],
            "num_experts_per_tok": 2,
            "first_k_dense_replace": cfg.get("first_k_dense_replace", 1),
            "hidden_size": cfg["hidden_size"],
            "moe_intermediate_size": cfg["moe_intermediate_size"],
            "vocab_size": cfg["vocab_size"],
        }
        json.dump(cfg, open(os.path.join(src, "config.json"), "w"))
        io.open(os.path.join(src, "chat_template.jinja"), "w",
                encoding="utf-8").write("QWEN-TOOLS")
        out = os.path.join(tmp, "qwen-jinja.waste")
        os.makedirs(out)
        H.run(out, src)
        got = io.open(os.path.join(out, "chat_template.jinja"),
                      encoding="utf-8").read()
        ck(got == "QWEN-TOOLS",
           "Qwen carries its chat template like any other release")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("CHAT.JSON FAILED" if fails else "CHAT.JSON OK")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
