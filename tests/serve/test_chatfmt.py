# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""
test_chatfmt.py — serving a container from its own chat.json.

Two things are being checked, and the second is the one that matters.

The first is ordinary: a chat.json is read, validated, rendered into
segments, and the reply is read back. The second is that everything the
format *cannot* express is refused by name rather than half-rendered —
tools, a reasoning channel, an image — and that markup the container's
tokenizer does not carry is refused at load rather than sent as prose. That
last one is the whole reason this file's validation is stricter than the
CLI's reader: `waste chat` has a person watching, and an HTTP client does
not.

The templates under test are the ones examples/ actually ships. A test that
built its own would pass while the shipped file was wrong.

    python3 tests/serve/test_chatfmt.py
"""

import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from serve import kimitools                              # noqa: E402
from serve.chatfmt import (ChatFormat, ChatFormatError,   # noqa: E402
                           PlainParser)
from tests.serve.fake_engine import FakeEngine, LINEAR_MARKERS   # noqa: E402

SHIPPED = REPO / "examples" / "chat-kimi-linear.json"
CHATML = REPO / "examples" / "chat.json"

KIMI_K2_MARKERS = {
    **LINEAR_MARKERS,
    21: "<|tool_calls_section_begin|>",
    22: "<|tool_calls_section_end|>",
    23: "<|tool_call_begin|>",
    24: "<|tool_call_argument_begin|>",
    25: "<|tool_call_end|>",
}


class Base(unittest.TestCase):
    """A container directory holding whatever chat.json the test wants."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="chatfmt-")
        self.addCleanup(shutil.rmtree, self.dir, True)

    def engine(self, chat_json=None, *, markers=None):
        """chat_json: a Path to copy, a dict or raw str to write, or None.

        Path means "the file examples/ ships"; str means "these exact
        bytes", which is how the unparseable case is written.
        """
        dst = os.path.join(self.dir, "chat.json")
        if isinstance(chat_json, Path):
            shutil.copyfile(chat_json, dst)
        elif chat_json is not None:
            text = (json.dumps(chat_json) if isinstance(chat_json, dict)
                    else chat_json)
            with io.open(dst, "w", encoding="utf-8") as f:
                f.write(text)
        return FakeEngine(no_markers=True, model_path=self.dir,
                          markers=dict(markers or LINEAR_MARKERS))

    def load(self, chat_json=None, **kw):
        return ChatFormat.load(self.engine(chat_json, **kw))

    def refuses(self, chat_json, *, contains):
        with self.assertRaises(ChatFormatError) as cm:
            self.load(chat_json)
        self.assertIn(contains, str(cm.exception))


class TestLoad(Base):
    def test_the_shipped_template_loads(self):
        fmt = self.load(SHIPPED)
        self.assertEqual(fmt.stop_marker, "<|im_end|>")
        self.assertEqual(fmt.stop_id, 15)
        self.assertEqual(fmt.markers[15], "<|im_end|>")
        self.assertEqual(sorted(fmt.roles), ["assistant", "system", "user"])

    def test_a_partial_marker_set_is_no_protocol(self):
        """Four of the five is a different protocol, not a smaller one.

        Half of this rendering encodes as ordinary text, so the model would
        read its own tool structure as prose and answer anyway. kimitools
        disqualifies the set rather than degrading it; this pins that.
        """
        for drop in KIMI_K2_MARKERS:
            if KIMI_K2_MARKERS[drop] not in kimitools.MARKERS:
                continue
            partial = {k: v for k, v in KIMI_K2_MARKERS.items() if k != drop}
            fmt = self.load(SHIPPED, markers=partial)
            self.assertEqual(
                {}, fmt.tool_markers,
                f"dropping {KIMI_K2_MARKERS[drop]} still resolved a protocol")

    def test_kimi_tool_markers_are_discovered_when_available(self):
        fmt = self.load(SHIPPED, markers=KIMI_K2_MARKERS)

        self.assertEqual(
            fmt.markers[21],
            "<|tool_calls_section_begin|>",
        )
        self.assertEqual(
            fmt.markers[22],
            "<|tool_calls_section_end|>",
        )
        self.assertEqual(
            fmt.markers[23],
            "<|tool_call_begin|>",
        )
        self.assertEqual(
            fmt.markers[24],
            "<|tool_call_argument_begin|>",
        )
        self.assertEqual(
            fmt.markers[25],
            "<|tool_call_end|>",
        )

    def test_markup_the_tokenizer_lacks_is_refused_at_load(self):
        """examples/chat.json is ChatML, and <|im_start|> is not in this
        vocabulary. Serving it would answer plausibly and wrongly."""
        self.refuses(CHATML, contains="<|im_start|>")

    def test_a_missing_file_says_so(self):
        self.refuses(None, contains="no chat.json")

    def test_unparseable_json(self):
        self.refuses("{not json", contains="cannot be read")

    def test_open_is_required(self):
        self.refuses({"user": ["<|im_user|>", "<|im_end|>"]},
                     contains='no "open"')

    def test_a_user_turn_is_required(self):
        self.refuses({"assistant": ["<|im_assistant|>", "<|im_end|>"],
                      "open": "<|im_assistant|>"},
                     contains='no "user" turn')

    def test_a_turn_that_never_ends_is_refused(self):
        """No control token in the assistant suffix and no "stop": every
        reply would run to max_tokens and report finish_reason 'length'."""
        self.refuses({"user": ["<|im_user|>", "<|im_end|>"],
                      "assistant": ["<|im_assistant|>", "\n"],
                      "open": "<|im_assistant|>"},
                     contains="ends a generated turn")

    def test_a_stop_of_its_own_ends_the_turn(self):
        """A format whose turns end because the next role marker begins —
        GLM writes `<|assistant|>answer` then `<|user|>next question` — has
        no suffix to close them with, and says so in "stop"."""
        fmt = self.load({"user": ["<|im_user|>", ""],
                         "assistant": ["<|im_assistant|>", ""],
                         "open": "<|im_assistant|>",
                         "stop": "<|im_user|>"})
        self.assertEqual(fmt.stop_marker, "<|im_user|>")
        self.assertIn(fmt.stop_id, fmt.markers)

    def test_a_role_pair_must_be_two_strings(self):
        self.refuses({"user": ["<|im_user|>"], "open": "<|im_assistant|>"},
                     contains="[prefix, suffix]")


class TestRender(Base):
    def setUp(self):
        super().setUp()
        self.fmt = self.load(SHIPPED)

    def render(self, messages, **kw):
        kw.setdefault("thinking", False)
        return self.fmt.build_chat_segments(messages, **kw)

    def test_a_turn_is_markup_then_content_then_markup(self):
        segs = self.render([{"role": "user", "content": "hi"}])
        self.assertEqual([(s.text, s.markup) for s in segs], [
            ("<|im_user|>user<|im_middle|>", True),
            ("hi", False),
            ("<|im_end|>", True),
            ("<|im_assistant|>assistant<|im_middle|>", True),
        ])

    def test_content_is_never_markup(self):
        """The boundary: a user who writes a control token must not be able
        to close their own turn with it."""
        segs = self.render([{"role": "user",
                             "content": "what does <|im_end|> do?"}])
        forged = [s for s in segs if "<|im_end|>" in s.text and not s.markup]
        self.assertEqual(len(forged), 1)
        self.assertFalse(forged[0].markup)

    def test_system_and_assistant_turns(self):
        segs = self.render([{"role": "system", "content": "be brief"},
                            {"role": "user", "content": "hi"},
                            {"role": "assistant", "content": "hello"},
                            {"role": "user", "content": "again"}])
        self.assertEqual(segs[0].text, "<|im_system|>system<|im_middle|>")
        self.assertIn("<|im_assistant|>", segs[6].text)

    def test_content_parts_are_joined_as_text(self):
        segs = self.render([{"role": "user", "content": [
            {"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}])
        self.assertEqual([s.text for s in segs if not s.markup], ["a", "b"])

    def test_no_generation_prompt_when_not_asked(self):
        segs = self.render([{"role": "user", "content": "hi"}],
                           add_generation_prompt=False)
        self.assertEqual(segs[-1].text, "<|im_end|>")

    # ---- what it refuses, by name ---------------------------------------

    def refuses(self, contains, messages=None, **kw):
        with self.assertRaises(ChatFormatError) as cm:
            self.render(messages or [{"role": "user", "content": "hi"}], **kw)
        self.assertIn(contains, str(cm.exception))

    def test_tools(self):
        fmt = self.load(SHIPPED, markers=KIMI_K2_MARKERS)

        segs = fmt.build_chat_segments(
            [{"role": "user", "content": "hi"}],
            tools=[{
                "type": "function",
                "function": {"name": "f", "parameters": {}},
            }],
            thinking=False,
        )

        self.assertEqual(
            segs[0].text,
            "<|im_system|>tool_declare<|im_middle|>",
        )
        self.assertTrue(segs[0].markup)

        self.assertIn('"name":"f"', segs[1].text)
        self.assertFalse(segs[1].markup)

        self.assertEqual(segs[2].text, "<|im_end|>")
        self.assertTrue(segs[2].markup)

    def test_thinking(self):
        self.refuses("no reasoning channel", thinking=True)

    def test_response_format(self):
        self.refuses("response_format",
                     response_format={"type": "json_object"})

    def test_a_tool_result_turn(self):
        segs = self.render([
            {"role": "tool", "content": "42", "tool_call_id": "a"}
        ])

        rendered = "".join(s.text for s in segs)

        self.assertIn("<|im_system|>tool<|im_middle|>", rendered)
        self.assertNotIn("<|im_system|>system<|im_middle|>", rendered)
        self.assertIn("## Return of a\n42", rendered)
        self.assertIn("<|im_end|>", rendered)

    def test_a_named_tool_result_turn(self):
        """The name the client sends is the name the turn opens with."""
        segs = self.render([
            {"role": "tool", "content": "42", "tool_call_id": "a",
             "name": "get_weather"}
        ])

        rendered = "".join(s.text for s in segs)

        self.assertIn("<|im_system|>get_weather<|im_middle|>", rendered)
        self.assertIn("## Return of a\n42", rendered)

    def test_an_assistant_turn_carrying_tool_calls(self):
        segs = self.render([
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "a",
                    "function": {
                        "name": "f",
                        "arguments": "{}",
                    },
                }],
            }
        ])

        rendered = "".join(s.text for s in segs)

        self.assertIn("<|tool_calls_section_begin|>", rendered)
        self.assertIn("<|tool_call_begin|>a", rendered)
        self.assertIn("<|tool_call_argument_begin|>{}", rendered)
        self.assertIn("<|tool_call_end|>", rendered)
        self.assertIn("<|tool_calls_section_end|>", rendered)

    def test_an_image_part(self):
        self.refuses("does not say how to place one", [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:,"}}]}])

    def test_a_role_the_template_does_not_describe(self):
        fmt = ChatFormat(roles={"user": ("<|im_user|>", "<|im_end|>")},
                         opening="<|im_assistant|>",
                         stop_marker="<|im_end|>", stop_id=15)
        with self.assertRaises(ChatFormatError) as cm:
            fmt.build_chat_segments([{"role": "system", "content": "x"}],
                                    thinking=False)
        self.assertIn("does not describe one", str(cm.exception))


class TestPlainParser(unittest.TestCase):
    def parser(self):
        return PlainParser(markers={15: "<|im_end|>"})

    def feed(self, p, pairs):
        return [p.feed_token(tid, piece) for tid, piece in pairs]

    def test_content_accumulates_and_deltas_are_increments(self):
        p = self.parser()
        deltas = self.feed(p, [(1001, "h"), (1002, "i")])
        self.assertEqual(p.content, "hi")
        self.assertEqual([d.content for d in deltas], ["h", "i"])
        self.assertFalse(p.finished)

    def test_the_stop_token_ends_the_turn_and_is_not_content(self):
        p = self.parser()
        self.feed(p, [(1001, "h"), (15, "<|im_end|>"), (1002, "x")])
        self.assertEqual(p.content, "h")
        self.assertTrue(p.finished)

    def test_a_token_whose_text_looks_like_the_marker_is_content(self):
        """Structure comes from the id. The model spelling out the marker in
        an answer must not end its own turn."""
        p = self.parser()
        self.feed(p, [(1234, "<|im_end|>")])
        self.assertEqual(p.content, "<|im_end|>")
        self.assertFalse(p.finished)

    def test_finish_flushes_nothing_and_is_safe(self):
        p = self.parser()
        self.feed(p, [(1001, "h")])
        self.assertEqual(p.finish().content, "")
        self.assertEqual(p.content, "h")

    def test_openai_message(self):
        p = self.parser()
        self.feed(p, [(1001, "h")])
        self.assertEqual(p.openai_message(),
                         {"role": "assistant", "content": "h"})

    def test_an_empty_reply_is_null_content(self):
        self.assertEqual(self.parser().openai_message(),
                         {"role": "assistant", "content": None})

    def test_there_are_no_channels_to_fill(self):
        p = self.parser()
        self.feed(p, [(1001, "h")])
        self.assertEqual(p.reasoning, "")
        self.assertEqual(p.tool_calls, [])


# The three things a Kimi format does not need and GLM-5.3-Flash does. Each
# is here because the format could not otherwise be written down, and each
# fails quietly when it is missing: no prelude and the model is addressed in
# a format it was not trained on; no stop and the reply runs into the next
# turn; no think pair and the model's scratch work is returned as the answer.
GLM_JSON = {
    "prelude": "<|im_system|>",
    "system": ["<|im_system|>", ""],
    "user": ["<|im_user|>", ""],
    "assistant": ["<|im_assistant|>", ""],
    "open": "<|im_assistant|>",
    "think": ["<|im_middle|>", "<|im_end|>"],
    "stop": "<|im_user|>",
    "effort": "<|im_system|>Reasoning Effort: {}",
}


class TestThinkChannel(Base):
    """A chat.json that carries a prelude, a reasoning channel and a stop.

    The markers are the fake tokenizer's rather than GLM's, because what is
    under test is the format machinery and not the vocabulary; GLM's own
    file is examples/chat-glm53.json and the converter installs it.
    """

    def fmt(self, **over):
        return self.load({**GLM_JSON, **over})

    def rendered(self, fmt, **kw):
        return "".join(seg.text for seg in fmt.build_chat_segments(
            [{"role": "user", "content": "hi"}], **kw))

    def test_the_prelude_opens_the_conversation(self):
        out = self.rendered(self.fmt(), thinking=True)
        self.assertTrue(out.startswith("<|im_system|><|im_user|>hi"), out)

    def test_the_generation_prompt_opens_the_channel(self):
        out = self.rendered(self.fmt(), thinking=True)
        self.assertTrue(out.endswith("<|im_assistant|><|im_middle|>"), out)

    def test_the_effort_is_a_turn_of_its_own(self):
        out = self.rendered(self.fmt(), thinking=True, thinking_effort="high")
        self.assertIn("Reasoning Effort: High", out)

    def test_an_effort_the_format_cannot_express_is_refused(self):
        fmt = self.fmt(effort="")
        with self.assertRaises(ChatFormatError) as cm:
            self.rendered(fmt, thinking=True, thinking_effort="high")
        self.assertIn("reasoning effort", str(cm.exception))

    def test_a_format_that_always_thinks_cannot_be_asked_not_to(self):
        """GLM's template has no path that leaves the channel closed, and
        answering with it closed puts a stray close marker in the reply."""
        with self.assertRaises(ChatFormatError) as cm:
            self.rendered(self.fmt(), thinking=False)
        self.assertIn("always opens a reasoning channel", str(cm.exception))

    def test_the_stop_is_the_next_role_marker(self):
        fmt = self.fmt()
        self.assertEqual(fmt.stop_marker, "<|im_user|>")

    def test_markup_the_tokenizer_lacks_is_refused_in_every_field(self):
        for field, value in (("prelude", "<|nope|>"),
                             ("think", ["<|nope|>", "<|im_end|>"]),
                             ("effort", "<|nope|>{}")):
            with self.assertRaises(ChatFormatError, msg=field):
                self.fmt(**{field: value})

    def test_think_must_be_a_pair(self):
        self.refuses({**GLM_JSON, "think": ["<|im_middle|>"]},
                     contains="[open, close]")

    def test_the_reply_splits_into_reasoning_and_content(self):
        fmt = self.fmt()
        p = PlainParser(markers=fmt.markers,
                        think_close_id=fmt.think_close_id, in_think=True)
        for tid, piece in [(1001, "weigh"), (1002, "ing"),
                           (fmt.think_close_id, ""), (1003, "Rome")]:
            p.feed_token(tid, piece)
        self.assertEqual(p.reasoning, "weighing")
        self.assertEqual(p.content, "Rome")
        self.assertEqual(p.openai_message(),
                         {"role": "assistant", "content": "Rome",
                          "reasoning_content": "weighing"})

    def test_the_close_marker_belongs_to_neither_channel(self):
        fmt = self.fmt()
        p = PlainParser(markers=fmt.markers,
                        think_close_id=fmt.think_close_id, in_think=True)
        d = p.feed_token(fmt.think_close_id, "<|im_end|>")
        self.assertEqual((d.reasoning, d.content), ("", ""))
        self.assertFalse(p.finished)

    def test_an_image_is_placed_when_the_format_names_a_block(self):
        """The block is markup and the caller's bytes went to the tower, so
        nothing of the image reaches the tokenizer as text."""
        fmt = self.fmt(image="<|im_middle|>")
        segs = fmt.build_chat_segments(
            [{"role": "user", "content": [
                {"type": "text", "text": "what is this"},
                {"type": "image_url", "image_url": {"url": "data:,"}}]}],
            thinking=True, image_prompts=["<|im_middle|>"])
        img = [s for s in segs if s.text == "<|im_middle|>" and s.markup]
        self.assertEqual(len(img), 2)      # the block, and the think opener
        self.assertIn("what is this", "".join(s.text for s in segs))

    def test_more_images_encoded_than_placed_is_refused(self):
        fmt = self.fmt(image="<|im_middle|>")
        with self.assertRaises(ChatFormatError) as cm:
            fmt.build_chat_segments(
                [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:,"}}]}],
                thinking=True, image_prompts=["<|im_middle|>", "<|im_middle|>"])
        self.assertIn("more images were encoded", str(cm.exception))

    def test_the_stop_still_ends_the_turn_from_inside_the_channel(self):
        """A reply that hits the stop before closing its reasoning is
        truncated, not an exception: what it managed to think is returned."""
        fmt = self.fmt()
        p = PlainParser(markers=fmt.markers,
                        think_close_id=fmt.think_close_id, in_think=True)
        p.feed_token(1001, "half a thou")
        p.feed_token(fmt.stop_id, "<|im_user|>")
        self.assertTrue(p.finished)
        self.assertEqual(p.reasoning, "half a thou")
        self.assertIsNone(p.openai_message()["content"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

# ---------------------------------------------------------------------------
# Kimi K2 native tool protocol — development tests
# ---------------------------------------------------------------------------

class TestKimiK2ToolProtocol(Base):

    def setUp(self):
        super().setUp()
        self.fmt = self.load(
            SHIPPED,
            markers=KIMI_K2_MARKERS,
        )

    def test_kimi_k2_tool_declaration(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    },
                    "required": ["city"]
                }
            }
        }]

        segs = self.fmt.build_chat_segments(
            [{"role": "user", "content": "weather in Paris?"}],
            tools=tools,
            thinking=False,
        )

        rendered = "".join(s.text for s in segs)

        self.assertIn(
            "<|im_system|>tool_declare<|im_middle|>",
            rendered,
        )
        self.assertIn('"name":"get_weather"', rendered)
        self.assertIn("<|im_end|>", rendered)

    def test_kimi_k2_assistant_tool_call_round_trip_render(self):
        messages = [
            {"role": "user", "content": "weather in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city":"Paris"}',
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "18 C",
            },
        ]

        segs = self.fmt.build_chat_segments(
            messages,
            thinking=False,
        )

        rendered = "".join(s.text for s in segs)

        self.assertIn("<|tool_calls_section_begin|>", rendered)
        self.assertIn("<|tool_call_begin|>call_1", rendered)
        self.assertIn(
            '<|tool_call_argument_begin|>{"city":"Paris"}',
            rendered,
        )
        self.assertIn("<|tool_call_end|>", rendered)
        self.assertIn("## Return of call_1", rendered)

# ---------------------------------------------------------------------------
# Kimi K2 native generated tool-call parsing
# ---------------------------------------------------------------------------

class TestKimiK2ToolParser(unittest.TestCase):

    MARKERS = {
        1001: "<|im_end|>",
        1002: "<|tool_calls_section_begin|>",
        1003: "<|tool_calls_section_end|>",
        1004: "<|tool_call_begin|>",
        1005: "<|tool_call_argument_begin|>",
        1006: "<|tool_call_end|>",
    }

    def parser(self):
        return PlainParser(markers=self.MARKERS)

    def feed(self, parser, items):
        for token_id, piece in items:
            parser.feed_token(token_id, piece)

    def test_kimi_tool_call_is_parsed(self):
        p = self.parser()

        self.feed(p, [
            (2000, "I'll check the weather."),
            (1002, "<|tool_calls_section_begin|>"),
            (1004, "<|tool_call_begin|>"),
            (2001, "functions.get_weather:0"),
            (1005, "<|tool_call_argument_begin|>"),
            (2002, '{"city": "Paris"}'),
            (1006, "<|tool_call_end|>"),
            (1003, "<|tool_calls_section_end|>"),
            (1001, "<|im_end|>"),
        ])

        self.assertEqual(
            p.content,
            "I'll check the weather.",
        )

        self.assertEqual(len(p.tool_calls), 1)

        call = p.tool_calls[0]

        self.assertEqual(call.name, "get_weather")
        self.assertEqual(call.index, 0)
        self.assertEqual(call.json_block, '{"city": "Paris"}')

        msg = p.openai_message()

        self.assertEqual(msg["role"], "assistant")
        self.assertEqual(
            msg["content"],
            "I'll check the weather.",
        )
        self.assertEqual(len(msg["tool_calls"]), 1)
        self.assertEqual(
            msg["tool_calls"][0]["function"]["name"],
            "get_weather",
        )
        self.assertEqual(
            msg["tool_calls"][0]["function"]["arguments"],
            '{"city": "Paris"}',
        )

    def test_literal_marker_text_remains_content(self):
        p = self.parser()

        self.feed(p, [
            (
                2000,
                "literal <|tool_call_begin|> text"
            ),
            (1001, "<|im_end|>"),
        ])

        self.assertEqual(
            p.content,
            "literal <|tool_call_begin|> text",
        )
        self.assertEqual(p.tool_calls, [])

    def test_kimi_tool_call_delta_reports_change(self):
        p = self.parser()

        p.feed_token(
            1002,
            "<|tool_calls_section_begin|>",
        )
        p.feed_token(
            1004,
            "<|tool_call_begin|>",
        )
        p.feed_token(
            2000,
            "functions.get_weather:0",
        )
        p.feed_token(
            1005,
            "<|tool_call_argument_begin|>",
        )

        delta = p.feed_token(
            2001,
            '{"city":"Paris"}',
        )

        self.assertIn(0, delta.tool_calls)

    def test_kimi_tool_call_without_arguments_marker(self):
        p = self.parser()

        self.feed(p, [
            (1002, "<|tool_calls_section_begin|>"),
            (1004, "<|tool_call_begin|>"),
            (2001, "functions.get_time:0"),
            (1006, "<|tool_call_end|>"),
            (1003, "<|tool_calls_section_end|>"),
            (1001, "<|im_end|>"),
        ])

        self.assertEqual(len(p.tool_calls), 1)
        call = p.tool_calls[0]
        self.assertEqual(call.name, "get_time")
        self.assertEqual(call.index, 0)
        self.assertEqual(call.json_block, "")

    def test_kimi_tool_call_stream_ended_in_header(self):
        p = self.parser()

        self.feed(p, [
            (1002, "<|tool_calls_section_begin|>"),
            (1004, "<|tool_call_begin|>"),
            (2001, "functions.get_time:0"),
        ])
        p.finish()

        self.assertEqual(len(p.tool_calls), 1)
        call = p.tool_calls[0]
        self.assertEqual(call.name, "get_time")
        self.assertEqual(call.index, 0)
        self.assertEqual(call.json_block, "")
