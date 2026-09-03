# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""
test_glmtools.py — GLM-5.3-Flash's native `<tool_call>` tool protocol.

chatfmt.py renders a container's chat.json. tools are the one thing four
prefix/suffix strings cannot express, so a container whose tokenizer carries
a native tool protocol gets its tool turns from `kimitools` (Kimi K2) or
`glmtools` (GLM's `<tool_call>` XML). This file checks the GLM half: that
the markers are discovered, that a request is rendered the way GLM's own
chat_template.jinja renders it, and that a reply the model writes is read
back into OpenAI tool_calls.

The templates under test are the ones examples/ actually ships — GLM's own
chat-glm53.json — with the fake tokenizer carrying GLM's tool markers, the
same split test_chatfmt.py makes: the machinery is under test, not the
vocabulary, and the real specials are exercised by the converter.

    python3 tests/serve/test_glmtools.py
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

from serve import glmtools                              # noqa: E402
from serve.chatfmt import (ChatFormat, ChatFormatError,   # noqa: E402
                           PlainParser)
from tests.serve.fake_engine import FakeEngine            # noqa: E402

SHIPPED = REPO / "examples" / "chat-glm53.json"

# GLM-5.3-Flash's own specials, given the ids the fake tokenizer reserves.
# Markers 11-18 are the role/reasoning/image block chat.json names; 21-29 are
# the tool protocol `glmtools` recognizes, all single tokens here.
GLM_MARKERS = {
    11: "<|system|>", 12: "<|user|>", 13: "<|assistant|>",
    14: "<think>", 15: "</think>",
    16: "<|begin_of_image|>", 17: "<|image|>", 18: "<|end_of_image|>",
    21: "<tool_call>", 22: "</tool_call>",
    23: "<arg_key>", 24: "</arg_key>",
    25: "<arg_value>", 26: "</arg_value>",
    27: "<tool_response>", 28: "</tool_response>",
    29: "<|observation|>",
}

TOOLS = [{"type": "function", "function": {
    "name": "get_weather",
    "description": "Current weather for a city",
    "parameters": {"type": "object",
                   "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]

CALL = {"role": "assistant", "content": "", "tool_calls": [{
    "id": "call_1", "type": "function",
    "function": {"name": "get_weather",
                 "arguments": '{"city":"Rome","days":3}'}}]}


class Base(unittest.TestCase):
    """A container directory holding GLM's own chat.json under that name."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="glmtools-")
        self.addCleanup(shutil.rmtree, self.dir, True)

    def engine(self, chat_json=SHIPPED, *, markers=None):
        dst = os.path.join(self.dir, "chat.json")
        if isinstance(chat_json, Path):
            shutil.copyfile(chat_json, dst)
        else:
            text = (json.dumps(chat_json) if isinstance(chat_json, dict)
                    else chat_json)
            with io.open(dst, "w", encoding="utf-8") as f:
                f.write(text)
        return FakeEngine(no_markers=True, model_path=self.dir,
                          markers=dict(markers if markers is not None
                                       else GLM_MARKERS))

    def load(self, chat_json=SHIPPED, **kw):
        return ChatFormat.load(self.engine(chat_json, **kw))


class TestLoad(Base):
    def test_the_shipped_glm_chat_json_loads(self):
        fmt = self.load()
        self.assertEqual(fmt.stop_marker, "<|user|>")
        self.assertEqual(fmt.stop_id, 12)
        self.assertEqual(fmt.think_close_id, 15)
        self.assertEqual(sorted(fmt.roles), ["assistant", "system", "user"])

    def test_glm_tool_protocol_is_discovered(self):
        fmt = self.load()
        self.assertEqual(fmt.tool_protocol, "glm")
        self.assertEqual(len(fmt.tool_markers), len(glmtools.MARKERS))
        # The observation opener and the response tags are part of the
        # protocol, so they are discovered and reach the reply reader's
        # marker map just like the call grammar.
        self.assertEqual(fmt.markers[29], "<|observation|>")
        self.assertIn("<tool_call>", set(fmt.tool_markers.values()))

    def test_a_partial_glm_marker_set_is_no_protocol(self):
        """Half of this XML renders as ordinary text, so a container that
        lacks one marker gets none — the same gate kimitools keeps."""
        for drop in glmtools.MARKERS:
            partial = {k: v for k, v in GLM_MARKERS.items() if v != drop}
            fmt = self.load(markers=partial)
            self.assertEqual(
                {}, fmt.tool_markers,
                f"dropping {drop} still resolved a protocol")
            self.assertEqual(fmt.tool_protocol, "")

    def test_a_no_tool_container_refuses_tools_by_name(self):
        fmt = self.load(markers={k: v for k, v in GLM_MARKERS.items()
                                 if v not in glmtools.MARKERS})
        self.assertEqual(fmt.tool_protocol, "")
        with self.assertRaises(ChatFormatError) as cm:
            fmt.build_chat_segments([{"role": "user", "content": "hi"}],
                                    tools=TOOLS, thinking=True)
        self.assertIn("GLM's <tool_call> XML protocol", str(cm.exception))


class TestRender(Base):
    def setUp(self):
        super().setUp()
        self.fmt = self.load()

    def render(self, messages, **kw):
        # GLM's generation prompt always opens the reasoning channel, so a
        # GLM format refuses thinking=False; the tool tests render with it on.
        kw.setdefault("thinking", True)
        return "".join(seg.text for seg in
                       self.fmt.build_chat_segments(messages, **kw))

    def segs(self, messages, **kw):
        kw.setdefault("thinking", True)
        return self.fmt.build_chat_segments(messages, **kw)

    def test_tool_declaration_opens_a_system_turn(self):
        """The declaration is system text with the signatures as JSON, not
        the Kimi `tool_declare` token literal."""
        segs = self.segs([{"role": "user", "content": "hi"}], tools=TOOLS)
        # The prelude ([gMASK]<sop>) opens the conversation, then the
        # declaration's <|system|> opener.
        self.assertEqual(segs[0].text, "[gMASK]<sop>")
        self.assertEqual(segs[1].text, "<|system|>")
        self.assertTrue(segs[1].markup)
        header = "".join(s.text for s in segs[2:4])
        self.assertTrue(header.startswith("# Tools"), header)
        self.assertIn("<tools>", header)
        # The signature is caller JSON, so it is a plain segment.
        body = "".join(s.text for s in segs)
        self.assertIn('"name": "get_weather"', body)
        self.assertIn("<tool_call>{function-name}<arg_key>", body)
        # The caller's JSON must not be able to forge a control token.
        self.assertFalse(any(s.markup for s in segs if '"name"' in s.text))

    def test_a_tool_call_is_flat_xml(self):
        """No section, no id — name, then one pair per argument."""
        segs = self.segs([{"role": "user", "content": "hi"}, CALL], tools=TOOLS)
        rendered = "".join(s.text for s in segs)
        self.assertIn(
            "<tool_call>get_weather<arg_key>city</arg_key>"
            "<arg_value>Rome</arg_value><arg_key>days</arg_key>"
            "<arg_value>3</arg_value></tool_call>",
            rendered)
        # The marker tags are their own markup segments; the caller's name and
        # values are plain. The declaration's `<tool_call>{function-name}`
        # is instruction text, so restrict the marker check to exact tags.
        for s in segs:
            if s.text in ("<tool_call>", "<arg_key>", "<arg_value>",
                          "</arg_key>", "</arg_value>", "</tool_call>"):
                self.assertTrue(s.markup)
            if s.text in ("get_weather", "Rome", "days", "3"):
                self.assertFalse(s.markup)

    def test_two_calls_render_consecutively(self):
        two = {"role": "assistant", "content": "", "tool_calls": [
            {"id": "a", "type": "function",
             "function": {"name": "get_weather",
                          "arguments": {"city": "Rome"}}},
            {"id": "b", "type": "function",
             "function": {"name": "get_time", "arguments": {"tz": "UTC"}}}]}
        out = self.render([{"role": "user", "content": "hi"}, two], tools=TOOLS)
        self.assertIn(
            "<tool_call>get_weather<arg_key>city</arg_key>"
            "<arg_value>Rome</arg_value></tool_call>"
            "<tool_call>get_time<arg_key>tz</arg_key>"
            "<arg_value>UTC</arg_value></tool_call>",
            out)

    def test_tool_calls_on_a_non_assistant_turn_are_refused(self):
        with self.assertRaises(ChatFormatError) as cm:
            self.fmt.build_chat_segments(
                [{"role": "user", "content": "hi", "tool_calls": [
                    {"id": "a", "function": {"name": "f",
                                             "arguments": {}}}]}],
                tools=TOOLS, thinking=True)
        self.assertIn("non-assistant", str(cm.exception))

    def test_a_tool_result_is_an_observation_turn(self):
        msg = {"role": "tool", "tool_call_id": "call_1", "content": "18C"}
        out = self.render([{"role": "user", "content": "hi"}, CALL, msg],
                          tools=TOOLS)
        self.assertIn("<|observation|>", out)
        self.assertIn("<tool_response>18C</tool_response>", out)

    def test_consecutive_results_share_one_observation(self):
        """GLM groups a run of results under a single <|observation|>."""
        msgs = [{"role": "user", "content": "hi"}, CALL,
                {"role": "tool", "tool_call_id": "call_1", "content": "18C"},
                {"role": "tool", "tool_call_id": "call_2", "content": "22C"}]
        out = self.render(msgs, tools=TOOLS)
        self.assertEqual(out.count("<|observation|>"), 1)
        self.assertIn("<tool_response>18C</tool_response>"
                      "<tool_response>22C</tool_response>", out)

    def test_two_result_runs_get_two_observations(self):
        out = self.render([
            {"role": "user", "content": "hi"}, CALL,
            {"role": "tool", "tool_call_id": "call_1", "content": "18C"},
            {"role": "assistant", "content": "then"},
            {"role": "tool", "tool_call_id": "call_2", "content": "22C"},
        ], tools=TOOLS)
        self.assertEqual(out.count("<|observation|>"), 2)


class TestGlmReplyParser(unittest.TestCase):
    MARKERS = {
        1001: "<|user|>",
        21: "<tool_call>", 22: "</tool_call>",
        23: "<arg_key>", 24: "</arg_key>",
        25: "<arg_value>", 26: "</arg_value>",
    }

    def parser(self):
        return PlainParser(markers=self.MARKERS, in_think=False,
                           tool_parser=glmtools.ToolParser())

    def feed(self, parser, items):
        for token_id, piece in items:
            parser.feed_token(token_id, piece)

    def test_a_glm_tool_call_is_read_back(self):
        p = self.parser()
        self.feed(p, [
            (2000, "I'll check the weather.\n"),
            (21, "<tool_call>"),
            (2001, "get_weather"),
            (23, "<arg_key>"), (2002, "city"), (24, "</arg_key>"),
            (25, "<arg_value>"), (2003, "Rome"), (26, "</arg_value>"),
            (22, "</tool_call>"),
        ])
        self.assertEqual(p.content, "I'll check the weather.\n")
        self.assertEqual(len(p.tool_calls), 1)
        call = p.tool_calls[0]
        self.assertEqual(call.name, "get_weather")
        self.assertEqual(call.arguments, {"city": "Rome"})
        msg = p.openai_message()
        self.assertEqual(msg["role"], "assistant")
        self.assertEqual(
            msg["tool_calls"][0]["function"]["arguments"],
            '{"city": "Rome"}')

    def test_a_non_string_value_is_read_back_as_json(self):
        p = self.parser()
        self.feed(p, [
            (21, "<tool_call>"),
            (2001, "get_weather"),
            (23, "<arg_key>"), (2002, "days"), (24, "</arg_key>"),
            (25, "<arg_value>"), (2003, "3"), (26, "</arg_value>"),
            (22, "</tool_call>"),
        ])
        self.assertEqual(p.tool_calls[0].arguments, {"days": "3"})

    def test_two_calls_are_read_back_in_order(self):
        p = self.parser()
        self.feed(p, [
            (21, "<tool_call>"),
            (2001, "get_weather"),
            (23, "<arg_key>"), (2002, "city"), (24, "</arg_key>"),
            (25, "<arg_value>"), (2003, "Rome"), (26, "</arg_value>"),
            (22, "</tool_call>"),
            (21, "<tool_call>"),
            (2004, "get_time"),
            (23, "<arg_key>"), (2005, "tz"), (24, "</arg_key>"),
            (25, "<arg_value>"), (2006, "UTC"), (26, "</arg_value>"),
            (22, "</tool_call>"),
        ])
        self.assertEqual([c.name for c in p.tool_calls],
                         ["get_weather", "get_time"])
        self.assertEqual(p.tool_calls[1].arguments, {"tz": "UTC"})

    def test_markers_are_only_structure_by_id(self):
        """A marker spelled out by an ordinary token is content, not markup —
        the same rule PlainParser keeps for everything else."""
        p = self.parser()
        self.feed(p, [(1234, "<tool_call> is the tag"), (1001, "<|user|>")])
        self.assertEqual(p.tool_calls, [])
        self.assertIn("is the tag", p.content)

    def test_finish_flushes_a_trailing_value(self):
        p = self.parser()
        self.feed(p, [
            (21, "<tool_call>"),
            (2001, "get_weather"),
            (23, "<arg_key>"), (2002, "city"), (24, "</arg_key>"),
            (25, "<arg_value>"), (2003, "Rome"),
        ])  # stream ends inside <arg_value>, before </tool_call>
        p.finish()
        self.assertEqual(p.tool_calls[0].arguments, {"city": "Rome"})


if __name__ == "__main__":
    unittest.main()