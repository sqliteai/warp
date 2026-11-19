"""
test_dsml_server.py — a DeepSeek-V4.1 (DSML) container through the HTTP server.

A streamed reply and the same reply asked for whole must say the same thing:
the text a client streams is the content of the final message.
"""

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from serve import dsml                                      # noqa: E402
from tests.serve.test_server import ServerTestCase          # noqa: E402

D = dsml.DSML
CALL = (f"<{D} calls>\n<{D} invoke name=\"get\">\n"
        f"<{D} parameter name=\"city\" string=\"true\">Rome</{D} parameter>\n"
        f"</{D} invoke>\n</{D} calls>")
TOOLS = [{"type": "function", "function": {
    "name": "get", "parameters": {"type": "object",
                                  "properties": {"city": {"type": "string"}}}}}]


class TestDSMLStreaming(ServerTestCase):
    engine_kwargs = {"no_markers": True,
                     "markers": {100 + i: m for i, m in enumerate(dsml.MARKERS)}}

    def streamed_content(self, **body):
        events = self.stream(**body)
        return "".join(e["choices"][0]["delta"].get("content") or ""
                       for e in events if e != "[DONE]" and e.get("choices"))

    def test_the_lead_in_to_a_tool_call_is_not_streamed(self):
        self.engine.reply = "</think>Let me check.\n\n" + CALL + dsml.EOS
        _, body = self.chat(tools=TOOLS)
        message = body["choices"][0]["message"]
        self.assertEqual(message["content"], "Let me check.")
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "get")
        self.assertEqual(self.streamed_content(tools=TOOLS), "Let me check.")

    def test_text_ending_in_newlines_is_delivered_whole(self):
        self.engine.reply = "</think>a\n\nb\n\n" + dsml.EOS
        _, body = self.chat()
        self.assertEqual(body["choices"][0]["message"]["content"], "a\n\nb\n\n")
        self.assertEqual(self.streamed_content(), "a\n\nb\n\n")

    def test_a_stop_string_cuts_text_still_held_back(self):
        self.engine.reply = "</think>abc STOP\nmore" + dsml.EOS
        _, body = self.chat(stop="STOP")
        self.assertEqual(body["choices"][0]["message"]["content"], "abc ")
        self.assertEqual(self.streamed_content(stop="STOP"), "abc ")

    def test_a_stop_string_does_not_match_the_lead_in_to_a_tool_call(self):
        # The "\n\n<" before a block is formatting, not content, so a stop
        # string cannot end the reply there and lose the call. A stop inside
        # the content still cuts it.
        self.engine.reply = "</think>Let me check.\n\n" + CALL + dsml.EOS
        for stop in ("\n", "<"):
            with self.subTest(stop=stop):
                _, body = self.chat(tools=TOOLS, stop=stop)
                choice = body["choices"][0]
                self.assertEqual(choice["finish_reason"], "tool_calls")
                self.assertEqual(choice["message"]["content"], "Let me check.")
                self.assertEqual(
                    choice["message"]["tool_calls"][0]["function"]["name"], "get")
                self.assertEqual(
                    self.streamed_content(tools=TOOLS, stop=stop), "Let me check.")

        self.engine.reply = "</think>one\ntwo\n\n" + CALL + dsml.EOS
        _, body = self.chat(tools=TOOLS, stop="\n")
        self.assertEqual(body["choices"][0]["message"]["content"], "one")
        self.assertNotIn("tool_calls", body["choices"][0]["message"])
        self.assertEqual(self.streamed_content(tools=TOOLS, stop="\n"), "one")


if __name__ == "__main__":
    unittest.main()
