# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""
test_main.py — what `python3 -m serve` decides before it opens anything.

The decision under test is the one that used to be advice in a manual: a
registry of swappable containers is only allowed to start if its swap
window fits. A swap holds two contexts at once — the new container is
opened before the old one is closed, which is the rollback guarantee — and
with the default `--budget 0` each context sizes itself to up to 3/4 of
waste_usable_ram, so the window was ~1.5x what the process may use.

It is arithmetic over numbers, so it is tested as arithmetic: this file
needs no container, no libwaste and no particular machine. Everything that
does need those — loading, serving, the banner — is in test_engine.py and
test_server.py.

    python3 tests/serve/test_main.py
"""
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from serve.__main__ import (RegistryBudgetError, check_registry_budget,
                            describe_registry, human, main)
from serve.engine import EngineError

GB = 1 << 30
MODELS = {"glm53": "/fake/glm53.waste", "ds41": "/fake/ds41.waste"}

class TestHumanBytes(unittest.TestCase):
    def test_units(self):
        self.assertEqual(human(0), "0 B")
        self.assertEqual(human(1024), "1.0 KB")
        self.assertEqual(human(3 * GB), "3.0 GB")
        self.assertEqual(human(2 * (1 << 40)), "2.0 TB")


class TestUsageWithModels(unittest.TestCase):
    def test_usage_with_models_is_refused(self):
        stderr = io.StringIO()
        with patch("sys.stderr", stderr):
            rc = main(["dummy.waste", "--models", "other.waste", "--usage", "hotlist.waste"])
        self.assertEqual(rc, 2)
        self.assertIn("--usage cannot be used with --models", stderr.getvalue())

    def test_usage_with_auto_register_is_refused(self):
        """--auto-register also opts into switching, so the same refusal
        applies: a learned hotlist is specific to one container, and a
        caller that did not name one is not allowed to teach the server a
        hotlist for a container it has not opened."""
        stderr = io.StringIO()
        with patch("sys.stderr", stderr):
            rc = main(["dummy.waste", "--auto-register", "--usage", "hotlist.waste"])
        self.assertEqual(rc, 2)
        self.assertIn("--usage cannot be used with --models or --auto-register",
                      stderr.getvalue())

class TestRegistryBudget(unittest.TestCase):
    """--models is refused unless --budget shows that 2 x budget fits."""

    def refuse(self, **kwargs) -> str:
        with self.assertRaises(RegistryBudgetError) as cm:
            check_registry_budget(MODELS, **kwargs)
        return str(cm.exception)

    def test_a_registry_without_a_budget_is_refused(self):
        """0 means "the engine chooses", and what it chooses is per-context
        — which is exactly how the window got to 1.5x the machine."""
        message = self.refuse(budget=0, usable=64 * GB)
        self.assertIn("--budget", message)
        self.assertIn("48.0 GB", message)      # what 0 resolves to per ctx
        self.assertIn("64.0 GB", message)      # what the process may use
        self.assertIn("32.0 GB", message)      # the largest that fits twice

    def test_a_pair_that_does_not_fit_is_refused(self):
        message = self.refuse(budget=40 * GB, usable=64 * GB)
        self.assertIn("40.0 GB", message)
        self.assertIn("80.0 GB", message)      # what the moment would cost
        self.assertIn("32.0 GB", message)      # and what to use instead

    def test_exactly_half_fits(self):
        """2 x budget == usable is the boundary and it is a pass: the
        engine's own cap already leaves a quarter of its budget to the OS,
        so refusing this would refuse a machine with nothing else running."""
        check_registry_budget(MODELS, budget=32 * GB, usable=64 * GB)


    def test_one_byte_over_the_boundary_is_refused(self):
        self.assertIn("does not fit twice",
                      self.refuse(budget=32 * GB + 1, usable=64 * GB))

    def test_a_single_model_is_never_checked(self):
        """No registry, no swap: one context against the machine is the
        engine's own business, whatever the budget says."""
        check_registry_budget({}, budget=0, usable=0)
        check_registry_budget({}, budget=GB, usable=GB // 4)

    def test_an_unmeasurable_machine_is_not_refused(self):
        """usable 0 = the platform would not say. Refusing to start over a
        number nobody can read is worse than the failure this prevents;
        serve/server.py's runtime check still refuses the load itself."""
        check_registry_budget(MODELS, budget=0, usable=0)
        check_registry_budget(MODELS, budget=64 * GB, usable=0)

class TestRegistryLines(unittest.TestCase):
    """What the operator is shown when choosing a budget: the startup
    banner, `--plan`, and the output of a refusal all print these."""

    class Plan:
        floor_bytes = 12 * GB
        recommended_bytes = 30 * GB

    def plan(self, path: str, ctx: int):
        return self.Plan()

    def test_each_container_is_priced(self):
        lines = describe_registry(MODELS, budget=24 * GB, usable=64 * GB,
                                  plan=self.plan)
        self.assertEqual(len(lines), 3)
        self.assertIn("glm53", lines[0])
        self.assertIn("floor 12.0 GB", lines[0])
        self.assertIn("recommended 30.0 GB", lines[0])
        self.assertIn("ds41", lines[1])

    def test_the_arithmetic_verdicts(self):
        fits = describe_registry(MODELS, budget=24 * GB, usable=64 * GB,
                                 plan=self.plan)
        self.assertIn("48.0 GB against 64.0 GB usable — fits", fits[2])
        over = describe_registry(MODELS, budget=48 * GB, usable=64 * GB,
                                 plan=self.plan)
        self.assertIn("does not fit", over[2])
        self.assertIn("32.0 GB", over[2])       # the largest that fits

    def test_no_budget_says_what_the_engine_would_take(self):
        lines = describe_registry(MODELS, budget=0, usable=64 * GB,
                                  plan=self.plan)
        self.assertIn("no --budget", lines[2])
        self.assertIn("48.0 GB", lines[2])      # 3/4 of usable, per context

    def test_an_unreadable_container_does_not_hide_the_others(self):
        def plan(path: str, ctx: int):
            raise EngineError("plan_memory", -2, path)

        lines = describe_registry(MODELS, budget=8 * GB, usable=64 * GB,
                                  plan=plan)
        self.assertTrue(all("unreadable" in line for line in lines[:2]))
        self.assertIn("fits", lines[2])

    def test_an_unmeasurable_machine_says_so(self):
        lines = describe_registry(MODELS, budget=8 * GB, usable=0,
                                  plan=self.plan)
        self.assertIn("not checked", lines[2])

if __name__ == "__main__":
    unittest.main()

