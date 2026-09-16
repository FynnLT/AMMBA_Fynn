"""Section 1c: the 02.09. test-run scripts are frozen, not maintained.

They call `run_slot` positionally and would need seven adaptations to the new
signature; no Chapter 5 result reads them. Freezing is only honest if it is
checked, so this asserts both halves of the claim: each script says what it is
frozen at, and no live harness module imports one.
"""
from pathlib import Path

HARNESS = Path(__file__).resolve().parent.parent / "harness"
LEGACY = HARNESS / "legacy"

FROZEN = ("eval_r0_r1.py", "eval_r2.py", "eval_r3_r4.py", "eval_r5.py",
          "blockd.py", "blockd2.py", "blocke.py")


def test_every_frozen_script_is_in_legacy_and_says_so():
    for name in FROZEN:
        path = LEGACY / name
        assert path.is_file(), f"{name} is not under harness/legacy/"
        head = path.read_text(encoding="utf-8")[:900]
        assert "FROZEN at the 02.09.2026" in head, name
        assert "fe52d63" in head, name
        assert not (HARNESS / name).exists(), f"{name} left behind in harness/"


def test_no_live_harness_module_imports_a_frozen_script():
    """A frozen script imported from live code is not frozen."""
    stems = {Path(name).stem for name in FROZEN}
    for module in sorted(HARNESS.glob("*.py")):
        source = module.read_text(encoding="utf-8")
        for line in source.splitlines():
            line = line.strip()
            if not line.startswith(("import ", "from ")):
                continue
            imported = line.split()[1].split(".")[0].rstrip(",")
            names = {imported} | {part.strip().split(".")[0]
                                  for part in line[len("import "):].split(",")}
            assert not (names & stems), f"{module.name}: {line}"


def test_readme_records_the_freeze():
    readme = (Path(__file__).resolve().parent.parent / "README.md")
    text = readme.read_text(encoding="utf-8")
    assert "fe52d63" in text
    assert "harness/legacy" in text


def test_gas_pilot_lives_with_its_toolchain():
    """A Hardhat script belongs next to hardhat.config.js: its own
    `path.join(__dirname, "../contracts")` only resolves from there."""
    repo = Path(__file__).resolve().parents[2]
    assert (repo / "amm-smart-contract" / "scripts" / "gas_pilot.js").is_file()
    assert not (HARNESS / "gas_pilot.js").exists()
