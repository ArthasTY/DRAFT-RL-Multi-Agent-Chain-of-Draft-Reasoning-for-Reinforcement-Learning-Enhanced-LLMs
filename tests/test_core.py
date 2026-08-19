from draftrl.core import (
    extract_json_object,
    jaccard_distance,
    parse_draft,
    stable_hash_features,
    verify_numeric,
)


def test_parse_valid_chain_of_draft() -> None:
    parsed = parse_draft("STEP 1: Add the two counts\nSTEP 2: Check the total\nFINAL: 42")
    assert parsed.cod_valid
    assert parsed.steps == ["Add the two counts", "Check the total"]
    assert parsed.final_answer == "42"


def test_parse_rejects_verbose_step() -> None:
    parsed = parse_draft("STEP 1: This step has far too many words now\nFINAL: 42")
    assert not parsed.cod_valid
    assert any("words" in violation for violation in parsed.violations)


def test_verifier_and_features_are_deterministic() -> None:
    assert verify_numeric("The answer is $1,234.0", "1234")
    assert not verify_numeric("1235", "1234")
    assert stable_hash_features("same text", 32) == stable_hash_features("same text", 32)
    assert jaccard_distance("one two", "one three") > 0


def test_extract_json_inside_code_fence() -> None:
    parsed = extract_json_object('```json\n{"coherence": 0.8}\n```')
    assert parsed == {"coherence": 0.8}
