from rag.filters import build_where, date_to_int, infer_section_contains


def test_date_to_int_ok():
    assert date_to_int("2024-12-31") == 20241231


def test_date_to_int_none_and_bad():
    assert date_to_int(None) is None
    assert date_to_int("") is None
    assert date_to_int("2024/12/31") is None


def test_build_where_none_when_no_filters():
    assert build_where(None, None, None, None, None) is None


def test_build_where_single_clause():
    assert build_where("tsla", None, None, None, None) == {"ticker": "TSLA"}


def test_build_where_and_multiple_clauses():
    where = build_where("tsla", "10-k", "2023-01-01", "2023-12-31", "Item 1A - Risk Factors")
    assert where is not None
    assert "$and" in where

    clauses = where["$and"]
    assert {"ticker": "TSLA"} in clauses
    assert {"form": "10-K"} in clauses
    assert {"reportDate_int": {"$gte": 20230101}} in clauses
    assert {"reportDate_int": {"$lte": 20231231}} in clauses
    assert {"section": "Item 1A - Risk Factors"} in clauses


def test_infer_section_contains_risk_factors():
    assert infer_section_contains("What are the main risk factors?") == "Risk Factors"
    assert infer_section_contains("Tell me about Item 1A") == "Risk Factors"


def test_infer_section_contains_mda():
    assert infer_section_contains("Discuss management's discussion and analysis") == "MD&A"
    assert infer_section_contains("liquidity and cash flows outlook") == "MD&A"


def test_infer_section_contains_none():
    assert infer_section_contains("What is Tesla's favorite color?") is None
