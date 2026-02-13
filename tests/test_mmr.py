from rag.mmr import mmr_select


def test_mmr_returns_all_if_n_le_k():
    docs = ["a", "b"]
    metas = [{"i": 1}, {"i": 2}]
    dists = [0.1, 0.2]

    out_docs, out_metas, out_dists = mmr_select(docs, metas, dists, k=5, lamb=0.7)
    assert out_docs == docs
    assert out_metas == metas
    assert out_dists == dists


def test_mmr_selects_exact_k_and_no_duplicates():
    docs = [
        "risk factor about product liability",
        "risk factor about production delays",
        "risk factor about suppliers",
        "risk factor about demand",
        "risk factor about key personnel",
        "risk factor about stock volatility",
    ]
    metas = [{"id": i} for i in range(len(docs))]
    dists = [0.10, 0.11, 0.12, 0.13, 0.14, 0.15]  # smaller = more relevant

    out_docs, out_metas, out_dists = mmr_select(docs, metas, dists, k=3, lamb=0.7)

    assert len(out_docs) == 3
    assert len(out_metas) == 3
    assert len(out_dists) == 3

    # No duplicates
    assert len({m["id"] for m in out_metas}) == 3


def test_mmr_prefers_most_relevant_first_pick():
    docs = ["x", "y", "z"]
    metas = [{"id": 0}, {"id": 1}, {"id": 2}]
    dists = [0.05, 0.20, 0.30]  # doc0 should be most relevant

    out_docs, out_metas, out_dists = mmr_select(docs, metas, dists, k=2, lamb=0.9)
    assert out_metas[0]["id"] == 0
