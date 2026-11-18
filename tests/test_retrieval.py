from rag import load_documents, retrieve, build_context

def test_load_documents():
    assert len(load_documents()) == 6

def test_cancellation_policy_retrieved():
    hits = retrieve("How many hours before an appointment must I cancel?", k=2)
    assert "cancellations.txt" in [name for name, _, _ in hits]

def test_context_includes_source():
    assert "[Source: cancellations.txt]" in build_context([("cancellations.txt", "18 hours", 0.9)])

def test_empty_corpus():
    assert retrieve("question", documents=[]) == []

def test_lab_results_retrieved():
    hits = retrieve("How are laboratory results communicated?", k=2)
    assert "lab_results.txt" in [name for name, _, _ in hits]

def test_excludes_zero_scores_and_tiebreaks_by_filename():
    docs = [
        ("b.txt", "alpha beta"),
        ("a.txt", "alpha beta"),
        ("c.txt", "unrelated jargon"),
    ]
    hits = retrieve("alpha beta", k=5, documents=docs)
    names = [name for name, _, _ in hits]
    assert "c.txt" not in names
    assert names == ["a.txt", "b.txt"]
