from ragsearch.search import rrf_fuse
from ragsearch.suggest import Suggester, build_suggestions, extract_phrases
from ragsearch.text import chunk_sentences, clean_article, clean_highlights, make_title, split_sentences


def test_clean_article_removes_source_tags_artefacts_and_boilerplate():
    raw = "London (CNN) -- The minister resigned . Click here for more. He said it was time . "
    assert clean_article(raw) == "The minister resigned. He said it was time."
    assert clean_article("(The Hollywood Reporter)Melissa Benoist is cast.") == "Melissa Benoist is cast."
    assert clean_article("The (big) cat sat.") == "The (big) cat sat."  # ordinary brackets kept


def test_clean_highlights_joins_lines_into_sentences():
    assert clean_highlights("Zully gave a kidney .\nSix patients helped") == "Zully gave a kidney. Six patients helped."


def test_split_sentences_and_title():
    text = 'He said "Yes." Then he left! Did it work? Yes.'
    assert split_sentences(text) == ['He said "Yes."', "Then he left!", "Did it work?", "Yes."]
    assert make_title("One two three four. Second sentence.", max_words=3) == "One two three…"


def test_chunks_respect_size_overlap_and_cover_all_text():
    sentences = [f"Sentence number {i} has exactly seven words." for i in range(60)]  # 7 words each
    text = " ".join(sentences)
    chunks = chunk_sentences(text, chunk_words=70, overlap_words=14)
    assert all(len(c.split()) <= 70 for c in chunks)
    for a, b in zip(chunks, chunks[1:]):          # consecutive chunks share trailing sentences
        assert split_sentences(a)[-1] in b
    assert all(s in " ".join(chunks) for s in sentences)  # nothing lost


def test_short_text_is_one_chunk():
    assert chunk_sentences("Just one short sentence.", 200, 50) == ["Just one short sentence."]


def test_rrf_rewards_documents_found_by_both_methods():
    bm25 = [(1, 9.0, 10), (2, 5.0, 20), (3, 1.0, 30)]
    dense = [(3, 0.9, 31), (1, 0.8, 11), (4, 0.7, 40)]
    fused = rrf_fuse({"bm25": bm25, "dense": dense})
    order = [doc for doc, *_ in fused]
    assert order[0] == 1                       # rank 1 + rank 2 beats everything
    assert order.index(3) < order.index(2)     # found by both beats found by one
    assert dict((d, m) for d, _, _, m in fused)[1] == ["bm25", "dense"]
    assert abs(fused[0][1] - (1 / 61 + 1 / 62)) < 1e-12


def test_suggester_matches_any_word_and_ranks_by_frequency():
    texts = ["David Cameron met Barack Obama."] * 6 + ["Cameron Diaz smiled."] * 5 + ["Monday was quiet."] * 9
    phrases = build_suggestions(texts, min_docs=5)
    assert "Monday" not in {p for p, _ in phrases}            # weekdays are not topics
    suggester = Suggester(phrases)
    assert suggester.suggest("cam")[:2] == ["David Cameron", "Cameron Diaz"]
    assert suggester.suggest("oba") == ["Barack Obama"]
    assert suggester.suggest("x") == []


def test_extract_phrases_strips_leading_function_words():
    assert "Prime Minister" in extract_phrases("The Prime Minister spoke.")


def test_split_sentences_keeps_abbreviations_and_initials_together():
    text = "A year ago, when Dr. Smith met Mr. Jones in the U.S. Army, J. K. Rowling spoke. Then at 10 a.m. Monday it ended."
    assert split_sentences(text) == [
        "A year ago, when Dr. Smith met Mr. Jones in the U.S. Army, J. K. Rowling spoke.",
        "Then at 10 a.m. Monday it ended."]
