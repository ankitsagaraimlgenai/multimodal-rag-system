from mmrag.ingestion.chunking import TokenChunker, context_windows

count = lambda t: len(t.split())
TEXT = ("Revenue grew twelve percent in the third quarter. Costs fell by four percent. "
        "The board approved a new dividend policy. Cash reserves reached a record high. "
        "Management expects stable growth next year. Risks include currency swings and supply delays.")


def test_chunks_respect_size_and_offsets():
    ch = TokenChunker(count, chunk_size=15, overlap=4).chunk(TEXT)
    assert len(ch) > 1
    for c in ch:
        assert c.tokens <= 15
        assert TEXT[c.start:c.end] == c.text            # offsets are exact
    assert ch[0].start == 0 and ch[-1].end == len(TEXT.rstrip())


def test_overlap_and_progress():
    ch = TokenChunker(count, chunk_size=15, overlap=8).chunk(TEXT)
    assert all(b.start < a.end for a, b in zip(ch, ch[1:]))     # consecutive chunks overlap
    assert all(b.start > a.start for a, b in zip(ch, ch[1:]))   # and always advance


def test_zero_overlap_has_no_overlap():
    ch = TokenChunker(count, chunk_size=15, overlap=0).chunk(TEXT)
    assert all(b.start >= a.end for a, b in zip(ch, ch[1:]))


def test_oversized_sentence_is_split_on_words():
    long = " ".join(f"w{i}" for i in range(100)) + "."
    ch = TokenChunker(count, chunk_size=20, overlap=5).chunk(long)
    assert len(ch) >= 5 and all(c.tokens <= 20 for c in ch)


def test_empty_and_whitespace():
    assert TokenChunker(count, 10, 2).chunk("") == []
    assert TokenChunker(count, 10, 2).chunk("  \n \n ") == []


def test_context_windows():
    ch = TokenChunker(count, 15, 4).chunk(TEXT)
    w = context_windows(ch, 1)
    assert w[0] == (ch[0].start, ch[1].end) and w[-1] == (ch[-2].start, ch[-1].end)
    assert context_windows(ch, 0)[1] == (ch[1].start, ch[1].end)


def test_overlap_must_be_smaller_than_size():
    try:
        TokenChunker(count, 10, 10)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
