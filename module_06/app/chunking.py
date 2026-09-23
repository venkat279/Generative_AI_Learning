"""Indexing - text chunking strategy: recursive character text splitter.

Splitting happens in two clean phases:
  1. Atomize - recursively break the text on the "biggest" separator first
     (paragraph breaks), falling back to smaller separators (line breaks,
     sentence ends, spaces, then raw characters as a last resort) only for
     pieces that are still too big. This produces small pieces that each
     fit within chunk_size, while keeping sentences/paragraphs intact
     wherever possible.
  2. Merge - greedily pack consecutive small pieces back together up to
     chunk_size, so we don't end up with one chunk per sentence.

Overlap is then applied by carrying the tail of each chunk into the start
of the next one, so context isn't lost right at a chunk boundary.
"""

SEPARATORS = ["\n\n", "\n", ". ", " "]


def _atomize(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    if not separators:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    separator, rest = separators[0], separators[1:]
    parts = text.split(separator)

    pieces: list[str] = []
    for part in parts:
        pieces.extend(_atomize(part, chunk_size, rest))
    return pieces


def _merge(pieces: list[str], chunk_size: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current} {piece}".strip() if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return chunks


def _apply_overlap(chunks: list[str], chunk_overlap: int) -> list[str]:
    if chunk_overlap <= 0 or len(chunks) < 2:
        return chunks

    overlapped = [chunks[0]]
    for i in range(1, len(chunks)):
        tail = chunks[i - 1][-chunk_overlap:]
        overlapped.append(f"{tail} {chunks[i]}".strip())
    return overlapped


def recursive_split(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 60,
    separators: list[str] | None = None,
) -> list[str]:
    """Split text into <= ~chunk_size character chunks with a small overlap."""
    separators = separators if separators is not None else SEPARATORS
    pieces = _atomize(text, chunk_size, separators)
    chunks = _merge(pieces, chunk_size)
    return _apply_overlap(chunks, chunk_overlap)
