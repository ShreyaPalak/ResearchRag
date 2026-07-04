from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=['\n\n', '\n', '. ', '? ', '! ', ' ', ''],
)


def chunk_pages(pages_data: list[dict]) -> tuple[list[str], list[int]]:
    """Split page-level text into overlapping chunks.

    Returns (chunks, chunk_pages) where chunk_pages[i] is the source
    page number for chunks[i].
    """
    chunks, chunk_pages = [], []
    for p in pages_data:
        for c in _splitter.split_text(p['text']):
            c = c.strip()
            if c:
                chunks.append(c)
                chunk_pages.append(p['page'])
    return chunks, chunk_pages
