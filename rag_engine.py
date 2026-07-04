import re
import textwrap

from groq import Groq

from config import GROQ_API_KEY, GROQ_MODEL, TOP_K
from pdf_processor import extract_pages_from_pdf
from text_chunker import chunk_pages
from embedder import embed_texts, embed_query
from vector_index import VectorIndex

NOT_FOUND_MSG = "I could not find this information in the uploaded document."

_groq_client = None
_index = VectorIndex()


def _get_groq_client() -> Groq:
    """Lazy-init the Groq client, validating the API key on first use."""
    global _groq_client
    if _groq_client is None:
        if not GROQ_API_KEY:
            raise ValueError(
                'GROQ_API_KEY is not set. Pass it as an environment variable, '
                'e.g.: docker run -e GROQ_API_KEY=gsk_... rag-app'
            )
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


def build_pipeline(pdf_path: str) -> int:
    """Extract, chunk, embed, and index a PDF. Returns number of chunks indexed."""
    pages_data = extract_pages_from_pdf(pdf_path)
    if not pages_data:
        raise ValueError('No text extracted — the PDF may be scanned/image-only.')

    chunks, chunk_page_nums = chunk_pages(pages_data)
    embeddings = embed_texts(chunks)
    _index.build(embeddings, chunks, chunk_page_nums)
    return len(chunks)


def retrieve_relevant_chunks(query: str, k: int = TOP_K) -> list[dict]:
    query_vec = embed_query(query)
    return _index.search(query_vec, k)


def build_rag_prompt(context_chunks: list[dict], question: str) -> str:
    context = '\n\n---\n\n'.join(
        f"[Page {c['page']}]\n{c['text']}" for c in context_chunks
    )
    return textwrap.dedent(f"""
        You are a precise document assistant. Your task is to answer the user's
        question using ONLY the information provided in the CONTEXT section below.

        Rules you must follow:

        1. Base your answer exclusively on the CONTEXT.
        2. Do not use external knowledge.
        3. If the answer is not explicitly stated in the CONTEXT, output EXACTLY:

          "{NOT_FOUND_MSG}"

        4. When Rule 3 applies:
            - Stop immediately.
            - Do not provide explanations.
            - Do not provide related information.
            - Do not provide the closest matching content.
            - Do not continue generating text.

        5. Never reveal your reasoning process.
        6. Never output <think> tags.
        7. Output only the final answer.
        8. Do not speculate, infer beyond what is written, or fabricate details.
        9. Quote or paraphrase from the CONTEXT where appropriate.
        10. Be concise and accurate.

        ============================================================
        CONTEXT:
        {context}
        ============================================================

        QUESTION:
        {question}

        ANSWER:
    """).strip()


def ask(question: str, k: int = TOP_K) -> str:
    if not question.strip():
        return 'Please provide a non-empty question.'

    relevant_chunks = retrieve_relevant_chunks(question, k=k)
    if not relevant_chunks:
        return NOT_FOUND_MSG

    prompt = build_rag_prompt(relevant_chunks, question)

    try:
        client = _get_groq_client()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    'role': 'system',
                    'content': (
                        'You are a helpful and precise document assistant. '
                        'Answer only from the provided context.'
                    ),
                },
                {'role': 'user', 'content': prompt},
            ],
            temperature=0.0,
            max_tokens=1024,
        )
    except ValueError:
        return '❌ GROQ_API_KEY is not set. Please set it as an environment variable.'
    except Exception as exc:
        return f'❌ Groq API error: {exc}'

    answer = response.choices[0].message.content.strip()
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()

    if answer != NOT_FOUND_MSG:
        cited_pages = sorted(set(c['page'] for c in relevant_chunks))
        answer += f"\n\n📄 Source page(s): {', '.join(map(str, cited_pages))}"

    return answer
