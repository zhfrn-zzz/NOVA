"""Web search tool — answers current-events and factual queries via DuckDuckGo.

Uses the ddgs library (no API key required) to fetch search results and
return concise summaries suitable for voice delivery.
"""

import asyncio
import logging

from ddgs import DDGS

logger = logging.getLogger(__name__)

_MAX_RESULTS = 3


async def web_search(query: str) -> str:
    """Search the web using DuckDuckGo and return a concise summary.

    Runs the synchronous DDGS client in a thread executor to avoid
    blocking the async event loop.  If the first attempt returns no
    results, retries once with a broadened query.

    Args:
        query: The search query string.

    Returns:
        Formatted search results as a single string, or an error message.
    """
    logger.info("Web search: %r", query)

    try:
        results = await asyncio.to_thread(_search_sync, query)

        # Retry once with a broader query if no results
        if not results:
            retry_query = f"{query} info"
            logger.info("Web search retry with: %r", retry_query)
            results = await asyncio.to_thread(_search_sync, retry_query)
    except Exception as e:
        logger.error("Web search failed: %s", e)
        return f"Pencarian gagal: {e}"

    if not results:
        return "Tidak ditemukan hasil pencarian."

    # Format results concisely for voice output
    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        body = r.get("body", "")
        lines.append(f"{i}. {title}: {body}")

    formatted = "\n".join(lines)
    logger.info("Web search returned %d results for %r", len(results), query)
    return formatted


def _search_sync(query: str) -> list[dict]:
    """Run a synchronous DuckDuckGo text search.

    Args:
        query: The search query.

    Returns:
        List of result dicts with 'title', 'href', 'body' keys.
    """
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=_MAX_RESULTS))


if __name__ == "__main__":

    async def main() -> None:
        print("=== Web Search Tool Test ===\n")

        queries = [
            "siapa presiden Indonesia 2024",
            "cuaca Jakarta hari ini",
            "latest Python release",
        ]

        for q in queries:
            print(f"Query: {q}")
            result = await web_search(q)
            print(f"Result:\n{result}\n")
            print("-" * 60)

    asyncio.run(main())
