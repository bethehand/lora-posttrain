"""Download the wiki corpus and build its search index once, before any eval or training run.

The wiki tool server does this in its own setup on first start: it downloads the page corpus
from Hugging Face, chroma's MiniLM embedding model from S3, and embeds 2,590 page titles. From
this server that takes several minutes and exceeds the launcher's tool-server startup timeout.
Later starts reuse the index in $WIKI_SEARCH_CACHE and the model in ~/.cache/chroma.

On the server, from $PRIME_RL_DIR:
    uv run python $REPO/scripts/warm_wiki_tools.py
"""

import asyncio

import verifiers.v1 as vf
from wiki_search.servers.wiki import WikiSearchToolset


async def main() -> None:
    toolset = WikiSearchToolset(vf.SharedToolsetConfig())
    await toolset.setup()
    hits = toolset.collection.query(query_texts=["Utsuro-bune meaning in Japanese"], n_results=3)
    print(f"{len(toolset.pages)} pages indexed; top hits: {hits['ids'][0]}")


if __name__ == "__main__":
    asyncio.run(main())
