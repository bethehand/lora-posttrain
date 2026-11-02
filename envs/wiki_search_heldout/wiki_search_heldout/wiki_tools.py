"""Upstream's wiki-search tools, unchanged except for the embedder's CPU threads (threads.py).

A tool server runs as `python -m <its class's module>`, so this module hosts exactly one toolset.
"""

import wiki_search_heldout.threads  # noqa: F401  (must run before chroma builds its session)
from wiki_search.servers.wiki import WikiSearchToolset


class WikiToolset(WikiSearchToolset):
    pass


if __name__ == "__main__":
    WikiToolset.run()
