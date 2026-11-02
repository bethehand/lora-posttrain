"""wiki-search tools with read_section's argument named `section` instead of `section_id`.

OLMo 3 7B Instruct writes `section_id` badly in its pythonic call format, as `section_ "..."`;
renaming the argument raised well-formed read calls from 22% to 70% in replayed episodes
(results/olmo-tool-call-diagnosis). Everything else, including the description and the
`section_id` keys in view_sections output, is upstream's WikiSearchToolset.
"""

import verifiers.v1 as vf
from wiki_search.servers.wiki import WikiSearchToolset


class SectionArgWikiToolset(WikiSearchToolset):
    @vf.tool
    def read_section(self, section: str) -> str:
        """Read the text of a section (`page_id:section`; use `page_id:full` for all)."""
        return WikiSearchToolset.read_section(self, section)


if __name__ == "__main__":
    SectionArgWikiToolset.run()
