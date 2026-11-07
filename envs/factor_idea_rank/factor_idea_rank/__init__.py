"""factor-idea's task with the training reward of prereg/03b-pilot.md."""

__all__ = ["FactorIdeaRankTaskset"]


def __getattr__(name):
    if name == "FactorIdeaRankTaskset":
        from factor_idea_rank.taskset import FactorIdeaRankTaskset

        return FactorIdeaRankTaskset
    raise AttributeError(name)
