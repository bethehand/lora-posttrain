"""One simple idea in, one factor expression out, scored in a look-back-only sandbox."""

__all__ = ["FactorIdeaTaskset"]


def __getattr__(name):
    # The sandbox has no use for the training framework, so it loads only when a run asks for the taskset.
    if name == "FactorIdeaTaskset":
        from factor_idea.taskset import FactorIdeaTaskset

        return FactorIdeaTaskset
    raise AttributeError(name)
