"""The one tool a model has. A tool server runs as `python -m` of its class's module."""

import verifiers.v1 as vf
from factor_idea import workers


class FactorToolset(vf.Toolset[vf.SharedToolsetConfig]):
    TOOL_PREFIX = None  # the model sees `test_factor`, not a prefixed name

    @vf.tool
    async def test_factor(self, expression: str) -> str:
        """Compute a candidate factor on July 2022 to December 2024. Returns an error if the expression is invalid; otherwise its rank IC and incremental IC, overall and by period."""
        return await workers.report(expression)


if __name__ == "__main__":
    FactorToolset.run()
