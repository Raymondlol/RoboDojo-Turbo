"""RoboDojo-Turbo: unofficial, verifiable evaluation speedups for the RoboDojo benchmark.

The package rewrites a local RoboDojo checkout in place (regex-anchored, pinned to known upstream file hashes, fully
reversible) and ships the tools used to show that each speedup leaves physics and scoring unchanged. It is not
affiliated with or endorsed by the RoboDojo maintainers; results produced with it are not official leaderboard entries.
"""
__version__ = "0.1.0a1"

MARK = "[robodojo-turbo]"      # every block we inject into upstream files starts with a line carrying this marker
ENV_PREFIX = "RDTURBO_"       # runtime switches; upstream already uses ROBODOJO_*
