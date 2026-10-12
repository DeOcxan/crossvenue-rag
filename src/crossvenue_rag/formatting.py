"""Text a person reads for a value: the number formats of docs/style.md rule 11.

Every number written into Markdown or printed to the terminal goes through one
of these functions. CSV and JSON files are not formatted here: each writer keeps
the precision the committed runs of its kind have, and its docstring names it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast, overload

MINUS = "\u2212"
# The decimals of each kind of number, one per row of docs/style.md rule 11.
METRIC_DECIMALS = 3
P_VALUE_DECIMALS = 3
SHARE_DECIMALS = 2
DESIGN_DECIMALS = 2
PERCENT_DECIMALS = 1
# Below LATENCY_INTEGER_FROM_MS; from it, a latency is an integer.
MS_DECIMALS = 1
USD_DECIMALS = 3
# A ratio or mean of counts (rule 11), one constant per kind. Tokens per word of the
# corpus: 2.47.
TOKENS_PER_WORD_DECIMALS = 2
# A judge's cost over its generator's: 7 to 138 times.
COST_RATIO_DECIMALS = 0
# A median or mean count per question, chunk or query: words, tokens, pairs.
COUNT_MEAN_DECIMALS = 0
# A bench mean the findings state: calls per query.
BENCH_SUMMARY_DECIMALS = 1
# A 1-minute load average, its median and its maximum alike: 4.09.
LOAD_AVERAGE_DECIMALS = 2


# Values that keep their own form (docs/style.md rule 11, last line), one constant per kind,
# with the decimals of the source the form comes from; `own_form` takes it.
# A threshold as configs/default.yaml sets it: eval.duplicate_threshold 0.92,
# generation.calibration_thresholds 0.61 and 0.70.
CONFIG_THRESHOLD_DECIMALS = 2
# The generators' temperature as configs/default.yaml sets it in generation.generator_params: 0.0.
TEMPERATURE_DECIMALS = 1
# An effect size d, as the conventional small, medium and large ones are written: 0.2, 0.5, 0.8.
EFFECT_SIZE_DECIMALS = 1
# A TF-IDF cosine threshold of the PancakeSwap counts (corpus_pancakeswap, claims), such as 0.4.
COSINE_THRESHOLD_DECIMALS = 1
# A pool fee tier as the venues' documentation writes it: 0.25%, 0.30%.
FEE_TIER_DECIMALS = 2
# A bound set as a constant in the code: retrieval_tables.NDCG_PRIME_GAIN_BOUND 0.01,
# the figures' SMALLEST_BAR_MS 0.05.
CODE_BOUND_DECIMALS = 2
# Latency from this many milliseconds is shown as an integer, below it with one decimal.
LATENCY_INTEGER_FROM_MS = 10.0
# A p-value below this is shown as "< 0.001".
SMALLEST_SHOWN_P = 0.001
# The text of a table cell with no value.
MISSING = "-"
# The hex digits of a shortened hash (docs/style.md rule 37); the full value is in a
# file the sentence names.
SHORT_HASH_DIGITS = 8
# A duration is written in seconds below this many seconds, in minutes below this
# many minutes, then in hours and minutes.
DURATION_UNIT_LIMIT = 90
SECONDS_PER_MINUTE = 60
MINUTES_PER_HOUR = 60
# A percent is a fraction times this.
PER_CENT = 100


def _minus(text: str) -> str:
    """Replaces the ASCII hyphen of a formatted negative number with U+2212.

    A value that rounds to zero is written without a sign (`-0.000` is `0.000`),
    so every formatter treats a rounded zero as `signed` does.
    """
    if text.startswith("-") and not any(digit in text for digit in "123456789"):
        text = text[1:]
    return text.replace("-", MINUS)


def metric(value: float) -> str:
    """A metric value in [0, 1], kappa or Spearman at three decimals: `0.620`."""
    return _minus(f"{value:.{METRIC_DECIMALS}f}")


def signed(value: float, digits: int = METRIC_DECIMALS) -> str:
    """A difference with its sign: `+0.280`, and `-0.062` with U+2212.

    Three decimals unless `digits` says otherwise. Zero, including a negative
    zero after rounding, is written with a plus sign: `+0.000`.
    """
    text = f"{value:+.{digits}f}"
    if float(text) == 0:
        text = "+" + text[1:]
    return _minus(text)


def interval(low: float, high: float) -> str:
    """A confidence interval of a difference: `[+0.160, +0.400]`."""
    return f"[{signed(low)}, {signed(high)}]"


def bounds(low: float, high: float) -> str:
    """A confidence interval of a metric value, kappa's say: `[0.317, 0.645]`."""
    return f"[{metric(low)}, {metric(high)}]"


def p_value(p: float) -> str:
    """A p-value at three decimals, or `< 0.001` below that."""
    if p < SMALLEST_SHOWN_P:
        return f"< {SMALLEST_SHOWN_P:.{P_VALUE_DECIMALS}f}"
    return f"{p:.{P_VALUE_DECIMALS}f}"


def design_parameter(value: float) -> str:
    """A power-analysis parameter at two decimals: `0.20`.

    A detectable delta, a power or a probability is a design value, which
    docs/style.md rule 11 gives at two decimals.
    """
    return _minus(f"{value:.{DESIGN_DECIMALS}f}")


def share(part: int, whole: int) -> str:
    """A share of a small count with the count: `0.98 (49/50)`."""
    return f"{part / whole:.{SHARE_DECIMALS}f} ({part}/{whole})"


def fixed(value: float, digits: int) -> str:
    """A number at `digits` decimals with no thousands separator: `0.25`, `2505.3`."""
    return _minus(f"{value:.{digits}f}")


def percent_figure(part: float, whole: float) -> str:
    """`part` as a percent of `whole`, without the sign, for text that writes it: `10.8`."""
    return _minus(f"{PER_CENT * part / whole:.{PERCENT_DECIMALS}f}")


def percent(fraction: float) -> str:
    """A descriptive share as a percent with one decimal: `10.8%`."""
    return _minus(f"{PER_CENT * fraction:.{PERCENT_DECIMALS}f}%")


def ms(value: float) -> str:
    """A latency in milliseconds: integer from 10 ms, one decimal below: `2,531`, `0.1`."""
    if abs(value) >= LATENCY_INTEGER_FROM_MS:
        return _minus(f"{value:,.0f}")
    return _minus(f"{value:.{MS_DECIMALS}f}")


def signed_ms(value: float) -> str:
    """A latency difference with its sign, at the precision of `ms`: `+83`, `-2.5` with U+2212."""
    return signed(value, 0 if abs(value) >= LATENCY_INTEGER_FROM_MS else MS_DECIMALS)


def cost(usd: float) -> str:
    """A cost in USD at three decimals: `$0.657`."""
    return "$" + cost_figure(usd)


def cost_figure(usd: float) -> str:
    """A cost in USD without the dollar sign, for text that writes it: `0.657`."""
    return _minus(f"{usd:.{USD_DECIMALS}f}")


def short_hash(digest: str) -> str:
    """A hash shortened to its first `SHORT_HASH_DIGITS` hex digits and an ellipsis (U+2026).

    For a sentence that names the file holding the full value; a hash no file holds is
    written in full.
    """
    return digest[:SHORT_HASH_DIGITS] + "\u2026"


def count(n: int) -> str:
    """A count with a thousands separator: `11,934`."""
    return _minus(f"{n:,}")


def ratio(value: float, decimals: int) -> str:
    """A ratio or mean of counts at the decimals of its kind's constant: `2.47`, `138`."""
    return _minus(f"{value:,.{decimals}f}")


def call_seconds(seconds: float) -> str:
    """A model-call duration in whole seconds: `243`. Retrieval latency is `ms`."""
    return _minus(f"{seconds:,.0f}")


def number(value: float, decimals: int) -> str:
    """A number that is none of the kinds above (a retrieval score), at `decimals` decimals."""
    return _minus(f"{value:,.{decimals}f}")


@overload
def own_form(value: object) -> str: ...
@overload
def own_form(value: float, decimals: int) -> str: ...
def own_form(value: object, decimals: int | None = None) -> str:
    """A seed, config value or identifier in its own form, as written: `20260924`, `60000`, `0.70`.

    docs/style.md rule 11 keeps these as they are written: they are settings or names
    rather than counts, so they take no thousands separator. A kind whose source writes
    a fixed number of decimals passes its constant as `decimals` (a calibration threshold,
    `CONFIG_THRESHOLD_DECIMALS`: `0.70`); without one, the value is written as Python
    writes it (`0.5`).
    """
    if decimals is None:
        return str(value)
    return fixed(cast(float, value), decimals)


def location(path: object, line: int) -> str:
    """A place in a file as a message names it, `path:line`; the line number is an identifier."""
    return f"{path}:{own_form(line)}"


def plural(n: int, singular: str, plural_form: str | None = None) -> str:
    """A count with its noun in the right number: `1 round`, `2 rounds`, `1,000 rounds`."""
    noun = singular if n == 1 else (plural_form or singular + "s")
    return f"{count(n)} {noun}"


def cpus(n: int | None) -> str:
    """A CPU count as a machine record states it: `1 CPU`, `10 CPUs`, or unknown for None."""
    return plural(n, "CPU") if n else "an unknown number of CPUs"


def revision_text(revision: str | None) -> str:
    """How a message names a model revision: ` at revision <revision>`, or nothing for None."""
    return f" at revision {revision}" if revision else ""


def optional(value: float | str | None, fmt: Callable[[float], str]) -> str:
    """`fmt(value)`, or `MISSING` when there is no value (`None` or an empty CSV cell).

    Raises:
        ValueError: `value` is text that is not a number.
    """
    if value is None or value == "":
        return MISSING
    return fmt(float(value))


def duration(seconds: float) -> str:
    """An elapsed time a person reads: `45s`, `12m`, `2h05m`."""
    # Rounded to the unit before it is chosen, so 89.6 s is 1m and 7,182 s is 2h00m.
    whole_seconds = round(seconds)
    if whole_seconds < DURATION_UNIT_LIMIT:
        return f"{whole_seconds}s"
    minutes = round(seconds / SECONDS_PER_MINUTE)
    if minutes < DURATION_UNIT_LIMIT:
        return f"{minutes}m"
    hours, rest = divmod(minutes, MINUTES_PER_HOUR)
    return f"{hours}h{rest:02d}m"
