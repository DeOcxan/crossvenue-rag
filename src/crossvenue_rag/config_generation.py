"""The config sections of the answer runs: generators, judges, prompts, budget and calibration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from . import formatting
from .config_pipeline import MAX_RETRIES, RetrievalMethod
from .config_section import Section, distinct, named
from .confined_paths import DirectoryPath, PlainName
from .errors import UserError, is_blank

# generation.primary_pair names two configurations, a and b.
PRIMARY_PAIR_SIZE = 2
# The answer metrics a judgment yields, in table order (`generation.metrics.METRICS`).
AnswerMetric = Literal["faithfulness", "venue_attribution_error", "coverage", "behaviour_correct"]
# The question scopes the answer comparisons run on (`generation.answer_rows.SCOPES`).
AnswerScope = Literal["comparison", "all"]
# The judge dimensions calibrated against the human ratings (`generation.calibration`).
CalibrationDimension = Literal["supported", "venue_correct", "behaviour_correct", "coverage"]
# A sha256 as the config pins a prompt: 64 lowercase hex digits.
SHA256_PATTERN = r"^[0-9a-f]{64}$"
# The highest sampling temperature OpenRouter accepts; its range is [0, 2].
MAX_TEMPERATURE = 2.0


# A list or dict default is given through `Field`: ruff (RUF012) treats a class as a pydantic
# model only when it has a `BaseModel` base in its own file, and `Section` is in config_section.
class ModelConfig(Section):
    """One model on OpenRouter; `model` is an exact, versioned id, so a run names its model."""

    model: str
    # OpenRouter provider routing: providers tried in this order, with fallback
    # to any other provider disabled when the list is non-empty.
    provider_order: list[str] = Field([])

    @field_validator("provider_order")
    @classmethod
    def _providers_named_once(cls, v: list[str]) -> list[str]:
        for provider in v:
            named(provider, "every provider_order entry")
        return distinct(v, "provider_order")

    # False where the serving endpoint rejects `temperature`; the parameter is
    # then not sent, and every result records the parameters actually sent.
    temperature_supported: bool = True

    @field_validator("model")
    @classmethod
    def _exact_id(cls, v: str) -> str:
        if is_blank(v) or "latest" in v.lower():
            raise ValueError(f"model {v!r}: an exact model id is required; an alias is refused")
        return v


class GeneratorParams(Section):
    """Sampling parameters shared by every generator, so they differ only in model."""

    reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] = "minimal"
    max_tokens: int = Field(4000, gt=0)
    temperature: float | None = Field(0.0, ge=0.0, le=MAX_TEMPERATURE)


class JudgeParams(Section):
    """The judges' parameters, the same for every candidate. No temperature."""

    effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    max_tokens: int = Field(8000, gt=0)


class MatrixEntry(Section):
    """One answer configuration: a generator over a retrieval method."""

    name: str = Field(min_length=1)
    generator: str
    retriever: RetrievalMethod


class GenerationConfig(Section):
    """Answer generation and judging: models, prompts, budget, matrix and calibration."""

    top_k: int = Field(8, gt=0)
    cache_dir: DirectoryPath = Path("data/cache/llm")
    max_retries: int = Field(5, ge=0, le=MAX_RETRIES)
    max_invalid_retries: int = Field(2, ge=0)
    backoff_seconds: float = Field(2.0, gt=0.0, allow_inf_nan=False)
    # Most one run may spend, in USD, from the cost OpenRouter reports.
    budget_usd: float = Field(3.0, gt=0.0, allow_inf_nan=False)
    timeout_seconds: float = Field(180.0, gt=0.0, allow_inf_nan=False)
    generator_prompt: Path = Path("prompts/generator.md")
    # sha256 of the prompt files; a call is refused when the file differs.
    # None skips the check.
    generator_prompt_sha256: str | None = Field(None, pattern=SHA256_PATTERN)
    # Answer JSON schema matching the prompt: 1 for prompts/generator.v1.md,
    # 2 (partial, missing) for prompts/generator.md. Checked against the
    # prompt's version line.
    answer_schema_version: Literal[1, 2] = 2
    judge_prompt: Path = Path("prompts/judge.md")
    judge_prompt_sha256: str | None = Field(None, pattern=SHA256_PATTERN)
    generators: dict[str, ModelConfig] = Field({})
    generator_params: GeneratorParams = GeneratorParams()
    # Judge candidates; `judge` names the one in use, chosen by calibration
    # against human ratings (docs/adr/0010-judge.md). None while no judge is chosen.
    # A judge's name is part of a file name in a run: `judgements_<judge>.jsonl`.
    judges: dict[PlainName, ModelConfig] = Field({})
    judge: str | None = None
    judge_params: JudgeParams = JudgeParams()
    matrix: list[MatrixEntry] = Field([])
    # Pre-registered primary comparison: [config a, config b], its metrics and
    # scope. Every other pair of configurations is secondary, Holm-adjusted.
    primary_pair: list[str] = Field([])
    primary_metrics: list[AnswerMetric] = Field([])
    primary_scope: AnswerScope = "comparison"
    # sha256 of the generator prompt the primary comparison was registered
    # for; under any other prompt the pair is secondary. None: every prompt.
    primary_prompt_sha256: str | None = Field(None, pattern=SHA256_PATTERN)
    # The answer run whose calibration.json (`crossvenue select-judge`) says
    # which judge dimensions passed; metrics of the others are marked.
    calibration_run: Path | None = None
    calibration_size: int = Field(40, gt=0)
    calibration_seed: int = 20260924
    # Questions whose answers never enter the calibration sample: the judge
    # prompt was revised on them in the trial run (methodology section 9).
    calibration_exclude: list[str] = Field([])
    # Pre-registered in methodology section 9: a judge dimension enters a
    # conclusion only at or above its threshold of agreement with the human
    # ratings (Cohen's kappa for the binary ones, Spearman for coverage).
    calibration_thresholds: dict[CalibrationDimension, float] = Field(
        {
            "supported": 0.61,
            "venue_correct": 0.61,
            "behaviour_correct": 0.61,
            "coverage": 0.70,
        }
    )
    # Rated answers left out of `behaviour_correct` against the human, with the
    # reason in methodology section 9 (the rating screen showed a wrong flag).
    calibration_exclude_behaviour: list[str] = Field([])

    @field_validator("calibration_thresholds")
    @classmethod
    def _agreement_thresholds(
        cls, v: dict[CalibrationDimension, float]
    ) -> dict[CalibrationDimension, float]:
        """Kappa and Spearman lie in [-1, 1], so a threshold outside it never or always passes."""
        outside = [
            f"{d} ({formatting.own_form(t, formatting.CONFIG_THRESHOLD_DECIMALS)})"
            for d, t in v.items()
            if not -1.0 <= t <= 1.0
        ]
        if outside:
            raise ValueError(
                f"calibration_thresholds must lie in [\u22121, 1]: {', '.join(outside)} does not"
            )
        return v

    @model_validator(mode="after")
    def _matrix_refers_to_known_generators(self) -> GenerationConfig:
        unknown = [m.generator for m in self.matrix if m.generator not in self.generators]
        if unknown:
            raise ValueError(f"generation.matrix names unknown generators: {', '.join(unknown)}")
        names = [m.name for m in self.matrix]
        distinct(names, "generation.matrix names")
        if self.primary_pair and (
            len(self.primary_pair) != PRIMARY_PAIR_SIZE
            or self.primary_pair[0] == self.primary_pair[1]
        ):
            raise ValueError(
                f"generation.primary_pair names {formatting.count(PRIMARY_PAIR_SIZE)} different "
                "configurations"
            )
        distinct(self.primary_metrics, "generation.primary_metrics")
        missing = [p for p in self.primary_pair if p not in names]
        if missing:
            raise ValueError(
                f"generation.primary_pair names unknown configurations: {', '.join(missing)}"
            )
        if self.judge is not None and self.judge not in self.judges:
            raise ValueError(f"generation.judge {self.judge!r} is not one of generation.judges")
        return self

    def chosen_judge(self, name: str | None = None) -> str:
        """The name of the judge to use: `name`, else the configured `judge`.

        Raises:
            UserError: No judge is named or configured, or the name is not a candidate.
        """
        chosen = name if name is not None else self.judge
        candidates = ", ".join(self.judges) or "none configured"
        if chosen is None:
            raise UserError(
                "no judge chosen", hint=f"pass --judge ({candidates}) or set generation.judge"
            )
        if chosen not in self.judges:
            source = "--judge" if name is not None else "generation.judge"
            message = f"{source} {chosen!r} is not a judge of generation.judges"
            raise UserError(message, hint=f"pass --judge with one of: {candidates}")
        return chosen

    def judge_model(self, name: str | None) -> ModelConfig:
        """The model of the judge `chosen_judge` picks.

        Raises:
            UserError: No judge is named or configured, or the name is not a candidate.
        """
        return self.judges[self.chosen_judge(name)]
