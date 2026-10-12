"""An offline generation setup: a config, a scripted transport, retrievers and chunks."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from crossvenue_rag.config import Config
from crossvenue_rag.config_generation import GenerationConfig, MatrixEntry, ModelConfig
from crossvenue_rag.config_pipeline import RetrievalMethod
from crossvenue_rag.generation.llm import LLM
from crossvenue_rag.generation.openrouter import HttpResponse
from crossvenue_rag.models import Chunk, ChunkSource, Question, Result

SECRET = "sk-test-SECRET-never-shown"


class FakeTransport:
    """Replays scripted (status, payload) pairs, or answers through a function."""

    def __init__(
        self,
        *responses: tuple[int, dict[str, Any]],
        respond: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.responses = list(responses)
        self.respond = respond
        self.calls: list[tuple[str, dict[str, str], dict[str, Any]]] = []

    def post(
        self, url: str, headers: dict[str, str], body: dict[str, Any], timeout: float
    ) -> HttpResponse:
        self.calls.append((url, headers, body))
        if self.respond is not None:
            return HttpResponse(200, self.respond(url, body), {})
        status, payload = self.responses.pop(0)
        return HttpResponse(status, payload, {})


def openrouter_payload(
    content: dict[str, Any] | str,
    cost: float | None = 0.0012,
    model: str = "google/gemini-3.8-flash-20260901",
    provider: str = "Google",
    reasoning: int = 30,
) -> dict[str, Any]:
    usage: dict[str, Any] = {
        "prompt_tokens": 1000,
        "completion_tokens": 200,
        "completion_tokens_details": {"reasoning_tokens": reasoning},
    }
    if cost is not None:
        usage["cost"] = cost
    text = content if isinstance(content, str) else json.dumps(content)
    return {
        "model": model,
        "provider": provider,
        "choices": [{"message": {"content": text}}],
        "usage": usage,
    }


def gen_config(tmp_path: Path, **patch: Any) -> GenerationConfig:
    base = {
        "cache_dir": tmp_path / "cache",
        "backoff_seconds": 1.0,
        "generators": {
            "gemini": ModelConfig(
                model="google/gemini-3.8-flash",
                provider_order=["google-vertex/global"],
                temperature_supported=False,
            ),
            "qwen": ModelConfig(model="qwen/qwen3.8-27b", provider_order=["deepinfra/bf16"]),
        },
        "judges": {
            "sonnet": ModelConfig(model="anthropic/claude-sonnet-5", provider_order=["anthropic"]),
            "opus": ModelConfig(model="anthropic/claude-opus-5.5", provider_order=["anthropic"]),
        },
    }
    return GenerationConfig(**{**base, **patch})


def chunk(chunk_id: str, venue: str, text: str, sources: list[ChunkSource] | None = None) -> Chunk:
    sources = sources or [ChunkSource(venue, f"{venue}/page.md", ["H"], f"https://{venue}", 0)]
    return Chunk(
        chunk_id=chunk_id,
        venue=venue,
        venue_type="cex",
        doc_path=sources[0].doc_path,
        heading_path=["H"],
        source_url=sources[0].source_url,
        text=text,
        token_count=10,
        position=0,
        sources=sources,
    )


def mainnet_testnet_chunk() -> Chunk:
    return chunk(
        "c-bin",
        "binance",
        "IP bans scale from 2 minutes to 3 days.",
        [
            ChunkSource(
                "binance", "rest-api.md", ["Limits"], "https://m", 0, environment="mainnet"
            ),
            ChunkSource(
                "binance", "testnet/rest-api.md", ["Limits"], "https://t", 0, environment="testnet"
            ),
        ],
    )


# A two-chunk, two-question matrix: generators answer from chunk 1, the judge
# accepts every claim and covers every gold venue.
CHUNKS = {
    "b1": chunk("b1", "binance", "Binance: recvWindow max 60000."),
    "y1": chunk("y1", "bybit", "Bybit: recv_window default 5000."),
}


class TwoChunkRetriever:
    def retrieve(
        self,
        query: str,
        method: RetrievalMethod,
        *,
        k: int | None = None,
        venues_hint: Sequence[str] | None = None,
    ) -> list[Result]:
        order = ["b1", "y1"] if method == "venue_split" else ["y1", "b1"]
        return [Result.from_chunk(CHUNKS[c], 1.0) for c in order][:k]


def respond(url: str, body: dict[str, Any]) -> dict[str, Any]:
    if not body["model"].startswith("anthropic/"):
        answer = {
            "answer": "Binance allows 60000 ms [1].",
            "claims": [{"text": "max 60000", "venue": "binance", "chunks": [1]}],
            "refused": False,
            "asked_clarification": False,
            "partial": False,
            "missing": [],
        }
        return openrouter_payload(answer)
    message = body["messages"][-1]["content"]
    judged_claims = message.count("-- cites chunks")
    venue_line = next(line for line in message.splitlines() if line.startswith("Venues with gold"))
    gold_venues = [v.strip() for v in venue_line.split(":", 1)[1].split(",") if v.strip() != "none"]
    return openrouter_payload(
        {
            "claims": [
                {"index": i, "supported": True, "venue_correct": True, "rationale": "ok"}
                for i in range(1, judged_claims + 1)
            ],
            "coverage": [
                {"venue": v, "gold_facts": 1, "covered_facts": 1, "rationale": ""}
                for v in gold_venues
            ],
            "behaviour_correct": True,
            "behaviour_rationale": "answered",
        },
        model=body["model"],
        provider="Anthropic",
    )


def config(tmp_path: Path) -> Config:
    gen = gen_config(
        tmp_path,
        top_k=2,
        generator_prompt="prompts/generator.md",
        judge_prompt="prompts/judge.md",
        matrix=[
            MatrixEntry(name="gemini@venue_split", generator="gemini", retriever="venue_split"),
            MatrixEntry(name="gemini@hybrid_rerank", generator="gemini", retriever="hybrid_rerank"),
        ],
        primary_pair=["gemini@venue_split", "gemini@hybrid_rerank"],
        primary_metrics=["venue_attribution_error", "coverage"],
    )
    return Config(generation=gen)


QUESTIONS = [
    Question(
        "cv-0001",
        "binance vs bybit recvWindow max?",
        "gold",
        ["binance", "bybit"],
        "comparison",
        category="comparison",
        gold_chunk_ids=["b1", "y1"],
    ),
    Question(
        "cv-0002",
        "binance recvWindow max?",
        "gold",
        ["binance"],
        "fact",
        category="fact",
        gold_chunk_ids=["b1"],
    ),
]


class Detector:
    """Detects binance when the text names it, and no other venue."""

    def detect(self, text: str) -> list[str]:
        return ["binance"] if "binance" in text.lower() else []


class DetectingRetriever:
    """Detects venues like the real retriever; every method returns the mainnet/testnet chunk.

    Records each call as (method, k).
    """

    venue_detector = Detector()

    def __init__(self) -> None:
        self.calls: list[tuple[str, int | None]] = []

    def retrieve(
        self,
        query: str,
        method: RetrievalMethod,
        *,
        k: int | None = None,
        venues_hint: Sequence[str] | None = None,
    ) -> list[Result]:
        self.calls.append((method, k))
        return [Result.from_chunk(mainnet_testnet_chunk(), 1.0)]


# The generator's answer to "binance testnet ip ban length?": one claim citing chunk 1.
ASK_ANSWER = {
    "answer": "On testnet, Binance bans scale from 2 minutes to 3 days [1].",
    "claims": [{"text": "bans scale 2m to 3d", "venue": "binance", "chunks": [1]}],
    "refused": False,
    "asked_clarification": False,
    "partial": False,
    "missing": [],
}


def ask_setup(
    tmp_path: Path, content: dict[str, Any] = ASK_ANSWER
) -> tuple[Config, LLM, DetectingRetriever, FakeTransport]:
    """What `ask` needs, offline: a config, a client whose model answers `content`, a retriever.

    The transport is returned too, to read the requests it received.
    """
    cfg = Config(generation=gen_config(tmp_path, generator_prompt="prompts/generator.md"))
    transport = FakeTransport(respond=lambda url, body: openrouter_payload(content))
    return cfg, LLM(cfg.generation, transport, api_key=SECRET), DetectingRetriever(), transport


def calibration_config(tmp_path: Path) -> Config:
    """`config` with results under tmp_path and 50 bootstrap resamples, for calibration runs."""
    cfg = config(tmp_path)
    cfg.paths.results = tmp_path / "results"
    cfg.eval.bootstrap_resamples = 50
    return cfg
