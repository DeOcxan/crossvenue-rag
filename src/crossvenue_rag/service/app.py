"""FastAPI app: POST /answer returns what `crossvenue ask` prints, as JSON.

Two JSON endpoints over `generation.ask.answer_question`, POST /answer and
GET /health, with no UI.
"""

from __future__ import annotations

import os
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from ..config import Config
from ..config_loading import DEFAULT_CONFIG, load_config
from ..errors import ExternalFailure, UserError, require_file
from ..generation.api_key import load_api_key
from ..generation.ask import DEFAULT_GENERATOR, DEFAULT_RETRIEVER, AskResult, answer_question
from ..generation.llm import LLM, BudgetExceeded
from ..retrieval.protocols import DetectingRanks
from ..retrieval.retriever import Retriever

# Decimals of the spend /health reports, in USD: a hundredth of a cent.
SPENT_DECIMALS = 4


class AnswerRequest(BaseModel):
    """The body of POST /answer: the question, the retrieval method and the generator.

    Any other field is refused (422), so a misspelled `generator` is not answered by the
    default one.
    """

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    # One of `retrieval.methods.AD_HOC_METHODS`; checked by `answer_question`.
    retriever: str = DEFAULT_RETRIEVER
    # One of generation.generators; checked against the config per request.
    generator: str = DEFAULT_GENERATOR


class HealthResponse(BaseModel):
    """The body of GET /health: whether the service can answer, and with what.

    `answering` is "available" or why it is not; the budget fields are None
    without a model client.
    """

    status: Literal["ok"] = "ok"
    answering: str
    generators: list[str]
    top_k: int | None
    budget_usd: float | None
    spent_usd: float | None


class _Service:
    """What the app answers from: the config, the retriever and the model client.

    Any of them not given is loaded at startup. Without an API key the app still
    starts; `/answer` then says why it cannot answer.
    """

    def __init__(
        self, cfg: Config | None, retriever: DetectingRanks | None, llm: LLM | None
    ) -> None:
        self.cfg = cfg
        self.retriever = retriever
        self.llm = llm
        self.unavailable: str | None = None
        # FastAPI runs `answer` in a thread pool; one request at a time uses the retriever,
        # the model client, its budget and its cache.
        self._lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(self, _app: FastAPI) -> AsyncIterator[None]:
        """Loads what was not given at startup and closes the retriever it opened at shutdown."""
        owned = self._load()
        yield
        if owned is not None:
            owned.close()

    def _load(self) -> Retriever | None:
        """Loads what was not given; returns the retriever it opened, to close at shutdown."""
        owned: Retriever | None = None
        if self.cfg is None:
            path = os.environ.get("CROSSVENUE_CONFIG", str(DEFAULT_CONFIG))
            require_file(path, hint="set CROSSVENUE_CONFIG to the path of a config file")
            self.cfg = load_config(path)
        if self.retriever is None:
            owned = Retriever(self.cfg)
            owned.require_current_index()
            owned.warmup()
            self.retriever = owned
        if self.llm is None:
            try:
                self.llm = LLM(self.cfg.generation, api_key=load_api_key())
            except UserError as exc:
                self.unavailable = exc.with_hint()
        return owned

    def health(self) -> HealthResponse:
        """Whether answering is available, the generators, `top_k` and the budget spent."""
        cfg, llm = self.cfg, self.llm
        return HealthResponse(
            answering=self.unavailable or "available",
            generators=sorted(cfg.generation.generators) if cfg else [],
            top_k=cfg.generation.top_k if cfg else None,
            budget_usd=llm.budget_usd if llm else None,
            spent_usd=round(llm.spent_usd, SPENT_DECIMALS) if llm else None,
        )

    def answer(self, request: AnswerRequest) -> AskResult:
        """What `ask` returns for the request; HTTP 503 while answering is unavailable."""
        if self.llm is None:
            raise HTTPException(
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                detail=self.unavailable or "no model client",
            )
        if self.cfg is None or self.retriever is None:
            # Startup loads both: only an app that never started lacks one.
            raise HTTPException(
                status_code=HTTPStatus.SERVICE_UNAVAILABLE, detail="the service has not started"
            )
        try:
            with self._lock:
                return answer_question(
                    self.cfg,
                    self.retriever,
                    self.llm,
                    request.question,
                    request.retriever,
                    request.generator,
                )
        except BudgetExceeded as exc:
            raise HTTPException(
                status_code=HTTPStatus.SERVICE_UNAVAILABLE, detail=exc.with_hint()
            ) from exc
        # A model call that failed or a reply that failed its check: the model is at fault.
        except ExternalFailure as exc:
            raise HTTPException(status_code=HTTPStatus.BAD_GATEWAY, detail=exc.with_hint()) from exc
        # What the request itself can cause: an empty question, an unknown generator.
        except UserError as exc:
            raise HTTPException(
                status_code=HTTPStatus.UNPROCESSABLE_ENTITY, detail=exc.with_hint()
            ) from exc


def create_app(
    cfg: Config | None = None, retriever: DetectingRanks | None = None, llm: LLM | None = None
) -> FastAPI:
    """The app, answering from the config, retriever and model client given.

    Any of them not given is built once, at startup, from the config in
    `CROSSVENUE_CONFIG` (default configs/default.yaml); tests pass fakes and
    need no network. Without the OpenRouter key the service starts, `GET
    /health` says answering is unavailable, and every `POST /answer` is a 503.
    The model client is one run, and its budget (`generation.budget_usd`)
    covers every paid call the process makes. Once it is spent, every request
    that needs a paid call is a 503 until a restart, and an answer in the cache
    is still served. `POST /answer` returns 422 for a request the service
    cannot answer as asked (an unknown generator or retriever, an empty
    question), 502 for a model call that failed, 503 for a missing key or a
    spent budget.
    """
    service = _Service(cfg, retriever, llm)
    app = FastAPI(title="crossvenue-rag", lifespan=service.lifespan)
    app.get("/health", response_model=HealthResponse)(service.health)
    app.post("/answer", response_model=AskResult)(service.answer)
    return app


def app_factory() -> FastAPI:
    """Entry point for `uvicorn --factory`."""
    return create_app()
