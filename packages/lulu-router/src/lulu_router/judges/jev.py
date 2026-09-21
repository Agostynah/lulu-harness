"""JevJudge: a content-reading judge backed by TypeSafe AI's Jev model.

Same slot as ClaudeCLIJudge -- both satisfy the `Judge` protocol and read
*what* came back, not just score geometry (that's GeometricJudge's job).
The difference is what's underneath: ClaudeCLIJudge shells out to a full
chat model per turn; Jev is a "System 1" evaluation model built for exactly
this shape of call -- a state plus one typed question, answered as a
calibrated probability instead of free text that then has to be parsed.
That fits the router's own cost-aware premise (docs/THESIS.md) better than
either existing judge: cheaper and faster than an LLM call, but still
reads candidate content unlike GeometricJudge's pure score-gap heuristic.

No JSON-parsing-from-prose step either (contrast ClaudeCLIJudge._parse) --
Jev's `noul` primitive returns the probability directly, so sufficiency
*is* the confidence, not something extracted from it.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from lulu_router.shard import SearchResult

API_URL = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"
DEFAULT_TIMEOUT_S = 10.0
CANDIDATE_CONTENT_CHARS = 300
SUFFICIENT_INSTRUCTIONS = (
    "Given the query and the candidate memory shards retrieved so far, is this "
    "enough to fully answer the query without contacting more shards?"
)


class JevJudge:
    name = "jev"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        # Set on every judge() call -- lets FallbackJudge (judges/fallback.py)
        # tell "Jev genuinely couldn't be reached" apart from "Jev answered
        # and the honest answer was low confidence", which must NOT trigger
        # a fallback (it's a real verdict, not a failure).
        self.available = True

    def judge(
        self,
        query: str,
        results: list[SearchResult],
        sources_contacted: int,
        total_sources: int,
    ) -> tuple[float, bool, str]:
        state = self._build_state(query, results, sources_contacted, total_sources)
        answer = self._call_api(state)
        self.available = _has_real_verdict(answer)
        return self._parse(answer)

    def _build_state(
        self,
        query: str,
        results: list[SearchResult],
        sources_contacted: int,
        total_sources: int,
    ) -> str:
        candidates = (
            "\n".join(f"- ({r.score:.3f}) {r.content[:CANDIDATE_CONTENT_CHARS]}" for r in results)
            or "(no candidates retrieved yet)"
        )
        return (
            f"query: {query}\n"
            f"shards contacted: {sources_contacted}/{total_sources}\n"
            f"candidates:\n{candidates}"
        )

    def _call_api(self, state: str) -> dict | None:
        body = json.dumps(
            {
                "state": state,
                "model": self.model,
                "questions": {
                    "sufficient": {
                        "type": "noul",
                        "instructions": SUFFICIENT_INSTRUCTIONS,
                    }
                },
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            API_URL,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            # Fail open toward "keep expanding" -- an unavailable judge
            # shouldn't look like a confident "sufficient" (same policy as
            # ClaudeCLIJudge for the same reason). FallbackJudge is what
            # actually recovers a real verdict when this happens; standalone
            # JevJudge use (no fallback wrapping) still degrades safely.
            return None

    @staticmethod
    def _parse(answer: dict | None) -> tuple[float, bool, str]:
        if answer is None:
            return 0.0, True, "jev: judge unavailable"
        try:
            confidence = float(answer["answers"]["sufficient"]["noul"])
        except (KeyError, TypeError, ValueError):
            return 0.0, True, f"jev: malformed response {answer!r}"
        should_expand = confidence < 0.5
        reasoning = f"jev: sufficient={confidence:.3f}"
        return confidence, should_expand, reasoning


def _has_real_verdict(answer: dict | None) -> bool:
    if answer is None:
        return False
    try:
        float(answer["answers"]["sufficient"]["noul"])
    except (KeyError, TypeError, ValueError):
        return False
    return True
