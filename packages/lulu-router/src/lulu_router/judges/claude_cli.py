"""ClaudeCLIJudge: v0 backend for the LLM-judge stopping criterion.

Shells out to the local `claude` CLI in headless mode (`claude -p ...
--model ...`), so judging runs on the Claude Code subscription with no
separate API key and no extra provider dependency for v0. It satisfies the
same `Judge` protocol as GeometricJudge, so it's a drop-in swap in every
strategy in strategies.py -- and it's deliberately built this way so it can
itself be swapped later for OllamaJudge / AnthropicAPIJudge /
OpenRouterJudge (see roadmap in docs/THESIS.md and the top-level plan)
without touching the router or any strategy.

The prompt lives in judges/prompts/sufficiency.md as a versioned file, not
inlined in this module, so it can be treated as a hyperparameter and
A/B-tested against the golden set the same way any of the six routing
strategies are (evals/run.py).
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from lulu_router.shard import SearchResult

PROMPT_PATH = Path(__file__).parent / "prompts" / "sufficiency.md"
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_TIMEOUT_S = 30.0
CANDIDATE_CONTENT_CHARS = 300


class ClaudeCLIJudge:
    name = "claude_cli"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        cli_path: str = "claude",
    ) -> None:
        self.model = model
        self.timeout_s = timeout_s
        self.cli_path = cli_path
        self._prompt_template = PROMPT_PATH.read_text(encoding="utf-8")

    def judge(
        self,
        query: str,
        results: list[SearchResult],
        sources_contacted: int,
        total_sources: int,
    ) -> tuple[float, bool, str]:
        prompt = self._build_prompt(query, results, sources_contacted, total_sources)
        raw = self._call_cli(prompt)
        return self._parse(raw)

    def _build_prompt(
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
        prompt = self._prompt_template
        prompt = prompt.replace("{{query}}", query)
        prompt = prompt.replace("{{candidates}}", candidates)
        prompt = prompt.replace("{{sources_contacted}}", str(sources_contacted))
        prompt = prompt.replace("{{total_sources}}", str(total_sources))
        return prompt

    def _call_cli(self, prompt: str) -> str:
        try:
            proc = subprocess.run(
                [self.cli_path, "-p", prompt, "--model", self.model],
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            # Fail open toward "keep expanding" rather than silently trusting
            # an empty result set -- an unavailable judge shouldn't look like
            # a confident "sufficient".
            return json.dumps(
                {"sufficient": False, "confidence": 0.0, "missing": f"judge unavailable: {exc}"}
            )
        return proc.stdout

    @staticmethod
    def _parse(raw: str) -> tuple[float, bool, str]:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return 0.0, True, f"unparseable judge output: {raw[:200]!r}"
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return 0.0, True, f"malformed JSON from judge: {raw[:200]!r}"
        confidence = float(data.get("confidence", 0.0))
        sufficient = bool(data.get("sufficient", confidence >= 0.6))
        missing = data.get("missing") or ""
        reasoning = f"llm: sufficient={sufficient} missing={missing!r}" if missing else "llm: sufficient"
        return confidence, not sufficient, reasoning
