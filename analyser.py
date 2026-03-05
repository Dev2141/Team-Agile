"""
Standalone Ollama VLM analyser.
No dependencies on the DVR app. Zero imports from the parent project.

Usage (programmatic):
    import asyncio
    from analyser import OllamaAnalyser, AnalysisResult

    analyser = OllamaAnalyser()
    result = asyncio.run(analyser.analyse_images([jpeg_bytes_1, jpeg_bytes_2]))
    print(result.verdict, result.reason)
"""
from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from typing import Optional

import httpx

# --------------------------------------------------------------------------- #
# Config (override via env vars or pass to constructor)                        #
# --------------------------------------------------------------------------- #

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL      = "qwen3-vl:235b-cloud"

# --------------------------------------------------------------------------- #
# Prompt                                                                       #
# --------------------------------------------------------------------------- #

_SYSTEM = (
    "You are a security camera AI. "
    "You will be shown images from a security camera and must assess "
    "whether there is a genuine threat."
)

_USER_TMPL = """\
You are given {n} image(s) from a security camera{context_note}.

Look carefully at all images. Determine whether the scene shows a genuine threat
(theft, assault, vandalism, trespass, suspicious intrusion) or is benign
(resident, delivery person, false alarm, pet, etc.).

Respond with ONLY a single valid JSON object — no markdown, no explanation outside it:
{{"verdict": "THREAT" or "BENIGN", "type": "theft|assault|vandalism|trespass|intrusion|none", "confidence": 0.0-1.0, "reason": "<one concise sentence>"}}

Rules:
- Output THREAT only when you are confident there is malicious or dangerous intent.
- When ambiguous, output BENIGN.
- confidence must be a float 0.0–1.0.
"""

_CORRECTION = (
    "Your response was not valid JSON. Reply ONLY with the JSON object. "
    'Example: {"verdict":"BENIGN","type":"none","confidence":0.2,"reason":"No threat."}'
)

# --------------------------------------------------------------------------- #
# Result                                                                       #
# --------------------------------------------------------------------------- #

@dataclass
class AnalysisResult:
    verdict:      str    # "THREAT" | "BENIGN"
    type:         str    # "theft" | "assault" | "vandalism" | "trespass" | "intrusion" | "none"
    confidence:   float  # 0.0 – 1.0
    reason:       str
    raw_response: str
    is_threat:    bool

    def to_dict(self) -> dict:
        return {
            "verdict":    self.verdict,
            "type":       self.type,
            "confidence": self.confidence,
            "reason":     self.reason,
            "is_threat":  self.is_threat,
        }

# --------------------------------------------------------------------------- #
# Analyser                                                                     #
# --------------------------------------------------------------------------- #

class OllamaAnalyser:
    """
    Send images to Ollama Qwen-VL and parse the structured verdict.

    Parameters
    ----------
    ollama_url : Ollama base URL (default http://localhost:11434)
    model      : model tag to use
    timeout    : HTTP read timeout in seconds (large model → long timeout)
    """

    def __init__(
        self,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = 300.0,
    ):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self._timeout = httpx.Timeout(connect=10.0, read=timeout, write=30.0, pool=10.0)

    # ------------------------------------------------------------------ #
    # Public methods                                                       #
    # ------------------------------------------------------------------ #

    async def analyse_images(
        self,
        images: list[bytes],
        context: str = "",
    ) -> Optional[AnalysisResult]:
        """
        Analyse a list of JPEG images.

        Parameters
        ----------
        images  : list of raw JPEG bytes
        context : optional text context, e.g. "person and bicycle detected"

        Returns None on network failure or persistent JSON parse failure.
        """
        if not images:
            return None

        images_b64 = [base64.b64encode(img).decode() for img in images]
        context_note = f" ({context})" if context else ""

        messages = [
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": _USER_TMPL.format(n=len(images), context_note=context_note),
                "images": images_b64,
            },
        ]

        raw = await self._chat(messages)
        if raw is None:
            return None

        result = _parse(raw)
        if result:
            return result

        # One retry with correction prompt
        messages += [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": _CORRECTION},
        ]
        raw2 = await self._chat(messages)
        if raw2 is None:
            return None
        return _parse(raw2)

    async def health_check(self) -> tuple[bool, str]:
        """
        Returns (ok: bool, message: str).
        Checks that Ollama is reachable and the model is listed.
        """
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                resp = await client.get(f"{self.ollama_url}/api/tags")
                if resp.status_code != 200:
                    return False, f"Ollama returned HTTP {resp.status_code}"
                data = resp.json()
                models = [m["name"] for m in data.get("models", [])]
                if not any(self.model in m for m in models):
                    return (
                        False,
                        f"Model '{self.model}' not found. Available: {', '.join(models) or 'none'}",
                    )
                return True, f"OK — model '{self.model}' is available"
        except httpx.ConnectError:
            return False, "Cannot connect to Ollama at " + self.ollama_url
        except Exception as exc:
            return False, str(exc)

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    async def _chat(self, messages: list[dict]) -> Optional[str]:
        payload = {"model": self.model, "messages": messages, "stream": False}
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(f"{self.ollama_url}/api/chat", json=payload)
                resp.raise_for_status()
                return resp.json()["message"]["content"]
        except httpx.TimeoutException:
            return None
        except Exception:
            return None


# --------------------------------------------------------------------------- #
# JSON parsing (tolerant)                                                      #
# --------------------------------------------------------------------------- #

_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)
_VALID_TYPES = {"theft", "assault", "vandalism", "trespass", "intrusion", "none"}


def _parse(raw: str) -> Optional[AnalysisResult]:
    text = re.sub(r"```[a-z]*\n?", "", raw).strip()

    for candidate in [text, *[m.group() for m in _JSON_RE.finditer(text)]]:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        verdict = str(obj.get("verdict", "")).upper()
        if verdict not in ("THREAT", "BENIGN"):
            continue

        threat_type = str(obj.get("type", "none")).lower()
        if threat_type not in _VALID_TYPES:
            threat_type = "none"

        try:
            confidence = float(obj.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.5

        return AnalysisResult(
            verdict=verdict,
            type=threat_type,
            confidence=confidence,
            reason=str(obj.get("reason", "")).strip(),
            raw_response=raw,
            is_threat=(verdict == "THREAT"),
        )
    return None
