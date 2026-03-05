"""
analyser.py — ThreatSense AI-DVR
Ollama VLM analyser. Uses /api/generate which is the correct endpoint
for cloud-routed models like qwen3-vl:235b-cloud.
"""
from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from typing import Optional

import httpx

# --------------------------------------------------------------------------- #
# Config                                                                        #
# --------------------------------------------------------------------------- #

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL      = "qwen3-vl:235b-cloud"   # matches installed model

# --------------------------------------------------------------------------- #
# Prompts                                                                       #
# --------------------------------------------------------------------------- #

_SYSTEM = """\
You are an expert AI security analyst embedded in a professional CCTV surveillance system.
You have deep expertise in behavioural threat assessment from video surveillance footage,
recognising criminal patterns (theft, assault, trespass, vandalism, intrusion, loitering),
and distinguishing genuine threats from benign activity (residents, delivery workers, pets, etc.).

CRITICAL RULES:
1. Base your verdict ONLY on what is visually evident in the provided images.
2. Do NOT hallucinate events not visible in the images.
3. When genuinely ambiguous, output BENIGN.
4. Be specific — describe what you SEE, not what you assume.
5. Output ONLY the JSON object — no markdown, no extra text.
"""

_USER_TMPL = """\
You are analysing {n} surveillance frame(s) extracted from a security camera{context_note}.
{frame_desc}

Study ALL frames carefully. Consider temporal continuity across frames.

ASSESSMENT CHECKLIST:
1. How many people are visible? What is the environment and time of day?
2. Describe each subject: appearance, posture, movement, objects they carry.
3. Identify any suspicious behaviours: forced entry, concealing items, running away, confrontational posture, loitering, looking around nervously.
4. Are there clear signs of: theft, assault, vandalism, trespass, intrusion, or weapons?

OUTPUT: Respond with ONLY this JSON object — no markdown, no explanation outside it:
{{"verdict": "THREAT" or "BENIGN", "type": "theft|assault|vandalism|trespass|intrusion|loitering|suspicious_package|other|none", "confidence": 0.00, "reason": "<one clear sentence summarising the headline finding>", "scene_description": "<2-3 sentences: environment, people, overall activity>", "behaviour_analysis": "<2-3 sentences: what subjects are doing and why suspicious or not>", "risk_factors": ["<risk factor 1>", "<risk factor 2>"], "recommended_action": "<what security operator should do>"}}

Rules:
- ALL fields must be present. risk_factors must be a JSON array (can be [] if BENIGN).
- Output THREAT only when confidence >= 0.60.
- confidence is a float 0.0-1.0.
"""

_CORRECTION = (
    "Your response was not valid JSON or was missing required fields. "
    "Reply ONLY with the complete JSON object. Example: "
    '{"verdict":"BENIGN","type":"none","confidence":0.3,'
    '"reason":"No threat visible.","scene_description":"Empty corridor.",'
    '"behaviour_analysis":"No suspicious movement.","risk_factors":[],'
    '"recommended_action":"No action needed."}'
)

# --------------------------------------------------------------------------- #
# Result                                                                        #
# --------------------------------------------------------------------------- #

@dataclass
class AnalysisResult:
    verdict:            str    # "THREAT" | "BENIGN"
    type:               str    # threat category
    confidence:         float  # 0.0 – 1.0
    reason:             str    # headline sentence
    scene_description:  str
    behaviour_analysis: str
    risk_factors:       list
    recommended_action: str
    raw_response:       str
    is_threat:          bool

    def to_dict(self) -> dict:
        return {
            "verdict":            self.verdict,
            "type":               self.type,
            "confidence":         self.confidence,
            "reason":             self.reason,
            "scene_description":  self.scene_description,
            "behaviour_analysis": self.behaviour_analysis,
            "risk_factors":       self.risk_factors,
            "recommended_action": self.recommended_action,
            "is_threat":          self.is_threat,
        }


# --------------------------------------------------------------------------- #
# Analyser                                                                      #
# --------------------------------------------------------------------------- #

class OllamaAnalyser:
    """
    Send images to Ollama using /api/generate (works for both local and
    cloud-routed models like qwen3-vl:235b-cloud).

    Parameters
    ----------
    ollama_url : Ollama base URL (default http://localhost:11434)
    model      : model tag to use
    timeout    : HTTP read timeout in seconds
    """

    def __init__(
        self,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = 300.0,
    ):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self._timeout = httpx.Timeout(connect=10.0, read=timeout, write=60.0, pool=10.0)

    # ------------------------------------------------------------------ #
    # Public                                                               #
    # ------------------------------------------------------------------ #

    async def analyse_images(
        self,
        images: list[bytes],
        context: str = "",
        frame_info: str = "",
    ) -> Optional[AnalysisResult]:
        """
        Analyse a list of JPEG images via Ollama /api/generate.

        Parameters
        ----------
        images     : list of raw JPEG bytes
        context    : optional text hint, e.g. "person detected by YOLO at entrance"
        frame_info : optional description of frame sampling, e.g. "sampled at 2 FPS"
        """
        if not images:
            return None

        images_b64 = [base64.b64encode(img).decode() for img in images]
        context_note = f" with context: {context}" if context else ""
        frame_desc   = f"Frame info: {frame_info}" if frame_info else ""

        # Build full prompt including system instructions
        full_prompt = (
            _SYSTEM + "\n\n" +
            _USER_TMPL.format(
                n=len(images),
                context_note=context_note,
                frame_desc=frame_desc,
            )
        )

        raw = await self._generate(full_prompt, images_b64)
        if raw is None:
            return None

        result = _parse(raw)
        if result:
            return result

        # One correction retry
        retry_prompt = (
            full_prompt + "\n\nPrevious response:\n" + raw +
            "\n\n" + _CORRECTION
        )
        raw2 = await self._generate(retry_prompt, images_b64)
        if raw2 is None:
            return None
        return _parse(raw2)

    async def health_check(self) -> tuple[bool, str]:
        """Returns (ok: bool, message: str). Checks Ollama reachability and model availability."""
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(8.0)) as client:
                resp = await client.get(f"{self.ollama_url}/api/tags")
                if resp.status_code != 200:
                    return False, f"Ollama returned HTTP {resp.status_code}"
                data   = resp.json()
                models = [m["name"] for m in data.get("models", [])]
                if not any(self.model in m for m in models):
                    avail = ", ".join(models) if models else "none"
                    return False, f"Model '{self.model}' not found. Available: {avail}"
                return True, f"OK — model '{self.model}' is ready"
        except httpx.ConnectError:
            return False, f"Cannot connect to Ollama at {self.ollama_url}"
        except Exception as exc:
            return False, str(exc)

    # ------------------------------------------------------------------ #
    # Internal — uses /api/generate (works for cloud models)              #
    # ------------------------------------------------------------------ #

    async def _generate(self, prompt: str, images_b64: list[str]) -> Optional[str]:
        """
        POST to /api/generate — the correct endpoint for qwen3-vl:235b-cloud.
        Passes images as base64 strings in the 'images' array field.
        """
        payload = {
            "model":  self.model,
            "prompt": prompt,
            "images": images_b64,
            "stream": False,
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json=payload,
                )
                resp.raise_for_status()
                return resp.json().get("response", "")
        except httpx.TimeoutException:
            return None
        except Exception:
            return None


# --------------------------------------------------------------------------- #
# JSON parsing (tolerant)                                                       #
# --------------------------------------------------------------------------- #

_JSON_RE    = re.compile(r"\{.*\}", re.DOTALL)
_VALID_TYPES = {
    "theft", "assault", "vandalism", "trespass", "intrusion",
    "loitering", "suspicious_package", "other", "none"
}


def _parse(raw: str) -> Optional[AnalysisResult]:
    # Strip markdown code fences
    text = re.sub(r"```[a-z]*\n?", "", raw).strip()

    for candidate in [text] + [m.group() for m in _JSON_RE.finditer(text)]:
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

        risk_factors = obj.get("risk_factors", [])
        if not isinstance(risk_factors, list):
            risk_factors = [str(risk_factors)] if risk_factors else []

        return AnalysisResult(
            verdict=verdict,
            type=threat_type,
            confidence=confidence,
            reason=str(obj.get("reason", "")).strip(),
            scene_description=str(obj.get("scene_description", "")).strip(),
            behaviour_analysis=str(obj.get("behaviour_analysis", "")).strip(),
            risk_factors=risk_factors,
            recommended_action=str(obj.get("recommended_action", "")).strip(),
            raw_response=raw,
            is_threat=(verdict == "THREAT"),
        )
    return None
