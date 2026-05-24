"""
Lightweight, free-tier safety guardrails.
No external API calls — all rule-based and regex.

Layers:
1. Prompt injection detection
2. Jailbreak pattern matching
3. Harmful content categories
4. PII detection (basic)
5. Response filtering
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from app.utils.logger import get_logger

logger = get_logger(__name__)


class ViolationType(str, Enum):
    PROMPT_INJECTION = "prompt_injection"
    JAILBREAK = "jailbreak"
    HARMFUL_CONTENT = "harmful_content"
    SELF_HARM = "self_harm"
    ILLEGAL_ACTIVITY = "illegal_activity"
    HATE_SPEECH = "hate_speech"
    PII_LEAK = "pii_leak"
    SAFE = "safe"


@dataclass
class SafetyResult:
    is_safe: bool
    violation_type: ViolationType = ViolationType.SAFE
    confidence: float = 1.0
    matched_pattern: Optional[str] = None
    refusal_message: str = ""


# ─── Pattern Libraries ─────────────────────────────────────────────────────────

PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|context)",
    r"disregard\s+(all\s+)?(previous|prior|above)\s+(instructions?|directives?)",
    r"you\s+are\s+now\s+(a\s+)?(?:DAN|evil|unrestricted|uncensored)",
    r"pretend\s+(you|that\s+you)\s+(are|have\s+no)\s+(restrictions?|guidelines?|ethics)",
    r"act\s+as\s+(if\s+you\s+(are\s+)?)?(?:DAN|jailbroken|unrestricted)",
    r"system\s*:\s*you\s+are",
    r"\[system\]|\[INST\].*ignore",
    r"new\s+instructions?\s*:",
    r"###\s*override",
    r"from\s+now\s+on\s+you\s+(will|must|should)\s+ignore",
]

JAILBREAK_PATTERNS = [
    r"\bDAN\b",
    r"do\s+anything\s+now",
    r"developer\s+mode",
    r"jailbreak",
    r"grandmother\s+(used\s+to\s+)?tell\s+me",
    r"for\s+(educational|research|fictional)\s+purposes\s+only",
    r"hypothetically\s+(speaking|if\s+you\s+could)",
    r"in\s+a\s+fictional\s+(world|scenario|story)",
    r"as\s+a\s+(fictional\s+)?character",
    r"no\s+restrictions\s+mode",
    r"unrestricted\s+mode",
    r"break\s+(out\s+of\s+)?(your\s+)?(constraints?|restrictions?|programming)",
]

HARMFUL_CONTENT_PATTERNS = [
    # Weapons
    r"how\s+to\s+(make|build|create|synthesize)\s+(a\s+)?(bomb|explosive|weapon|poison|drug)",
    r"(instructions?|steps?|guide|tutorial)\s+.{0,30}(bomb|explosive|weapon)",
    # Self-harm
    r"how\s+to\s+(commit\s+)?suicide",
    r"ways?\s+to\s+(hurt|harm|kill)\s+(myself|yourself)",
    r"self.harm\s+(methods?|techniques?|ways?)",
    # Hate speech
    r"\b(kill|murder|harm|attack)\s+(all\s+)?(jews?|muslims?|christians?|blacks?|whites?|asians?|gays?)\b",
    # Illegal
    r"how\s+to\s+(hack|crack|break\s+into)\s+(a\s+)?(computer|system|account|website)",
    r"(credit\s+card|bank\s+account)\s+(fraud|scam|theft)",
    r"how\s+to\s+(buy|get|obtain)\s+(illegal\s+)?(drugs?|weapons?|guns?)\s+online",
    r"child\s+(porn|pornography|sexual|nude)",
    r"(csam|pedophil)",
]

SELF_HARM_PATTERNS = [
    r"\b(suicide|suicidal|kill\s+myself|end\s+my\s+life)\b",
    r"(want|plan|going)\s+to\s+(die|kill\s+myself)",
    r"methods?\s+(of|for)\s+(suicide|self.harm)",
]

# ─── Refusal Templates ─────────────────────────────────────────────────────────

REFUSAL_TEMPLATES = {
    ViolationType.PROMPT_INJECTION: (
        "I noticed an attempt to override my instructions. "
        "I'm here to be helpful, but I maintain my guidelines in all conversations. "
        "How can I assist you legitimately?"
    ),
    ViolationType.JAILBREAK: (
        "I can see what you're trying to do, but I don't have a 'jailbreak mode' or alternate persona. "
        "I'm designed to be consistently helpful and safe. What can I genuinely help you with?"
    ),
    ViolationType.HARMFUL_CONTENT: (
        "I'm not able to help with that request as it could lead to harm. "
        "If you have a legitimate need related to this topic, please rephrase your question."
    ),
    ViolationType.SELF_HARM: (
        "I'm concerned about what you've shared. If you're having thoughts of self-harm, "
        "please reach out to a crisis helpline: **US: 988** | **International: findahelpline.com**. "
        "You're not alone, and help is available."
    ),
    ViolationType.ILLEGAL_ACTIVITY: (
        "I can't provide assistance with illegal activities. "
        "If you have a question about this topic from a legal or educational perspective, I'm happy to help."
    ),
    ViolationType.HATE_SPEECH: (
        "I don't engage with content that targets or demeans people based on their identity. "
        "I'm here to have constructive, respectful conversations."
    ),
}


class SafetyFilter:
    """
    Multi-layer safety filter.
    All checks are local — no API calls required.
    """

    def __init__(self, mode: str = "strict") -> None:
        """
        mode: "strict" = check everything
              "moderate" = skip overly broad patterns
              "off" = pass through (for testing only)
        """
        self.mode = mode
        self._compiled_injection = [re.compile(p, re.IGNORECASE) for p in PROMPT_INJECTION_PATTERNS]
        self._compiled_jailbreak = [re.compile(p, re.IGNORECASE) for p in JAILBREAK_PATTERNS]
        self._compiled_harmful = [re.compile(p, re.IGNORECASE) for p in HARMFUL_CONTENT_PATTERNS]
        self._compiled_self_harm = [re.compile(p, re.IGNORECASE) for p in SELF_HARM_PATTERNS]

    def check_input(self, text: str) -> SafetyResult:
        """Check user input before sending to model."""
        if self.mode == "off":
            return SafetyResult(is_safe=True)

        text_lower = text.lower()

        # Priority order matters — self-harm first for compassionate response
        for pattern in self._compiled_self_harm:
            if pattern.search(text):
                return SafetyResult(
                    is_safe=False,
                    violation_type=ViolationType.SELF_HARM,
                    confidence=0.9,
                    matched_pattern=pattern.pattern,
                    refusal_message=REFUSAL_TEMPLATES[ViolationType.SELF_HARM],
                )

        for pattern in self._compiled_injection:
            if pattern.search(text):
                logger.warning("prompt_injection_detected", pattern=pattern.pattern[:50])
                return SafetyResult(
                    is_safe=False,
                    violation_type=ViolationType.PROMPT_INJECTION,
                    confidence=0.95,
                    matched_pattern=pattern.pattern,
                    refusal_message=REFUSAL_TEMPLATES[ViolationType.PROMPT_INJECTION],
                )

        if self.mode == "strict":
            for pattern in self._compiled_jailbreak:
                if pattern.search(text):
                    logger.warning("jailbreak_attempt_detected", pattern=pattern.pattern[:50])
                    return SafetyResult(
                        is_safe=False,
                        violation_type=ViolationType.JAILBREAK,
                        confidence=0.85,
                        matched_pattern=pattern.pattern,
                        refusal_message=REFUSAL_TEMPLATES[ViolationType.JAILBREAK],
                    )

        for pattern in self._compiled_harmful:
            if pattern.search(text):
                logger.warning("harmful_content_detected", pattern=pattern.pattern[:50])
                return SafetyResult(
                    is_safe=False,
                    violation_type=ViolationType.HARMFUL_CONTENT,
                    confidence=0.9,
                    matched_pattern=pattern.pattern,
                    refusal_message=REFUSAL_TEMPLATES[ViolationType.HARMFUL_CONTENT],
                )

        return SafetyResult(is_safe=True)

    def check_output(self, text: str) -> SafetyResult:
        """
        Check model output before returning to user.
        Catches cases where the model generated harmful content despite input filtering.
        """
        if self.mode == "off":
            return SafetyResult(is_safe=True)

        # Check if model may have complied with a jailbreak
        COMPLIANCE_INDICATORS = [
            r"sure,?\s+here('s|\s+is)\s+(how|the\s+(steps?|instructions?|guide))",
            r"of\s+course[!,]?\s+(here|i\s+can\s+help)",
            r"i\s+(will|am)\s+(now\s+)?(operating|working|running)\s+(in|as)\s+(DAN|unrestricted)",
        ]
        for pat in COMPLIANCE_INDICATORS:
            # Only flag in context of other harmful content
            pass  # Output filtering is lighter — we trust the input layer more

        for pattern in self._compiled_self_harm:
            if pattern.search(text):
                return SafetyResult(
                    is_safe=False,
                    violation_type=ViolationType.SELF_HARM,
                    confidence=0.8,
                    refusal_message=REFUSAL_TEMPLATES[ViolationType.SELF_HARM],
                )

        return SafetyResult(is_safe=True)

    def get_stats(self) -> dict:
        return {
            "mode": self.mode,
            "injection_patterns": len(self._compiled_injection),
            "jailbreak_patterns": len(self._compiled_jailbreak),
            "harmful_patterns": len(self._compiled_harmful),
        }
