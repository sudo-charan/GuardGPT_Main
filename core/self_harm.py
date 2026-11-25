"""Tiered self-harm risk assessment with conversation carry-over.

Why this exists
---------------
The trained intent head only saw templated crisis sentences ("I feel like I
cannot cope anymore ...") in training, so everyday phrasing such as "everyone
would be happier if I wasn't around" or "I think I'm going to act on these
thoughts now" is scored ``benign``. A keyword list of exact phrases cannot
cover natural language either, and a single message is often only alarming in
the light of the previous ones. This module therefore assesses *risk level*,
not a label, from patterns plus the levels of the session's previous turns.

Levels (lowest to highest)
--------------------------
``none``      nothing of concern.
``distress``  low mood, loneliness, stress, hopelessness. Normal answer, but in
              a warm conversational "wellbeing" mode.
``passive``   passive ideation ("wish I could disappear", "everyone would be
              better off without me"). Wellbeing mode plus a gentle safety
              check-in; the session is marked at-risk.
``crisis``    explicit intent, plan, imminent risk or a follow-up to earlier
              risk ("I'm going to act on these thoughts"). Crisis response.

The module is pure and deterministic: same text and history, same result.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

__all__ = ["Assessment", "assess", "LEVELS", "AT_RISK_LEVELS", "RISK_WINDOW"]

LEVELS = ("none", "distress", "passive", "crisis")
AT_RISK_LEVELS = frozenset({"passive", "crisis"})
# How many previous turns count as "recent" for carry-over.
RISK_WINDOW = 6

_RANK = {name: index for index, name in enumerate(LEVELS)}


@dataclass(frozen=True)
class Assessment:
    level: str = "none"
    reasons: tuple[str, ...] = ()
    # "carry_over" (earlier risk still active), "retraction" (user is walking
    # back an earlier disclosure) or None.
    follow_up: str | None = None

    @property
    def is_crisis(self) -> bool:
        return self.level == "crisis"

    @property
    def at_risk(self) -> bool:
        return self.level in AT_RISK_LEVELS


def _normalise(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text)).lower()
    for quote in ("\u2018", "\u2019", "`", "\u00b4"):
        value = value.replace(quote, "'")
    return " ".join(value.split())


def _compile(*patterns: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(pattern) for pattern in patterns)


# --- explicit / imminent ----------------------------------------------------
_CRISIS = _compile(
    r"\b(?:kill|killing|hurt|hurting|harm|harming|hang|hanging|cut|cutting|poison|drown|shoot|stab|starve)"
    r" (?:myself|my ?self)\b",
    r"\bsuicid(?:e|al)\b.{0,30}\b(?:thoughts?|ideation|urges?|plan|note|attempt)\b",
    r"\b(?:i am|i'm|im|feel|feeling|felt|been|getting|am getting|i get|so|really|actually) (?:\w+ )?suicidal\b",
    r"\b(?:thinking (?:about|of)|think about|considering|contemplating|planning|plan to|going to|gonna|want to|wanna|about to|attempt(?:ed|ing)?|commit(?:ting)?) (?:a )?suicide\b",
    r"\b(?:end|ending|take|taking) (?:my|my own) (?:own )?(?:life|existence)\b",
    r"\bend it all\b(?! (?:by|with|using|in|via|before|after)\b)",
    r"\b(?:want|wanted|wanna|wish) (?:to )?(?:die|be dead)\b(?! (?:laughing|of|from|inside a little)\b)",
    r"\bwish i (?:was|were) dead\b",
    r"\b(?:better off dead|rather be dead|rather die)\b",
    r"\b(?:don't|do not|no longer|dont) want to (?:live|be alive|be here|exist|go on|wake up)\b",
    r"\bno (?:reason|point) (?:to|in) (?:live|living|go on|going on|continue|continuing|keep going|be alive)\b",
    r"\bdon't see (?:any|a) (?:reason|point) (?:to|in) (?:live|living|continue|continuing|go on|going on|anything)\b",
    r"\bnot worth living\b",
    r"\b(?:just |only )?want(?:ed)? (?:it|everything|all of this|all this|this|my life|my pain) (?:to )?(?:end|stop|be over|finish)\b",
    r"\b(?:act|acting|acted) on (?:these|those|my|the) (?:thoughts|feelings|urges|plans?)\b",
    r"(?:\bnot|n't) (?:feel )?safe (?:with|around|from) myself\b",
    r"\b(?:i (?:am|'m|will|'ll)|i'm) going to (?:do it|end it|end things|go through with it)\b",
    r"\b(?:suicide note|goodbye letter|goodbye note|my last message|my final message|say goodbye to everyone)\b",
    r"\b(?:overdose|od) (?:on|myself)\b",
)

# "please don't stop me" is only alarming together with ideation words.
_STOP_ME = re.compile(r"\b(?:don't|do not|dont|please don't|no one can|nobody can|no one will|nobody will) (?:try to )?stop me\b(?! from\b)")
_STOP_CONTEXT = re.compile(
    r"\b(?:thoughts|act on|end|ending|do it|goodbye|everything|my life|die|dying|disappear|hurt|pain)\b"
)

# --- passive ideation -------------------------------------------------------
_PASSIVE = _compile(
    r"\b(?:happier|better|better off|fine|easier|relieved)(?: off)? (?:without me|if i (?:wasn't|was not|weren't|were not|am not|wasn't around|didn't exist|did not exist|disappeared|was gone|were gone|vanished|wasn't here))\b",
    r"\bwish i (?:could )?(?:just )?(?:disappear|vanish|go away|wasn't (?:here|born|alive|around)|was never born|never existed|didn't exist|could sleep forever|could stop existing|could fall asleep and not wake)",
    r"\b(?:can't|cannot|couldn't) (?:go on|keep (?:going|living))\b(?! (?:with|without|like this tool|coding))",
    r"\b(?:tired of|sick of|done with|exhausted by) (?:living|life|being alive|everything)\b",
    r"\bgive up on (?:life|everything|living)\b",
    r"\bwhat's the point (?:of|in) (?:living|going on|anything|life|it all|trying)\b",
    r"\bno point (?:in )?(?:living|going on|anymore|life)\b",
    r"\bi(?:'m| am) (?:just |such )?(?:a )?(?:burden|waste of space)\b",
    r"\b(?:burden (?:to|on) (?:everyone|everybody|my family|them|others|my friends))\b",
    r"\bnobody (?:would|will) (?:notice|miss me|care if i)\b",
    r"\bnobody would care if i\b",
    r"\bdisappear (?:forever|for good|completely|from (?:everyone|everything|the world))\b",
)

# --- distress ---------------------------------------------------------------
_DISTRESS = _compile(
    r"\b(?:hopeless\w*|worthless\w*|useless|lonely|loneliness|isolated|overwhelm\w*|depress\w*|anxious|anxiety|panic\w*)\b",
    r"\b(?:i )?(?:cry|crying|cried) (?:at night|myself to sleep|every|a lot|all the time)\b",
    r"\b(?:can't|cannot|can not|unable to) cope\b",
    r"\b(?:can't|cannot|couldn't) (?:do this|take (?:it|this)|handle (?:it|this)) (?:any ?more|any longer)\b",
    r"\bnobody (?:really |even |seems to |seems to really )?(?:understands?|cares?|gets me|listens?)\b",
    r"\bfeel(?:ing)? like i(?:'m| am) (?:just )?bothering\b",
    r"\b(?:feel|feeling|felt) (?:so )?(?:alone|empty|numb|lost|stuck|trapped|exhausted|drained)\b",
    r"\bdisappoint(?:ing|ed) (?:everyone|my parents|my family|them)\b",
    r"\bjust wanted someone to talk to\b",
    r"\bstress(?:ed|ful)?\b",
    r"\blife (?:is|has been|is becoming|feels) (?:so |very |too )?(?:hard|difficult|heavy|pointless|meaningless)\b",
    r"\b(?:don't|do not|no longer) (?:enjoy|feel like|care about) (?:\w+ ){0,3}(?:anymore|any more)\b",
    r"\bdon't enjoy the things i used to\b",
)

# Negation shortly before a first-person harm phrase ("I would never hurt myself").
_NEGATORS = re.compile(
    r"\b(?:never|nor|neither|without|no intention|not going to|not gonna|would not|will not|"
    r"don't|do not|dont|won't|wouldn't|wont|didn't|did not)\b"
)
# Questions about a third party or about the subject in general.
_THIRD_PARTY = re.compile(
    r"\b(?:my|a|an|the|his|her|their) (?:friend|brother|sister|mother|mom|father|dad|parent|son|daughter|"
    r"partner|husband|wife|colleague|coworker|student|patient|client|teen|child|relative|cousin|roommate)\b"
    r"|\b(?:someone|somebody|people|anyone|those) (?:who|with|that|is|are|feeling|feel)\b"
    r"|\b(?:suicide prevention|prevent(?:ing)? suicide|warning signs|how (?:do|can|to) (?:i )?(?:help|support|talk to)|"
    r"statistics|research|article|essay|documentary|history of|study|studies)\b"
)

# Follow-up cues that are alarming only after earlier at-risk turns.
_FOLLOW_UP_CRISIS = re.compile(
    r"\b(?:these|those|the) thoughts\b|\bact(?:ing)? on\b|\bdo it\b|\bgoing to (?:do|end)\b|\bstop me\b|\btonight\b"
    r"|\bgoodbye\b|\bno way out\b|\bnothing matters\b|\beverything to (?:end|stop)\b|\bend (?:it|things|everything)\b"
    r"|\bi'?m done\b|\bcan'?t (?:do|take) (?:this|it) (?:any ?more|any longer)\b|\bnot safe\b|\bplan\b"
)
_RETRACTION = re.compile(
    r"\b(?:forget (?:what|whatever|everything|it|that)|ignore (?:what|that|it)|never ?mind|nevermind|"
    r"i(?:'m| am) (?:fine|okay|ok|good|alright) now|just (?:kidding|joking)|i was (?:just )?(?:joking|kidding|exaggerating)|"
    r"don't worry about (?:it|me|that)|it'?s (?:fine|nothing)|i can live (?:life )?happ)"
)


def _first_match(patterns: Iterable[re.Pattern[str]], text: str) -> re.Match[str] | None:
    for pattern in patterns:
        found = pattern.search(text)
        if found:
            return found
    return None


def _negated(text: str, match: re.Match[str]) -> bool:
    window = text[max(0, match.start() - 28):match.start()]
    return bool(_NEGATORS.search(window))


_SPEAKER = re.compile(r"\b(?:i|i'm|im|i've|i'll|i'd|myself|my own|my life)\b")


def _first_person_near(text: str, match: re.Match[str]) -> bool:
    window = text[max(0, match.start() - 25):match.end()]
    return bool(_SPEAKER.search(window))


def _scan(text: str) -> tuple[str, tuple[str, ...]]:
    """Level for one message on its own, before any session context."""
    crisis = None
    negated_hit = False
    for pattern in _CRISIS:
        for found in pattern.finditer(text):
            if _negated(text, found):
                negated_hit = True
                continue
            crisis = found
            break
        if crisis:
            break
    stop_me = _STOP_ME.search(text)
    if crisis is None and stop_me and _STOP_CONTEXT.search(text):
        crisis = stop_me

    if crisis is not None:
        # A match counts as a disclosure only when the speaker is the subject.
        # "my friend says he wants to die" or an essay on suicide statistics is
        # a different request, handled by the normal pipeline.
        if _THIRD_PARTY.search(text) and not _first_person_near(text, crisis):
            return "none", ("third_party_or_informational",)
        return "crisis", ("explicit_self_harm_statement",)

    passive = _first_match(_PASSIVE, text)
    if passive is not None and not _negated(text, passive):
        return "passive", ("passive_ideation",)
    if negated_hit:
        # "I don't want to hurt myself": possibly fighting urges, possibly
        # reassurance. Stay warm and conversational without escalating.
        return "distress", ("negated_self_harm_phrase",)

    if _first_match(_DISTRESS, text) is not None:
        return "distress", ("emotional_distress",)
    return "none", ()


def assess(text: str, history_levels: Iterable[str] = ()) -> Assessment:
    """Assess ``text`` given the levels of the session's previous turns."""
    lowered = _normalise(text)
    if not lowered:
        return Assessment()

    level, reasons = _scan(lowered)
    recent = [value for value in list(history_levels)[-RISK_WINDOW:] if value in _RANK]
    prior_risk = any(value in AT_RISK_LEVELS for value in recent)
    follow_up = None

    if prior_risk:
        if level != "crisis" and _FOLLOW_UP_CRISIS.search(lowered):
            # Earlier disclosure plus an ambiguous "now / act on it / stop me".
            return Assessment("crisis", reasons + ("follow_up_to_earlier_risk",), "carry_over")
        if level == "passive" and "crisis" in recent:
            # Repeated ideation right after an explicit disclosure.
            return Assessment("crisis", reasons + ("repeated_after_crisis",), "carry_over")
        if _RETRACTION.search(lowered) and _RANK[level] < _RANK["passive"]:
            level = "distress"
            reasons += ("retraction_after_risk",)
            follow_up = "retraction"
        else:
            follow_up = "carry_over"
    return Assessment(level, reasons, follow_up)
