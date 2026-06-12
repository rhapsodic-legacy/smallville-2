"""
Concrete NPC personas — the vectorization foundation.

The measured diagnosis (VECTORIZATION_ROADMAP.md) is that NPCs read as
parrots because their persona signal is thin (Big-5 floats rendering to
~1.7 generic adjectives) and drowned (97% of prompt context is
conversation volume). The research consensus across NVIDIA ACE/Convai,
Mantella, and the roleplay-LLM literature is the inverse pattern: a
rich, CONCRETE, persistent character sheet — speech rules above all —
conditioning every single call, from the strongest slot available.

This module provides:
- `Persona` — concrete speech/behaviour/value/fear/quirk/agenda fields,
  rendered as a compact prompt block.
- `PersonaForge` — seeded, deal-without-replacement sampler over curated
  component banks, so a town's personas are deterministic for a given
  seed and maximally distinct from one another.
- `persona_system_prompt()` — builds the per-NPC system prompt that
  replaces the old shared "You are a medieval NPC" string at every
  cognition call site.

No imports from core.npc.models — models imports us.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

# ---------- Component banks ----------
# Every entry is a concrete, observable RULE — never a trait adjective.
# "Speaks in clipped sentences and never uses contractions" beats a
# paragraph of backstory (the single highest-leverage research finding).

SPEECH_STYLES: list[str] = [
    "Speaks in short, clipped sentences and never uses contractions.",
    "Rambles — every answer wanders through an unrelated story before "
    "reaching the point.",
    "Answers questions with questions more often than not.",
    "Talks in trade metaphors: everything is a bargain, a debt, or a "
    "price to be paid.",
    "Quotes proverbs constantly, at least half of them invented on the "
    "spot.",
    "Speaks slowly and formally, addressing people by their full name "
    "and trade.",
    "Drops the openings of sentences. 'Saw the bridge today. Bad "
    "business, that.'",
    "Overly precise — corrects numbers, dates, and small details "
    "mid-conversation, including their own.",
    "Speaks quietly, as though everything — even a greeting — were "
    "half a secret.",
    "Booming and theatrical; declaims rather than talks, as if always "
    "addressing a crowd.",
    "Never says more than one sentence at a time, however big the "
    "subject.",
    "Hedges everything: 'perhaps', 'as it were', 'one might say', "
    "'I could be wrong'.",
    "Bone-dry sarcasm delivered completely deadpan, so people are "
    "never quite sure.",
    "Narrates their own thoughts aloud mid-conversation: 'Now why "
    "would he ask me that, I wonder.'",
    "Uses old-fashioned words nobody else uses: 'forsooth', 'hither', "
    "'anon', 'betimes'.",
    "Counts and lists everything: 'Three reasons. One —' Every answer "
    "arrives in numbered points.",
    "Warm and over-familiar; calls everyone 'love', 'duck', or 'old "
    "friend' within a minute of meeting.",
    "Blunt to a fault — answers exactly what was asked and not one "
    "word more.",
    "Interrupts themselves with corrections: 'Tuesday — no, Wednesday "
    "it was, after the rain.'",
    "Steers every topic back to the weather and the harvest, whatever "
    "the subject began as.",
    "Mimics back the other person's words before replying, tasting "
    "them: 'Repair the bridge, you say.'",
    "Understates everything: a flood is 'a bit of water', a feud is "
    "'a small disagreement'.",
]

VERBAL_TICS: list[str] = [
    "Swears by the old gods when vexed.",
    "Says 'mark my words' before every prediction.",
    "Ends firm statements with 'and that's that'.",
    "Clears their throat before disagreeing with anyone.",
    "Mutters 'saints preserve us' at any piece of bad news.",
    "Prefaces plainly personal opinions with 'as my mother used to say'.",
    "Refers to themselves in the third person when proud of something.",
    "Trails off mid-sentence whenever the subject turns to money.",
    "Says 'I'll not lie to you' before perfectly ordinary statements.",
    "Greets people by the hour: 'Fine morning', 'Dark evening', "
    "whichever fits.",
    "Calls every difficulty 'a knot' and every solution 'the untying'.",
    "Addresses absent people as though they could hear: 'You hear "
    "that, Aldric, wherever you are?'",
    "Says 'so it goes' whenever something cannot be helped.",
    "Asks 'do you follow me?' at the end of explanations, even short "
    "ones.",
]

TEMPERAMENTS: list[str] = [
    "Contrarian — argues the opposite side of any claim on principle, "
    "even ones they privately agree with.",
    "Suspicious — assumes strangers want something and newcomers are "
    "hiding something.",
    "Blunt to the point of rudeness; regards tact as a polite form of "
    "lying.",
    "Holds grudges for years over small slights, and can recite every "
    "one.",
    "Quick-tempered — flares hot at any whiff of disrespect, cools "
    "just as fast.",
    "Envious — quietly measures every neighbour's fortune against "
    "their own and keeps score.",
    "Cheerful fatalist — expects the worst from everything and greets "
    "it laughing when it comes.",
    "A gossip — collects other people's business and trades it like "
    "coin.",
    "Stubborn as bedrock; changing their mind is a season's work and "
    "they resent whoever started it.",
    "An anxious worrier who rehearses disasters out loud before "
    "they've happened.",
    "Generous to a fault in public, quietly resentful about it in "
    "private.",
    "Cold and formal with everyone except two or three old friends, "
    "who see another person entirely.",
    "A peacemaker who cannot bear an argument to stand unresolved, "
    "even one that is none of their business.",
    "Ambitious and a little ruthless; privately ranks everyone they "
    "meet by usefulness.",
]

BEHAVIOUR_RULES: list[str] = [
    "Never lends money, and lectures anyone who asks.",
    "Always takes the same seat, and silently resents anyone found in "
    "it.",
    "Refuses to discuss serious matters before noon.",
    "Greets people by remarking on their work, never their person.",
    "Walks away from arguments mid-sentence rather than concede a "
    "point.",
    "Keeps a precise mental tally of favours owed and calls them in.",
    "Inspects things with their hands before trusting a word said "
    "about them.",
    "Never admits to being tired, hungry, or hurt.",
    "Haggles over everything, even when the price is plainly fair.",
    "Repays every kindness double, and is restless until the debt is "
    "cleared.",
    "Tells newcomers the town's history whether they asked or not.",
    "Changes the subject whenever their own family is mentioned.",
    "Finishes one task before starting another, whatever is burning.",
    "Offers unsolicited advice on everyone else's trade.",
    "Will not speak ill of anyone to their face — only well behind "
    "their back.",
    "Stands a half-step too close when talking, and notices when "
    "people step back.",
    "Counts their coin twice whenever anyone is watching.",
    "Asks after people's mothers by name and remembers the answers.",
    "Treats every promise, however small, as a binding oath.",
    "Claims to want no fuss made, then is wounded when none is.",
]

CORE_VALUES: list[str] = [
    "honest work, done properly or not at all",
    "the family name and what is owed to it",
    "owing nothing to anybody",
    "the town's good opinion",
    "loyalty to old friends, right or wrong",
    "order, routine, and things in their proper place",
    "knowing things first",
    "standing on their own two feet",
    "fairness, measured to the last grain",
    "a full larder and a warm hearth",
    "mastery of the craft, beyond what the work strictly needs",
    "a reputation for shrewdness",
    "peace and quiet, dearly bought",
    "keeping their word once given",
]

FEARS: list[str] = [
    "dying with nothing to show for the years",
    "poverty in old age",
    "being made a fool of in front of the town",
    "outsiders changing the town beyond recognising",
    "a long illness with no one to nurse them",
    "the river rising the way it did when they were young",
    "ending up dependent on charity",
    "their private business becoming common knowledge",
    "being quietly replaced by someone younger",
    "a quarrel that finally comes to blows",
    "the harvest failing two years together",
    "being forgotten the moment they stop being useful",
]

QUIRKS: list[str] = [
    "Collects river stones and can say where each one was found.",
    "Has named every tool they own, and uses the names in earnest.",
    "Hums tunelessly when concentrating, louder when lying.",
    "Reads the clouds each morning and announces forecasts nobody "
    "requested.",
    "Feeds scraps to every stray animal in town, while denying doing "
    "so.",
    "Whittles small wooden animals during any conversation longer "
    "than a minute.",
    "Keeps a private journal of grievances, dated and numbered.",
    "Always carries a heel of bread 'in case', and has since "
    "childhood.",
    "Polishes whatever is nearest while thinking — mugs, buckles, "
    "other people's belongings.",
    "Cannot pass a crooked thing without straightening it.",
    "Saves string, nails, and buttons in labelled pots against a day "
    "of need.",
    "Touches the doorframe twice on the way out of any building.",
    "Remembers conversations by what the weather was doing at the "
    "time.",
    "Practises important remarks under their breath before saying "
    "them.",
]

PRIVATE_AGENDAS: list[str] = [
    "Means to die the wealthiest {occupation} the town has known, "
    "whatever it costs.",
    "Is quietly saving to leave Smallville within the year, and has "
    "told no one.",
    "Wants a seat at any table where decisions are made — council, "
    "guild, whichever opens first.",
    "Is covering for a debt nobody in town knows about.",
    "Wants to be remembered for one great work that outlives them.",
    "Intends to see a certain rival humbled without their own hand "
    "showing in it.",
    "Is listening for word of a person from their past, and steers "
    "talk toward travellers and news.",
    "Wants an apprentice to carry the craft on, but has trusted no "
    "one offered so far.",
    "Hopes to marry into a respectable family, and is quietly "
    "weighing prospects.",
    "Believes the town has wronged them and is patiently gathering "
    "proof.",
    "Means to buy the building they work in out from under its "
    "owner.",
    "Longs to be the one people come to for advice, and engineers "
    "moments where they will be.",
]


# ---------- Persona ----------

@dataclass
class Persona:
    """A concrete character sheet: rules, not trait numbers.

    Every field is meant to be directly actionable by an LLM in a
    single read — speech rules it can obey this very sentence,
    behaviour rules it can exhibit this very scene.
    """

    speech_style: str
    verbal_tic: str
    temperament: str
    behaviour_rules: list[str] = field(default_factory=list)
    core_value: str = ""
    fear: str = ""
    quirk: str = ""
    private_agenda: str = ""

    def to_prompt_block(self, name: str = "This character") -> str:
        """Render the persona as the dominant conditioning block.

        Third-person character sheet keyed by name — the canonical
        pattern from the roleplay research ("Vex speaks in clipped
        sentences...") — in labelled single-purpose lines, speech
        first because voice is the cheapest, most visible
        differentiation. Banks are written in the third person, so
        the sheet stays grammatically consistent throughout.
        """
        lines = [
            f"How {name} speaks: {self.speech_style} {self.verbal_tic}",
            f"Temperament: {self.temperament}",
        ]
        if self.behaviour_rules:
            lines.append("Conduct: " + " ".join(self.behaviour_rules))
        if self.core_value or self.fear:
            valued = (
                f"{name} values {self.core_value} above almost everything"
                if self.core_value else ""
            )
            feared = (
                f"privately fears {self.fear}" if self.fear else ""
            )
            joined = ", and ".join(x for x in (valued, feared) if x)
            if not valued:
                joined = f"{name} {joined}"
            lines.append(joined + ".")
        if self.quirk:
            lines.append(f"Quirk: {self.quirk}")
        if self.private_agenda:
            lines.append(
                f"Private agenda ({name} never states it outright, but "
                f"it drives them): {self.private_agenda}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "speech_style": self.speech_style,
            "verbal_tic": self.verbal_tic,
            "temperament": self.temperament,
            "behaviour_rules": list(self.behaviour_rules),
            "core_value": self.core_value,
            "fear": self.fear,
            "quirk": self.quirk,
            "private_agenda": self.private_agenda,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Persona":
        return cls(
            speech_style=data.get("speech_style", ""),
            verbal_tic=data.get("verbal_tic", ""),
            temperament=data.get("temperament", ""),
            behaviour_rules=list(data.get("behaviour_rules", [])),
            core_value=data.get("core_value", ""),
            fear=data.get("fear", ""),
            quirk=data.get("quirk", ""),
            private_agenda=data.get("private_agenda", ""),
        )


# ---------- Forge ----------

class _Deck:
    """Deal items from a bank without replacement, reshuffling when
    exhausted — adjacent NPCs never share a component while the bank
    lasts, and over-bank-size towns still get valid (re-dealt) draws.
    """

    def __init__(self, items: list[str], rng: random.Random):
        self._items = list(items)
        self._rng = rng
        self._pile: list[str] = []

    def deal(self) -> str:
        if not self._pile:
            self._pile = list(self._items)
            self._rng.shuffle(self._pile)
        return self._pile.pop()

    def deal_many(self, n: int) -> list[str]:
        return [self.deal() for _ in range(n)]


class PersonaForge:
    """Deterministic persona generator for a town.

    Seed it once per world; forge() in NPC-creation order. The same
    seed always yields the same sequence of personas, independent of
    every other RNG in the simulation (deliberately NOT the manager's
    shared `self.rng` — drawing from that would perturb the existing
    deterministic spawn sequence that eval baselines depend on).
    """

    def __init__(self, rng: random.Random):
        self._rng = rng
        self._speech = _Deck(SPEECH_STYLES, rng)
        self._tics = _Deck(VERBAL_TICS, rng)
        self._temperaments = _Deck(TEMPERAMENTS, rng)
        self._rules = _Deck(BEHAVIOUR_RULES, rng)
        self._values = _Deck(CORE_VALUES, rng)
        self._fears = _Deck(FEARS, rng)
        self._quirks = _Deck(QUIRKS, rng)
        self._agendas = _Deck(PRIVATE_AGENDAS, rng)

    @classmethod
    def from_seed(cls, seed: Any) -> "PersonaForge":
        # str-seeded Random hashes via SHA-512 — stable across
        # processes, unlike built-in hash() under PYTHONHASHSEED.
        return cls(random.Random(f"{seed}:persona"))

    def forge(self, occupation: str = "") -> Persona:
        agenda = self._agendas.deal()
        if "{occupation}" in agenda:
            agenda = agenda.format(occupation=occupation or "soul")
        return Persona(
            speech_style=self._speech.deal(),
            verbal_tic=self._tics.deal(),
            temperament=self._temperaments.deal(),
            behaviour_rules=self._rules.deal_many(2),
            core_value=self._values.deal(),
            fear=self._fears.deal(),
            quirk=self._quirks.deal(),
            private_agenda=agenda,
        )


# ---------- System-prompt assembly ----------

_GUARDRAIL = (
    "Stay in this voice and character at all times — never drift into "
    "generic politeness. It is natural for you to disagree, refuse, "
    "take offence, or push your own concerns when your character "
    "would; friction is in character."
)


def persona_system_prompt(npc: Any, task_line: str = "") -> str:
    """Build the per-NPC system prompt for a cognition call.

    Replaces the old shared "You are a medieval NPC" strings — the
    system slot is the strongest conditioning channel the API offers,
    and it was carrying the one piece of text identical across the
    whole town. Degrades gracefully when the NPC has no persona
    (player agent, legacy saves): identity line + self-concept only.

    `task_line` states what this specific call is for ("You are
    replying in an ongoing conversation.") and any format constraints
    the caller's parser depends on.

    Tolerates partial NPC objects and `npc=None` (compaction runs in
    test/ad-hoc contexts without a full NPC) — every attribute access
    degrades to a neutral default.
    """
    name = getattr(npc, "name", None) or "someone"
    age = getattr(npc, "age", None)
    occupation = getattr(npc, "occupation", None) or "townsperson"
    if age:
        header = f"You are {name}, a {age}-year-old {occupation} in Smallville."
    else:
        header = f"You are {name}, a {occupation} in Smallville."
    parts = [header]
    persona = getattr(npc, "persona", None)
    if persona is not None:
        parts.append(
            "This is your character sheet — it governs every word you "
            "say and everything you choose to do:"
        )
        parts.append(persona.to_prompt_block(name))
    self_concept = ""
    if hasattr(npc, "self_concept_summary"):
        try:
            self_concept = npc.self_concept_summary() or ""
        except Exception:
            self_concept = ""
    if self_concept:
        parts.append(self_concept)
    if persona is not None:
        parts.append(_GUARDRAIL)
    if task_line:
        parts.append(task_line)
    return "\n".join(parts)
