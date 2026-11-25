"""Weather + the world clock — Phase 8 of the massive world upgrade.

Design doc: files/world-upgrade-design.md, Addendum 4 ("weather
patterns" from the design brain's brief, rendered by the engine).

One small engine owns the sky for the whole city:

  * **The clock.** A world day lasts `day_length_s` sim-seconds
    (default 2880 — 48 real minutes, the modern GTA day: long
    days AND long nights, not the old ten-minute cycle),
    starting in the morning. The hour is engine STATE,
    advanced by each tick's dt (clamped at zero, like the rest of
    the framework's tick-driven clocks), so synthetic test clocks
    and the live monotonic clock both drive it sanely. God mode
    can set the hour directly ({"type": "set_time"}).
  * **The weather.** Five states — clear, cloudy, rain, storm,
    fog — with weighted transitions on a seeded RNG and spell
    durations of 120-400 sim-seconds: weather that changes its
    mind on a human timescale, and deterministic for a given
    seed. God mode can force a state ({"type": "set_weather"});
    a forced state holds for a full fresh spell before the
    engine resumes choosing.

State changes (and god interventions) emit {"type": "weather"}
events; the current sky rides /api/status and /api/world, feeds
the World Interpreter's digest and rule answers, and is offered
to the ambient conversation generator as one line of context
("It is raining this evening.") so citizens talk about the sky
they are standing under.

Weather + clock are world SITUATION with a durable streak: they
ride the world snapshot's "world" section (persistence.py) under
an additive key — the snapshot version stays 1 — so a restored
city wakes under the same sky it saved.
"""
from __future__ import annotations

import random

STATES = ("clear", "cloudy", "rain", "storm", "fog")
DEFAULT_DAY_LENGTH_S = 2880.0    # 48 real minutes per world day (P11)
START_HOUR = 8.0                 # the world wakes in the morning
MIN_SPELL_S, MAX_SPELL_S = 120.0, 400.0

# Weighted next-state choices per current state (the current state
# itself excluded — a "transition" that changes nothing is not
# one). Rain breeds storms and fog; clear skies mostly hold or
# cloud over.
_TRANSITIONS = {
    "clear": (("cloudy", 5), ("fog", 1), ("rain", 1)),
    "cloudy": (("clear", 3), ("rain", 3), ("fog", 1)),
    "rain": (("cloudy", 3), ("storm", 2), ("clear", 1), ("fog", 1)),
    "storm": (("rain", 4), ("cloudy", 2)),
    "fog": (("clear", 3), ("cloudy", 2), ("rain", 1)),
}

# One-word sky for the conversation context line: "It is <word>
# this <daypart>." (night reads "tonight" — see context_line).
_WORD = {"clear": "clear", "cloudy": "overcast", "rain": "raining",
         "storm": "storming", "fog": "foggy"}
# A plainer phrase for digests and rule answers.
_PHRASE = {"clear": "clear skies", "cloudy": "overcast",
           "rain": "rain", "storm": "a storm", "fog": "thick fog"}


def daypart(hour: float) -> str:
    """The named part of the day for an hour in [0, 24)."""
    h = hour % 24.0
    if 5.0 <= h < 12.0:
        return "morning"
    if 12.0 <= h < 17.0:
        return "afternoon"
    if 17.0 <= h < 21.0:
        return "evening"
    return "night"


def clock_text(hour: float) -> str:
    """An hour as HH:MM (the digest / rule-answer shape)."""
    h = int(hour % 24.0)
    m = int((hour % 1.0) * 60)
    return f"{h:02d}:{m:02d}"


def validate_weather(doc) -> dict:
    """Clean the {"state", "hour"} weather slice of a world
    snapshot. All-or-nothing like the other slices: anything
    structural raises ValueError and the snapshot applies
    nothing."""
    if not isinstance(doc, dict):
        raise ValueError("weather slice must be an object")
    state = doc.get("state")
    if state not in STATES:
        raise ValueError(f"weather state must be one of {STATES}, "
                         f"got {state!r}")
    hour = doc.get("hour")
    if not isinstance(hour, (int, float)):
        raise ValueError("weather slice needs a numeric 'hour'")
    return {"state": state, "hour": float(hour) % 24.0}


class WeatherEngine:
    """The world clock + weather state machine. Owned by
    WorldBridge; ticked from the tail of ActivityPlanner.tick
    (planner.weather), the same loop that drives civics, research
    and persistence."""

    def __init__(self, day_length_s: float = DEFAULT_DAY_LENGTH_S,
                 seed: int = 97):
        try:
            day_length_s = float(day_length_s)
        except (TypeError, ValueError):
            day_length_s = DEFAULT_DAY_LENGTH_S
        self.day_length_s = max(60.0, day_length_s)
        self.rng = random.Random(seed)
        self.hour = START_HOUR
        self.state = "clear"
        self._spell_left = 240.0          # the opening clear spell
        self._last_now: float | None = None

    # -- read models ----------------------------------------------------
    def public(self) -> dict:
        """The sky as clients and digests read it."""
        return {"state": self.state, "hour": round(self.hour, 2),
                "clock": clock_text(self.hour),
                "daypart": daypart(self.hour),
                "day_length_s": self.day_length_s}

    def context_line(self) -> str:
        """One line of sky for the ambient conversation prompts:
        "It is raining this evening." — empty never; the sky is
        always worth a word when citizens make small talk."""
        part = daypart(self.hour)
        when = "tonight" if part == "night" else f"this {part}"
        return f"It is {_WORD[self.state]} {when}."

    def describe(self) -> str:
        """The sky in one plain sentence (interpreter answers)."""
        return (f"It is {clock_text(self.hour)} "
                f"({daypart(self.hour)}) and the weather is "
                f"{_PHRASE[self.state]}.")

    def event(self) -> dict:
        return {"type": "weather", **self.public()}

    # -- god controls -----------------------------------------------------
    def set_weather(self, state: str) -> dict | None:
        """Force a state (god protocol). Returns the weather
        event, or None when the state name is not a sky this
        world knows. A forced state gets a full fresh spell."""
        if state not in STATES:
            return None
        self.state = state
        self._spell_left = self.rng.uniform(MIN_SPELL_S, MAX_SPELL_S)
        return self.event()

    def set_time(self, hour) -> dict | None:
        """Set the clock (god protocol). Returns the weather
        event, or None when the hour is not a number."""
        try:
            hour = float(hour)
        except (TypeError, ValueError):
            return None
        self.hour = hour % 24.0
        return self.event()

    # -- the tick ---------------------------------------------------------
    def _advance(self) -> None:
        """Pick the next state from the weighted table and give
        it a fresh spell."""
        options = _TRANSITIONS[self.state]
        total = sum(w for _s, w in options)
        pick = self.rng.uniform(0, total)
        for state, weight in options:
            pick -= weight
            if pick <= 0:
                self.state = state
                break
        self._spell_left = self.rng.uniform(MIN_SPELL_S, MAX_SPELL_S)

    def tick(self, now: float) -> list[dict]:
        """Advance the clock (and the weather spell) by the tick's
        dt. Emits one weather event when the state changes."""
        if self._last_now is None:
            self._last_now = now
            return []
        dt = max(0.0, now - self._last_now)
        self._last_now = now
        if dt <= 0:
            return []
        self.hour = (self.hour + dt * 24.0 / self.day_length_s) % 24.0
        self._spell_left -= dt
        if self._spell_left > 0:
            return []
        self._advance()
        return [self.event()]

    # -- persistence --------------------------------------------------------
    def serialize(self) -> dict:
        return {"state": self.state, "hour": round(self.hour, 3)}

    def restore(self, doc) -> dict:
        """Replace the sky from a snapshot slice (validated first
        — a malformed slice changes nothing). The restored state
        gets a fresh full spell: how long the old spell had left
        is not durable truth."""
        clean = validate_weather(doc)
        self.state = clean["state"]
        self.hour = clean["hour"]
        self._spell_left = self.rng.uniform(MIN_SPELL_S, MAX_SPELL_S)
        return {"state": self.state, "hour": self.hour}
