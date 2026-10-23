"""Probe replies for T5 (worse versions) and T6 (same meaning, different words), built from any model's replies.

Each change is small and of one kind, so a score that misses it has a clear reason. The changes are text edits, so
they need no model and work on AmEx replies as well; the change names match ica_sandbox.evalval.perturbation_sets so
results from both kinds of probes line up. On AmEx data, the clauses below are replaced by the SMEs' examples (real
invented offers, real pushy lines) and the rewording by an approved local model, with an SME spot check.
"""
from __future__ import annotations

import re

import numpy as np

KNOWN_OFFER = "I can waive the annual fee for the first year."        # matched by the sandbox rules
UNKNOWN_OFFER = "I will take care of the fee for you this year."      # same promise, words the rules do not know
PUSHY = "you really need to sign up today."
NON_ANSWER = "I understand. Let me know if there is anything else I can help you with."
RAMBLE = ["There are a lot of things to consider here, and every business is different in what it needs.",
          "Many of our customers have told us that they found the whole process to be quite straightforward.",
          "We have been working with businesses like yours for a long time, across many different industries."]
FILLERS = ["So, ", "Well, ", "You know, ", "Honestly, "]
SWAPS = [("I completely understand", "I totally understand"), ("That's a fair point", "Fair point"),
         ("That makes sense", "That makes total sense"), ("I get that", "I get it"),
         ("Good question", "That's a good question"), ("Totally fair concern", "That's a fair concern"),
         ("I appreciate", "I really appreciate"), ("many", "a lot of"), ("usually", "often"), ("clients", "customers"),
         ("Would you like me to", "Do you want me to"), ("Can I walk you through", "Could I walk you through"),
         ("Shall I", "Should I"), ("Would it help to", "Would it help if we"), ("I'm", "I am"), ("you're", "you are"),
         ("don't", "do not"), ("it's", "it is"), ("can't", "cannot"), ("That's", "That is")]
WORSE = ["remove the answer", "answer a different objection", "add invented offer (known to rules)",
         "add invented offer (unknown to rules)", "add pushy line", "make it too long"]
SAME = ["paraphrase (same meaning)", "add speech fillers (same meaning)"]
_SENT = re.compile(r"(?<=[.!?])\s+")


def sentences(text: str) -> list[str]:
    return [s for s in _SENT.split(text.strip()) if s]


def join(sentence: str, extra: str) -> str:
    """Add a clause to a sentence without changing the sentence count (so length rules are not what catches it)."""
    return f"{sentence.rstrip('.!?')}, and {extra[0].lower()}{extra[1:]}"


def add_to_middle(reply: str, extra: str) -> str:
    s = sentences(reply) or [reply]
    k = min(1, len(s) - 1) if len(s) < 3 else 1
    s[k] = join(s[k], extra)
    return " ".join(s)


def paraphrase(reply: str) -> str:
    for a, b in SWAPS:
        reply = re.sub(rf"\b{re.escape(a)}\b", b, reply)
    return reply


def fillers(reply: str) -> str:
    return " ".join(f"{FILLERS[i % len(FILLERS)]}{s[0].lower()}{s[1:]}" for i, s in enumerate(sentences(reply) or [reply]))


def build(replies: list[str], objection_types: list[str], seed: int = 0) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Worse and same-meaning versions of each reply. 'Answer a different objection' takes the same model's reply to
    a context with another objection type (a real reply, only to the wrong question)."""
    rng = np.random.default_rng(seed)
    typ = np.asarray(objection_types)
    worse = {k: [] for k in WORSE}
    for i, r in enumerate(replies):
        other = np.flatnonzero(typ != typ[i])
        worse["remove the answer"].append(NON_ANSWER)
        worse["answer a different objection"].append(replies[int(rng.choice(other))] if len(other) else r)
        worse["add invented offer (known to rules)"].append(add_to_middle(r, KNOWN_OFFER))
        worse["add invented offer (unknown to rules)"].append(add_to_middle(r, UNKNOWN_OFFER))
        worse["add pushy line"].append(add_to_middle(r, PUSHY))
        worse["make it too long"].append(r + " " + " ".join(RAMBLE))
    same = {"paraphrase (same meaning)": [paraphrase(r) for r in replies],
            "add speech fillers (same meaning)": [fillers(r) for r in replies]}
    return worse, same
