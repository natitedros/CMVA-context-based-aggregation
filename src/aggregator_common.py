"""Shared pieces for the contextual aggregator: the moral profile and the prompt format."""
import pandas as pd

from config import VALUES
from prepare_data import has_value

AGG_DIR_NAME = "aggregator"


def profile_from_moral(moral_field):
    """MIC's foundation tags on a Rule of Thumb -> a 0/1 vector over our five values."""
    return [int(has_value(moral_field, v)) for v in VALUES]


def profile_text(profile):
    active = [v for v, p in zip(VALUES, profile) if p]
    return ", ".join(active) if active else "none"


def build_prompt(question, profile, answers):
    """The aggregator's input: question, moral profile, and one answer from each value's agent."""
    lines = [f"Question: {question}",
             f"Values that matter for this user: {profile_text(profile)}",
             "Answers from five moral agents:"]
    lines += [f"[{v}] {answers[v]}" for v in VALUES]
    lines.append("Write one answer to the question that best reflects the values that matter for this user.")
    return "\n".join(lines)


def load_test_agent_answers(results_dir, arm):
    """Agent answers to the 500 held-out questions, produced earlier by generate_and_score.py."""
    out = None
    for v in VALUES:
        g = pd.read_csv(f"{results_dir}/generations_{arm}_{v}_sd.csv")[["Q", "reply"]]
        g = g.drop_duplicates("Q").rename(columns={"reply": f"ans_{v}"})
        out = g if out is None else out.merge(g, on="Q")
    return out
