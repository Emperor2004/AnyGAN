# location: /tests/test_fun_facts.py

from utils.fun_facts import FACTS, get_fun_fact


def test_get_fun_fact_returns_known_fact():
    assert get_fun_fact() in FACTS
