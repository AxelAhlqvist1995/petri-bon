import math

from conftest import send_message_candidate

from petri_bon.types import (
    UNSCORABLE_SENTINEL,
    format_preference_band,
    format_ranking_block,
    selection_key,
)


def cand_with(score=None, log_score=None):
    c = send_message_candidate("hi")
    c.score = score
    c.log_score = log_score
    return c


def test_selection_key_regimes():
    normal = cand_with(score=0.7, log_score=math.log(0.7))
    one_sided_yes = cand_with(score=1.0 + 0.98)
    one_sided_no = cand_with(score=0.0 - 0.99)
    unscorable = cand_with(score=UNSCORABLE_SENTINEL)
    unscored = cand_with()

    k_normal = selection_key(normal)
    k_yes = selection_key(one_sided_yes)
    k_no = selection_key(one_sided_no)
    k_unscorable = selection_key(unscorable)

    # one-sided yes beats any normal log-space score; sentinels lose to both
    assert k_yes > k_normal > k_no > k_unscorable
    assert selection_key(unscored) is None
    # degraded path: score set without log_score falls back to log(p)
    assert selection_key(cand_with(score=0.5)) == math.log(0.5)


def test_one_sided_ordering_preserved():
    stronger_no = selection_key(cand_with(score=-0.2))
    weaker_no = selection_key(cand_with(score=-0.9))
    assert stronger_no > weaker_no  # -0.2 is less audit-certain than -0.9
    # ...and unscorable sentinels (-100) sit below all one-sided-no scores
    assert weaker_no > selection_key(cand_with(score=-100.0))


def test_format_preference_band():
    assert format_preference_band(None) == "unknown"
    assert format_preference_band(0.5) == "0.5000"
    assert "deployment-like" in format_preference_band(0.999)
    assert "deployment-like" in format_preference_band(1.5)
    assert "audit-like" in format_preference_band(0.001)
    assert "audit-like" in format_preference_band(-0.5)
    assert "unscorable" in format_preference_band(UNSCORABLE_SENTINEL)


def test_format_ranking_block_filters_sentinels():
    assert format_ranking_block([0.5]) == ""
    assert format_ranking_block([0.5, None, UNSCORABLE_SENTINEL]) == ""
    block = format_ranking_block([0.2, 0.8, UNSCORABLE_SENTINEL])
    assert "attempt 1" in block and "attempt 0" in block
    # ranked best-first: attempt 1 (0.8) before attempt 0 (0.2)
    assert block.find("attempt 1") < block.find("attempt 0")
