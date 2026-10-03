import config as C
from src import plot_style as S


def test_every_family_has_a_fixed_colour_and_marker():
    families = {spec["family"] for spec in C.MODELS.values()} | {"combination"}
    assert families <= set(S.FAMILIES)
    assert len(set(S.PALETTE)) == len(S.FAMILIES)
    assert set(S.MARKER) == set(S.FAMILIES)


def test_palette_keeps_the_validated_order():
    # Order matters: it is what passed the colour-vision checks.
    assert S.PALETTE[:3] == ["#2a78d6", "#eb6834", "#1baf7a"]


def test_tier_and_rule_names_map_to_families():
    assert S.family("lgbm_oni_x2") == "boosting"
    assert S.family("sarimax_x1") == "statistical"
    assert S.family("fixed_share") == "combination"
    assert S.family("snaive_drift") == "simple"
