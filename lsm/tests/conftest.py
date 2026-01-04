# exploration/Transcript/lsm/tests/conftest.py
import io
import tempfile
import textwrap
import pytest

from lsm.core import load_liwc_dic, build_compiled_patterns

@pytest.fixture
def liwc_sample_path(tmp_path):
    """
    Minimal J-LIWC-like dictionary (categories: ppron=1, ipron=2).
    Uses % to separate category definitions and vocabulary. Includes wildcard (*).
    """
    dic_text = textwrap.dedent("""\
        %
        1    ppron
        2    ipron
        %
        私*    1
        自分    1
        何か    2
    """)
    p = tmp_path / "mini.dic"
    p.write_text(dic_text, encoding="utf-8")
    return str(p)

@pytest.fixture
def compiled_patterns(liwc_sample_path):
    cat_map, word_map = load_liwc_dic(liwc_sample_path)
    allowed = {"ppron", "ipron"}
    return build_compiled_patterns(word_map, cat_map, allowed)
