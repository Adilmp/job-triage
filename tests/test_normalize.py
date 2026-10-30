from triage.normalize import content_hash, fingerprint, strip_html


def test_greenhouse_escaped_html_becomes_clean_text():
    # Greenhouse content=true returns the HTML itself HTML-escaped (P1), with a nested &amp;nbsp;.
    raw = ("&lt;div class=&quot;content-intro&quot;&gt;&lt;h2&gt;About us&lt;/h2&gt;"
           "&lt;p&gt;We build R&amp;amp;D tools.&amp;nbsp;Join&amp;nbsp;us.&lt;/p&gt;"
           "&lt;ul&gt;&lt;li&gt;Python&lt;/li&gt;&lt;li&gt;SQL&lt;/li&gt;&lt;/ul&gt;&lt;/div&gt;")
    out = strip_html(raw)
    assert "<" not in out and "&lt;" not in out and "&nbsp;" not in out
    assert "About us" in out and "R&D tools." in out and "Join us." in out
    assert "- Python" in out and "- SQL" in out


def test_script_and_style_content_is_dropped():
    html = "<style>.a{color:red}</style><p>Hello</p><script>alert('x')</script><p>World</p>"
    assert strip_html(html) == "Hello\n\nWorld"


def test_plain_html_and_whitespace_collapse():
    assert strip_html("<p>a   b\t\tc</p>\n\n\n\n<p>d</p>") == "a b c\n\nd"
    assert strip_html("line1<br/>line2") == "line1\nline2"
    assert strip_html("") == ""
    assert strip_html(None) == ""
    assert strip_html("   \n ") == ""


def test_content_hash_ignores_case_whitespace_and_unicode_form():
    a = content_hash("ML Engineer", "Sydney, AU", "Build  things.\nWith Python.")
    assert a == content_hash("ml engineer ", "sydney, au", "Build things. With python.")
    assert a == content_hash("ML Engineer", "Sydney, AU", "Build things. With Python.")
    assert len(a) == 64


def test_content_hash_changes_when_post_changes():
    base = content_hash("ML Engineer", "Sydney", "desc")
    assert base != content_hash("ML Engineer", "Sydney", "desc v2")
    assert base != content_hash("ML Engineer", "Remote", "desc")
    assert base != content_hash("Senior ML Engineer", "Sydney", "desc")
    assert content_hash("a", None, "d") == content_hash("a", "", "d")


def test_fingerprint_matches_same_role_across_boards():
    a = fingerprint("Marloo Pty Ltd", "Sr. ML Eng")
    assert a == fingerprint("marloo", "Senior ML Engineer")
    assert a == fingerprint("MARLOO PTY. LTD.", "Senior  ML  Engineer ")
    assert len(a) == 64


def test_fingerprint_normalization_rules():
    assert fingerprint("Acme Inc.", "Jr. Mgr") == fingerprint("Acme", "Junior Manager")
    assert fingerprint("R&D Labs LLC", "Engineer") == fingerprint("R and D Labs", "Engineer")
    assert fingerprint("Acme", "Data Engineer (m/f/d)") == fingerprint("Acme", "Data Engineer")
    assert fingerprint("ＡＣＭＥ", "Engineer") == fingerprint("acme", "engineer")   # NFKC full-width


def test_fingerprint_separates_different_roles_and_companies():
    assert fingerprint("Acme", "Data Engineer") != fingerprint("Acme", "Senior Data Engineer")
    assert fingerprint("Acme", "Data Engineer") != fingerprint("Acmeco", "Data Engineer")
    # the company/title boundary is part of the hash, so these differ
    assert fingerprint("ab", "c d") != fingerprint("a", "b c d")
