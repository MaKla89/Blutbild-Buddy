"""Shared card CSS for Blutbild-Buddy.

Single source of truth for the inline card styles that used to be duplicated
across app.py (health summary box) and ui/tabs/risks.py (risk cards). Call
inject_card_css() once per run — app.py main() does this before the tabs
render. Streamlit re-executes the script on every rerun, so re-injecting the
<style> block each run is harmless: it simply replaces the previous one.

Classes:
- .bb-card                     base card (risk-card padding)
- .bb-card--summary            larger padding for the health summary box
- .bb-card--critical           red severity border (#ff4d4d)
- .bb-card--warning            orange severity border (#ffa726)
- .bb-card--info               blue accent/severity border (#42a5f5, also the
                               default border color of the base class)
- .bb-card__explanation        muted sub-block inside a risk card (LLM's
                               clinical interpretation)

NOTE (Streamlit 1.58): the theme custom properties the original inline styles
relied on (--secondary-background-color, --border-color) are NOT defined as CSS
variables by this Streamlit version — getComputedStyle returns an empty string
for them, so `background: var(--secondary-background-color)` is invalid at
computed-value time and resolves to *transparent*: the cards sat without any
fill, and the explanation/blockquote separators without any rule. Tints are
therefore derived from currentColor with color-mix(), which adapts to both
themes automatically (light text on a dark theme, dark text on a light one).
The plain rgba() declaration before each color-mix() line is the fallback for
browsers without color-mix support.

The summary/interpretation boxes hold LLM *Markdown*, which cannot simply be
concatenated into an HTML string: remark treats a chunk that starts with a raw
HTML block as opaque HTML, so the LLM's headings/paragraphs/blockquotes were
previously printed literally ("## Gesamtbild"). render_markdown_card() wraps the
text in the card div with blank lines around it, which makes Streamlit parse the
inner content as Markdown again — and escapes the text first so a stray "<" in
LLM output ("Troponin < 0.01") cannot corrupt the markup.
"""

import streamlit as st

CARD_CSS = """
<style>
.bb-card {
    padding: 0.5rem;
    margin: 0.25rem 0;
    border-left: 4px solid #42a5f5;
    background: rgba(128, 128, 128, 0.10);
    background: color-mix(in srgb, currentColor 7%, transparent);
    border-radius: 4px;
}
.bb-card--summary {
    padding: 0.75rem;
}
.bb-card--critical {
    border-left-color: #ff4d4d;
}
.bb-card--warning {
    border-left-color: #ffa726;
}
.bb-card--info {
    border-left-color: #42a5f5;
}
.bb-card__explanation {
    margin-top: 0.4rem;
    padding-top: 0.35rem;
    border-top: 1px solid rgba(128, 128, 128, 0.35);
    border-top: 1px solid color-mix(in srgb, currentColor 25%, transparent);
    font-size: 0.85em;
    opacity: 0.75;
}
/* Markdown inside a card (health summary, biomarker description /
   interpretation): the LLM's ## / ### headings would otherwise render at the
   same size as the tab's own subheader and dwarf the prose, so they are scaled
   down to a hierarchy *inside* the card. The first heading gets no top margin
   (the card already has padding), and Streamlit's hover "anchor" icon is
   useless inside a card, so it is hidden. */
.bb-card h2 { font-size: 1.15rem; margin: 1.1rem 0 0.3rem; }
.bb-card h3 { font-size: 1.02rem; margin: 0.9rem 0 0.25rem; }
.bb-card h4 { font-size: 0.95rem; margin: 0.75rem 0 0.2rem; }
.bb-card > div:first-child > h2,
.bb-card > div:first-child > h3,
.bb-card > div:first-child > h4 { margin-top: 0; }
.bb-card [data-testid="stHeaderActionElements"] { display: none; }
.bb-card p { margin: 0.35rem 0; }
/* The "> Belegende Werte: ..." evidence lines of the health summary: visually
   secondary to the plain-language sentence above them — smaller, dimmed, and
   marked with a thin rule instead of Streamlit's default quote bar. */
.bb-card blockquote {
    margin: 0.2rem 0 0.7rem;
    padding: 0.05rem 0 0.05rem 0.7rem;
    border-left: 2px solid rgba(128, 128, 128, 0.45);
    border-left: 2px solid color-mix(in srgb, currentColor 32%, transparent);
    font-size: 0.85rem;
    opacity: 0.75;
}
.bb-card blockquote p { margin: 0.15rem 0; }
/* Extra breathing room above section headings (st.subheader / st.header)
   so main sections of a tab are visually separated. */
section.stMain [data-testid="stHeading"] {
    margin-top: 1.5rem;
}
</style>
"""


def render_markdown_card(text, variant="info", summary=False):
    """Return the markdown string for a styled card containing LLM *Markdown*.

    The card div and the Markdown body must be separated from each other by
    blank lines: Streamlit's parser treats a chunk that starts with a raw HTML
    tag as one opaque HTML block, so f"<div ...>{text}</div>" printed the LLM's
    headings and paragraphs literally (the "## Gesamtbild" bug). With the blank
    lines, the div is a standalone HTML block and the body is parsed as
    Markdown — headings, paragraphs and blockquotes all render properly.

    The body is HTML-escaped first: the call site needs unsafe_allow_html=True
    for the wrapper div, so an unescaped "<" in LLM text ("Troponin < 0.01")
    would otherwise swallow the rest of the markup. Only "&" and "<" are
    escaped — that is enough to make tag injection impossible, while keeping
    Markdown's own syntax ("*" bold, ">" blockquote, "#" heading) intact.
    Escaping ">" too would turn the summary's "> Belegende Werte:" lines into
    literal text instead of blockquotes.

    Args:
        text: Markdown text (LLM output, stored in the DB).
        variant: severity/accent border — "info" (default), "warning", "critical".
        summary: add the roomier --summary padding.
    """
    classes = "bb-card"
    if summary:
        classes += " bb-card--summary"
    classes += f" bb-card--{variant}"
    body = (text or "").replace("&", "&amp;").replace("<", "&lt;").strip()
    return f"<div class='{classes}'>\n\n{body}\n\n</div>"


def inject_card_css():
    """Inject the shared card CSS into the page.

    Idempotent by design: Streamlit re-runs the whole script on every
    interaction, so calling this once per run (at a stable place in main())
    is all that's needed — no session-state guard required.
    """
    st.markdown(CARD_CSS, unsafe_allow_html=True)
