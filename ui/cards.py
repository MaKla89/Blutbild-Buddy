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

Only CSS variables that the original inline styles used are referenced, so
light and dark themes behave exactly as before.
"""

import streamlit as st

CARD_CSS = """
<style>
.bb-card {
    padding: 0.5rem;
    margin: 0.25rem 0;
    border-left: 4px solid #42a5f5;
    background: var(--secondary-background-color);
    color: var(--text-color);
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
    border-top: 1px solid var(--border-color);
    font-size: 0.85em;
    opacity: 0.75;
}
/* Extra breathing room above section headings (st.subheader / st.header)
   so main sections of a tab are visually separated. */
section.stMain [data-testid="stHeading"] {
    margin-top: 1.5rem;
}
</style>
"""


def inject_card_css():
    """Inject the shared card CSS into the page.

    Idempotent by design: Streamlit re-runs the whole script on every
    interaction, so calling this once per run (at a stable place in main())
    is all that's needed — no session-state guard required.
    """
    st.markdown(CARD_CSS, unsafe_allow_html=True)
