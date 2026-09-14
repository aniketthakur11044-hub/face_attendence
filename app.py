import streamlit as st
import pandas as pd
import os
from datetime import datetime
import face_utils as fu

st.set_page_config(page_title="Attendance", page_icon="🕰", layout="centered")

# ---------------------------------------------------------------------------
# Styling — vintage / archive aesthetic
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,700;1,900&family=Special+Elite&family=Libre+Baskerville:wght@400;700&display=swap');

:root {
    --ink: #2B1B12;
    --panel: #3E2B1E;
    --paper: #F3E7CE;
    --paper-dim: #E4D5B4;
    --rust: #9C3B2C;
    --gold: #C9A227;
    --sage: #6E8266;
    --muted: #C9B694;
    --line: rgba(243,231,206,0.22);
}

.stApp {
    background-color: var(--ink);
    background-image:
        radial-gradient(ellipse at center, rgba(0,0,0,0) 40%, rgba(0,0,0,0.55) 100%),
        repeating-linear-gradient(0deg, rgba(255,255,255,0.015) 0px, rgba(255,255,255,0.015) 1px, transparent 1px, transparent 3px);
}
html, body, [class*="css"] { font-family: 'Libre Baskerville', serif; }
#MainMenu, footer, header { visibility: hidden; }

.stApp, .stApp p, .stApp li, .stApp label, .stApp div,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label {
    color: var(--paper);
}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
    color: var(--muted) !important;
    font-style: italic;
}
.stTextInput input, .stNumberInput input, .stDateInput input,
div[data-baseweb="input"] input, div[data-baseweb="select"] div {
    color: var(--paper) !important;
    background-color: var(--panel) !important;
}
div[data-baseweb="input"], div[data-baseweb="select"] > div, div[data-baseweb="popover"] {
    background-color: var(--panel) !important;
    border-color: var(--gold) !important;
}
::placeholder { color: var(--muted) !important; opacity: 0.8; }
[data-testid="stSlider"] label, [data-testid="stSlider"] div,
[data-testid="stTickBar"] * { color: var(--paper) !important; }
[data-testid="stAlert"] p { color: var(--ink) !important; font-weight: 700; font-style: normal; }
[data-testid="stCheckbox"] label p { color: var(--paper) !important; font-style: normal; }

[data-testid="stDataFrame"], [data-testid="stDataEditor"] {
    background-color: var(--paper) !important;
    border-radius: 2px;
    padding: 8px;
    border: 1px solid var(--gold);
}
[data-testid="stDataFrame"] *, [data-testid="stDataEditor"] * {
    color: var(--ink) !important;
    font-family: 'Libre Baskerville', serif !important;
}

.hero-mark {
    font-family: 'Special Elite', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--gold) !important;
    margin-bottom: 1.1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.hero-mark .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--rust); display: inline-block; }
.hero-title {
    font-family: 'Playfair Display', serif;
    font-style: italic;
    font-weight: 900;
    font-size: 2.9rem;
    line-height: 1.08;
    letter-spacing: -0.01em;
    color: var(--paper) !important;
    margin: 0 0 1.1rem 0;
    text-shadow: 2px 2px 0 rgba(0,0,0,0.25);
}
.hero-sub {
    font-family: 'Libre Baskerville', serif;
    font-size: 1rem;
    color: var(--muted) !important;
    max-width: 50ch;
    line-height: 1.6;
    margin-bottom: 1.6rem;
}

.scan-frame {
    position: relative; width: 100%; height: 64px;
    background: var(--panel); border: 2px solid var(--gold);
    outline: 1px solid rgba(201,162,39,0.35); outline-offset: 3px;
    overflow: hidden; margin: 0.5rem 0 2rem 0;
}
.scan-corner { position: absolute; width: 14px; height: 14px; border: 3px solid var(--gold); }
.scan-corner.tl { top: -2px; left: -2px; border-right: none; border-bottom: none; }
.scan-corner.tr { top: -2px; right: -2px; border-left: none; border-bottom: none; }
.scan-corner.bl { bottom: -2px; left: -2px; border-right: none; border-top: none; }
.scan-corner.br { bottom: -2px; right: -2px; border-left: none; border-top: none; }
.scan-line {
    position: absolute; left: 0; width: 100%; height: 2px; background: var(--gold);
    box-shadow: 0 0 10px var(--gold); animation: scan 2.4s ease-in-out infinite;
}
@keyframes scan { 0%, 100% { top: 6px; } 50% { top: calc(100% - 8px); } }
.scan-caption {
    position: absolute; bottom: 7px; left: 14px; font-size: 0.72rem;
    color: var(--muted) !important; font-family: 'Special Elite', monospace; letter-spacing: 0.03em;
}

.ticker-wrap {
    overflow: hidden; background: var(--gold); transform: skewY(-1deg);
    margin: 0 0 2.2rem 0; padding: 0.5rem 0;
    border-top: 2px solid var(--ink); border-bottom: 2px solid var(--ink);
}
.ticker-track { display: flex; white-space: nowrap; width: max-content; animation: marquee 14s linear infinite; }
.ticker-track span {
    font-family: 'Special Elite', monospace; font-size: 0.95rem; letter-spacing: 0.05em;
    text-transform: uppercase; color: var(--ink) !important; padding: 0 2rem;
}
@keyframes marquee { 0% { transform: translateX(0); } 100% { transform: translateX(-50%); } }

.stat-grid { display: flex; gap: 1.4rem; margin: 0 0 2.4rem 0; flex-wrap: wrap; }
.stat-card {
    background: var(--paper); border: 1px solid var(--ink);
    outline: 1px dashed var(--rust); outline-offset: -5px;
    padding: 1.1rem 1.3rem; flex: 1; min-width: 130px; box-shadow: 5px 5px 0 var(--rust);
}
.stat-card:nth-child(1) { transform: rotate(-1.2deg); }
.stat-card:nth-child(2) { transform: rotate(0.9deg); }
.stat-card:nth-child(3) { transform: rotate(-0.5deg); }
.stat-card:nth-child(4) { transform: rotate(0.7deg); }
.stat-num {
    font-family: 'Playfair Display', serif; font-style: italic; font-weight: 900;
    font-size: 2rem; color: var(--rust) !important; line-height: 1;
}
.stat-label {
    font-size: 0.78rem; color: var(--ink) !important; margin-top: 0.35rem;
    font-family: 'Special Elite', monospace; text-transform: uppercase; letter-spacing: 0.03em;
}

.roster-wrap { display: flex; flex-wrap: wrap; gap: 0.6rem; margin: 0 0 2.4rem 0; }
.chip {
    font-family: 'Special Elite', monospace; font-size: 0.8rem; padding: 0.35rem 0.9rem;
    border-radius: 2px; border: 1.5px solid var(--line); color: var(--muted) !important;
}
.chip.in { background: var(--gold); color: var(--ink) !important; border-color: var(--ink); }
.chip.out { background: var(--rust); color: var(--paper) !important; border-color: var(--ink); }

@keyframes stampIn {
    0% { transform: scale(2.2) rotate(-24deg); opacity: 0; }
    60% { transform: scale(0.92) rotate(-9deg); opacity: 1; }
    100% { transform: scale(1) rotate(-9deg); opacity: 1; }
}
.stamp {
    display: inline-block; font-family: 'Special Elite', monospace; font-size: 1.15rem;
    text-transform: uppercase; letter-spacing: 0.04em; color: var(--sage) !important;
    border: 4px double var(--sage); padding: 0.5rem 1.2rem; border-radius: 4px;
    margin: 0.3rem 0.4rem 0.3rem 0; transform: rotate(-9deg);
    animation: stampIn 0.5s ease-out; background: rgba(110,130,102,0.08);
}
.stamp.exit { color: var(--rust) !important; border-color: var(--rust); transform: rotate(7deg); background: rgba(156,59,44,0.08); }

.section-heading {
    font-family: 'Playfair Display', serif; font-weight: 700; font-style: italic; font-size: 1.35rem;
    color: var(--paper) !important; margin: 0 0 1rem 0; padding-top: 0.6rem; border-top: 2px double var(--gold);
}

.step-row { display: flex; gap: 1rem; align-items: flex-start; margin-bottom: 1.1rem; }
.step-num {
    font-family: 'Playfair Display', serif; font-weight: 900; color: var(--ink) !important;
    background: var(--gold); border-radius: 50%; width: 2.3rem; height: 2.3rem;
    display: flex; align-items: center; justify-content: center; flex-shrink: 0; border: 2px solid var(--ink);
}
.step-title { font-weight: 700; color: var(--paper) !important; margin-bottom: 0.15rem; font-family: 'Playfair Display', serif; }
.step-desc { color: var(--muted) !important; font-size: 0.9rem; line-height: 1.5; }

/* Buttons - use Streamlit's real primary/secondary kinds, not a div-wrap hack */
div[data-testid="stButton"] button {
    border-radius: 2px !important;
    font-family: 'Special Elite', monospace !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    width: 100%;
    transition: transform 0.08s ease, box-shadow 0.08s ease, background-color 0.08s ease;
}
div[data-testid="stButton"] button[kind="primary"] {
    background-color: var(--gold) !important;
    color: var(--ink) !important;
    border: 2px solid var(--ink) !important;
    box-shadow: 4px 4px 0 var(--rust) !important;
}
div[data-testid="stButton"] button[kind="primary"]:hover {
    background-color: var(--paper) !important;
    transform: translate(2px, 2px);
    box-shadow: 2px 2px 0 var(--rust) !important;
}
div[data-testid="stButton"] button[kind="secondary"] {
    background-color: transparent !important;
    color: var(--rust) !important;
    border: 2px solid var(--rust) !important;
    box-shadow: 4px 4px 0 var(--panel) !important;
}
div[data-testid="stButton"] button[kind="secondary"]:hover {
    background-color: var(--rust) !important;
    color: var(--paper) !important;
}
div[data-testid="stButton"] button:active { transform: translate(4px, 4px); box-shadow: 0 0 0 !important; }

.id-badge {
    display: inline-block; font-family: 'Special Elite', monospace; font-size: 0.68rem;
    letter-spacing: 0.04em; color: var(--ink) !important; background: var(--gold);
    padding: 0.15rem 0.5rem; border-radius: 2px; margin-right: 0.4rem; border: 1px solid var(--ink);
}

section[data-testid="stSidebar"] { background-color: var(--panel); border-right: 2px solid var(--gold); }
section[data-testid="stSidebar"] * { color: var(--paper) !important; font-family: 'Libre Baskerville', serif; }

.sidebar-title {
    font-family: 'Playfair Display', serif;
    font-style: italic;
    font-weight: 900;
    font-size: 1.5rem;
    line-height: 1.15;
    color: var(--gold) !important;
    padding: 0.4rem 0.2rem 0.8rem 0.2rem;
}
.sidebar-rule {
    border-top: 2px double var(--gold);
    margin: 0 0.2rem 1rem 0.2rem;
}

/* Sidebar nav buttons — vintage ledger-tab look, distinct from action buttons */
section[data-testid="stSidebar"] div[data-testid="stButton"] button {
    text-align: left !important;
    justify-content: flex-start !important;
    background-color: transparent !important;
    color: var(--muted) !important;
    border: none !important;
    border-left: 4px solid transparent !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    padding: 0.7rem 0.9rem !important;
    font-family: 'Special Elite', monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.01em;
    text-transform: none !important;
    margin-bottom: 0.2rem;
    transition: background-color 0.15s ease, border-color 0.15s ease, color 0.15s ease;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover {
    background-color: rgba(201,162,39,0.14) !important;
    color: var(--paper) !important;
    border-left-color: var(--gold) !important;
    transform: none !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"] {
    background-color: rgba(201,162,39,0.20) !important;
    color: var(--gold) !important;
    border-left: 4px solid var(--gold) !important;
    font-weight: 700;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] button:active {
    transform: none !important;
    box-shadow: none !important;
}

h1, h2, h3 { font-family: 'Playfair Display', serif; color: var(--paper) !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state — a single version counter that we bump any time the
# underlying data (people or attendance rows) changes shape, so widgets
# that depend on that data always get a fresh key instead of colliding
# with stale state (this was the root cause of buttons "not working").
# ---------------------------------------------------------------------------
if "menu" not in st.session_state:
    st.session_state.menu = "Home"
if "data_version" not in st.session_state:
    st.session_state.data_version = 0


def _bump_and_rerun():
    st.session_state.data_version += 1
    st.rerun()


def _go_to(page):
    st.session_state.menu = page


MENU_ITEMS = [
    ("Home", "🏠  Home"),
    ("Register New Person", "➕  Register New Person"),
    ("Manage People", "🗂️  Manage People"),
    ("Train Model", "⚙️  Train Model"),
    ("Take Attendance", "📷  Take Attendance"),
    ("View Attendance Records", "📜  View Attendance Records"),
]

st.sidebar.markdown("""
    <div class="sidebar-title">🕰 Attendance<br>Archive</div>
    <div class="sidebar-rule"></div>
""", unsafe_allow_html=True)

for page_key, page_label in MENU_ITEMS:
    is_active = st.session_state.menu == page_key
    if st.sidebar.button(page_label, key=f"nav_{page_key}", type="primary" if is_active else "secondary"):
        st.session_state.menu = page_key
        st.rerun()

menu = st.session_state.menu

# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------
if menu == "Home":
    people_df = fu.get_people()
    today_df = fu._load_today_df()

    in_ids = set(today_df[today_df["Type"] == "in"]["ID"].astype(str)) if not today_df.empty else set()
    out_ids = set(today_df[today_df["Type"] == "out"]["ID"].astype(str)) if not today_df.empty else set()

    checked_in_count = len(in_ids)
    checked_out_count = len(out_ids)
    last_scan = today_df["Time"].max() if not today_df.empty else "—"

    st.markdown("""
        <div class="hero-mark"><span class="dot"></span>ledger of record</div>
        <div class="hero-title">Attendance, kept<br>the old-fashioned way.</div>
        <div class="hero-sub">Point a camera at someone and they're checked in or out automatically —
        no ID cards to swipe, no queues, no paper roster.</div>
        <div class="scan-frame">
            <div class="scan-corner tl"></div>
            <div class="scan-corner tr"></div>
            <div class="scan-corner bl"></div>
            <div class="scan-corner br"></div>
            <div class="scan-line"></div>
            <div class="scan-caption">scanning for a face...</div>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.button("Take attendance", on_click=_go_to, args=("Take Attendance",), type="primary")
    with col2:
        st.button("Register someone new", on_click=_go_to, args=("Register New Person",))

    st.markdown("""
        <div class="ticker-wrap"><div class="ticker-track">
            <span>entered in the ledger</span><span>entered in the ledger</span><span>entered in the ledger</span>
            <span>entered in the ledger</span><span>entered in the ledger</span><span>entered in the ledger</span>
        </div></div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="stat-grid">
            <div class="stat-card"><div class="stat-num">{len(people_df)}</div><div class="stat-label">people registered</div></div>
            <div class="stat-card"><div class="stat-num">{checked_in_count}</div><div class="stat-label">checked in today</div></div>
            <div class="stat-card"><div class="stat-num">{checked_out_count}</div><div class="stat-label">checked out today</div></div>
            <div class="stat-card"><div class="stat-num">{last_scan}</div><div class="stat-label">last scan</div></div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Today\'s roster</div>', unsafe_allow_html=True)
    if not people_df.empty:
        chips = []
        for _, row in people_df.iterrows():
            pid, name = str(row["ID"]), row["Name"].replace("_", " ")
            state = "out" if pid in out_ids else ("in" if pid in in_ids else "")
            chips.append(f'<span class="chip {state}">{name}</span>')
        chips_html = "".join(chips)
    else:
        chips_html = '<span class="chip">No one registered yet</span>'
    st.markdown(f'<div class="roster-wrap">{chips_html}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-heading">How it works</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="step-row">
            <div class="step-num">1</div>
            <div><div class="step-title">Register a face + unique ID</div>
            <div class="step-desc">Give them an ID (like an employee/student number) and capture about 40 photos.</div></div>
        </div>
        <div class="step-row">
            <div class="step-num">2</div>
            <div><div class="step-title">Train the model</div>
            <div class="step-desc">Teach the system to tell registered faces apart.</div></div>
        </div>
        <div class="step-row">
            <div class="step-num">3</div>
            <div><div class="step-title">Scan to check in / check out</div>
            <div class="step-desc">Open the webcam, pick Check In or Check Out, and everyone recognized gets logged.</div></div>
        </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Register (Add new student / person)
# ---------------------------------------------------------------------------
elif menu == "Register New Person":
    st.markdown('<div class="hero-mark"><span class="dot"></span>register</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading" style="font-size:1.7rem;">Add a new person</div>', unsafe_allow_html=True)

    col_id, col_name = st.columns(2)
    with col_id:
        person_id = st.text_input("Unique ID (e.g. EMP001)", key=f"reg_id_{st.session_state.data_version}")
    with col_name:
        name = st.text_input("Name (e.g. john_doe)", key=f"reg_name_{st.session_state.data_version}")

    num_samples = st.slider("Number of face samples to capture", 20, 100, 40)
    st.caption("Clicking Start opens a webcam window. Look at the camera and move your head "
               "slightly for varied angles. Press Q anytime to stop early.")

    if person_id and fu.id_exists(person_id):
        st.warning(f"ID '{person_id}' is already taken — pick a different one.")

    if st.button("Start capturing", type="primary"):
        if not person_id.strip() or not name.strip():
            st.error("Please enter both a unique ID and a name.")
        elif fu.id_exists(person_id):
            st.error(f"ID '{person_id}' is already registered. Choose a unique ID.")
        else:
            with st.spinner("Opening webcam..."):
                success, msg = fu.capture_faces(person_id.strip(), name.strip(), num_samples)
            if success:
                st.success(msg)
                st.warning("Now go to 'Train Model' so the system learns this new face.")
                _bump_and_rerun()
            else:
                st.error(msg)

# ---------------------------------------------------------------------------
# Manage People (show data + delete student/person)
# ---------------------------------------------------------------------------
elif menu == "Manage People":
    st.markdown('<div class="hero-mark"><span class="dot"></span>manage</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading" style="font-size:1.7rem;">Registered people</div>', unsafe_allow_html=True)

    people_df = fu.get_people()
    if people_df.empty:
        st.warning("No one is registered yet.")
    else:
        st.dataframe(
            people_df[["ID", "Name", "Photos"]].rename(columns={"Photos": "Photo count"}),
            use_container_width=True, hide_index=True,
        )

        st.markdown('<div class="section-heading">Delete a person</div>', unsafe_allow_html=True)
        st.caption("Removes their photos and registry entry. Retrain the model afterwards.")

        id_to_delete = st.selectbox(
            "Select ID to delete",
            options=people_df["ID"].tolist(),
            key=f"select_del_{st.session_state.data_version}",
        )
        confirm = st.checkbox(
            "I understand this permanently deletes this person.",
            key=f"confirm_del_{st.session_state.data_version}",
        )
        if st.button("Delete this person", type="secondary"):
            if not confirm:
                st.error("Please check the confirmation box first.")
            else:
                success, msg = fu.delete_person(id_to_delete)
                if success:
                    st.success(msg)
                    _bump_and_rerun()
                else:
                    st.error(msg)

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
elif menu == "Train Model":
    st.markdown('<div class="hero-mark"><span class="dot"></span>train</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading" style="font-size:1.7rem;">Train the recognition model</div>', unsafe_allow_html=True)
    st.caption("Run this every time you register or delete a person, or add more images.")
    if st.button("Train now", type="primary"):
        with st.spinner("Training..."):
            success, msg = fu.train_model()
        if success:
            st.success(msg)
        else:
            st.error(msg)

# ---------------------------------------------------------------------------
# Take Attendance (Check In / Check Out)
# ---------------------------------------------------------------------------
elif menu == "Take Attendance":
    st.markdown('<div class="hero-mark"><span class="dot"></span>scan</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading" style="font-size:1.7rem;">Take attendance</div>', unsafe_allow_html=True)
    st.caption("Pick Check In when someone arrives, Check Out when they leave. The webcam runs for "
               "about 20 seconds; press Q to stop early.")

    col_in, col_out = st.columns(2)
    with col_in:
        do_in = st.button("Check In", type="primary")
    with col_out:
        do_out = st.button("Check Out", type="secondary")

    if do_in or do_out:
        action = "in" if do_in else "out"
        with st.spinner("Scanning faces..."):
            marked, msg = fu.recognize_and_mark(action=action)

        if marked:
            if action == "in":
                stamps = "".join(f'<div class="stamp">✓ {n.replace("_", " ")} — Arrived</div>' for _, n in marked)
                st.markdown(stamps, unsafe_allow_html=True)
                st.balloons()
            else:
                stamps = "".join(f'<div class="stamp exit">🚪 {n.replace("_", " ")} — Departed</div>' for _, n in marked)
                st.markdown(stamps, unsafe_allow_html=True)
                st.snow()
            _bump_and_rerun()
        else:
            st.warning("No one new was marked.")
            st.caption(msg)

# ---------------------------------------------------------------------------
# Records (show data + delete records)
# ---------------------------------------------------------------------------
elif menu == "View Attendance Records":
    st.markdown('<div class="hero-mark"><span class="dot"></span>records</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading" style="font-size:1.7rem;">Attendance records</div>', unsafe_allow_html=True)

    date = st.date_input("Select date", datetime.now())
    df = fu.load_attendance_for_date(date)

    if df.empty:
        st.warning("No attendance record found for this date.")
    else:
        st.caption("Tick the rows you want to remove, then click 'Delete selected rows'.")
        display_df = df.copy()
        display_df.insert(0, "Delete", False)

        editor_key = f"editor_{date}_{st.session_state.data_version}"
        edited_df = st.data_editor(
            display_df,
            use_container_width=True,
            hide_index=False,
            disabled=["ID", "Name", "Type", "Time"],
            key=editor_key,
        )

        col_del, col_dl = st.columns(2)
        with col_del:
            if st.button("Delete selected rows", type="secondary"):
                rows_to_delete = edited_df.index[edited_df["Delete"]].tolist()
                if not rows_to_delete:
                    st.error("No rows selected.")
                else:
                    success, msg = fu.delete_attendance_rows(date, rows_to_delete)
                    if success:
                        st.success(msg)
                        _bump_and_rerun()
                    else:
                        st.error(msg)
        with col_dl:
            st.download_button("Download CSV", df.to_csv(index=False), file_name=f"attendance_{date}.csv")

        st.markdown('<div class="section-heading">Danger zone</div>', unsafe_allow_html=True)
        confirm_all = st.checkbox(
            f"I understand this deletes ALL records for {date}.",
            key=f"confirm_all_{date}_{st.session_state.data_version}",
        )
        if st.button("Delete ALL records for this date", type="secondary"):
            if not confirm_all:
                st.error("Please check the confirmation box first.")
            else:
                success, msg = fu.delete_all_attendance(date)
                if success:
                    st.success(msg)
                    _bump_and_rerun()
                else:
                    st.error(msg)