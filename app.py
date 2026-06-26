import streamlit as st
import pandas as pd
import json
import os
import html
import numpy as np
import requests
from io import StringIO
from dotenv import load_dotenv
import base64

# Define available datasets
DATASETS = {
    # "Charles": "dataset/label_Charles.csv",
    "Iury": "dataset/label_Iury.csv",
    "Jessica": "dataset/label_Jessica.csv",
    # "Jose": "dataset/label_Jose.csv",
    # "Lukas": "dataset/label_Lukas.csv",
}

# ---------------------------------------------------------------------------
# Page config + criteria
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Issue Report Labeling System", layout="wide")
st.title("🐛 Issue Report Labeling System")

CRITERIA_STANDARD = """
Cell contains most of the following key values:

1. Issue triggering steps
2. Stack traces (for crashes/errors)
3. Product/component version
4. Hardware/configuration info
5. Clear, descriptive title/summary
6. Related software versions
7. Test cases demonstrating the issue
8. Code examples showing the problem
9. Precise, unambiguous language
10. Operating system details
"""

CRITERIA_NOT_STANDARD = """
Missing the above criteria. Such as:

- Missing or unclear reproduction steps.
- Unclear core issue.
- Lacks technical evidence.
- Vague or ambiguous language.
"""

# ---------------------------------------------------------------------------
# Sidebar: labeler selection + criteria
# ---------------------------------------------------------------------------
st.sidebar.header("Labeler")
selected_user = st.sidebar.selectbox("Select User:", list(DATASETS.keys()))

with st.sidebar.expander("📋 Criteria for Standard", expanded=False):
    st.markdown(CRITERIA_STANDARD)
with st.sidebar.expander("📋 Criteria for Not Standard", expanded=False):
    st.markdown(CRITERIA_NOT_STANDARD)

# Load the selected dataset
CSV_FILE = DATASETS[selected_user]


# @st.cache_data
def load_data(csv_file):
    if not os.path.exists(csv_file):
        st.error(f"⚠️ File {csv_file} not found!")
        return pd.DataFrame(columns=["body", "label", "reason"])

    df = pd.read_csv(csv_file)

    # Extract "html_url" safely from JSON in "body" column
    def extract_url(json_str):
        try:
            data = json.loads(json_str)
            return data.get("html_url", "")
        except (json.JSONDecodeError, TypeError):
            return ""

    df["html_url"] = df["body"].apply(extract_url)

    # Ensure label and reason columns exist
    df["label"] = df.get("label", np.nan)
    df["reason"] = df.get("reason", "").fillna("")

    return df


df = load_data(CSV_FILE)

# ---------------------------------------------------------------------------
# Per-user session store (persists across navigation; NOT reset each rerun)
# ---------------------------------------------------------------------------
store_key = f"store::{selected_user}"
if store_key not in st.session_state:
    st.session_state[store_key] = {
        "label": {i: df.at[i, "label"] for i in df.index},
        "reason": {i: df.at[i, "reason"] for i in df.index},
    }
store = st.session_state[store_key]

# Current report position for this user
pos_key = f"pos::{selected_user}"
if pos_key not in st.session_state:
    st.session_state[pos_key] = 0


def save_data_to_github(df_to_save, token, repo, path):
    headers = {"Authorization": f"token {token}"}

    # Convert the updated dataframe to CSV string
    updated_csv = df_to_save.to_csv(index=False)

    # Get file sha before updating it
    sha_url = f"https://api.github.com/repos/{repo}/contents/{path}"
    sha_response = requests.get(sha_url, headers=headers)

    try:
        sha_data = sha_response.json()
        sha = sha_data.get('sha')
    except requests.exceptions.JSONDecodeError as e:
        st.error(f"⚠️ Error decoding JSON in SHA response: {str(e)}")
        st.write(f"Raw SHA Response: {sha_response.text}")
        return False

    if not sha:
        st.error(f"⚠️ SHA not found in the response. Please check the file path.")
        return False

    # Prepare the payload to update the file in GitHub
    update_payload = {
        "message": "Update dataset with new labels and reasons",
        "content": base64.b64encode(updated_csv.encode('utf-8')).decode('utf-8'),
        "sha": sha
    }

    # Send PUT request to update the file
    update_url = f"https://api.github.com/repos/{repo}/contents/{path}"
    update_response = requests.put(update_url, headers=headers, json=update_payload)

    if update_response.status_code in [200, 201]:
        return True
    else:
        st.error(f"⚠️ Error saving data: {update_response.json().get('message', 'Unknown error')}")
        st.write(update_response.json())
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def safe_str(val):
    """Coerce nan/None to empty string for widget values."""
    if val is None:
        return ""
    if isinstance(val, float) and pd.isna(val):
        return ""
    return str(val)


def text_color_for(bg_hex):
    """Pick black/white text for readability on a label color."""
    try:
        r = int(bg_hex[0:2], 16)
        g = int(bg_hex[2:4], 16)
        b = int(bg_hex[4:6], 16)
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        return "#000000" if lum > 150 else "#ffffff"
    except Exception:
        return "#000000"


def commit_widget_to_store(pos):
    """Read the currently displayed report's widgets into the session store."""
    choice = st.session_state.get(f"radio::{selected_user}::{pos}", "Unlabelled")
    if choice == "Standard":
        store["label"][pos] = 1
    elif choice == "Not Standard":
        store["label"][pos] = 0
    else:
        store["label"][pos] = np.nan
    store["reason"][pos] = st.session_state.get(f"reason::{selected_user}::{pos}", "")


def push_to_github():
    try:
        token = st.secrets["GITHUB_TOKEN"]
    except Exception:
        token = None
    if not token:
        st.error("⚠️ GitHub token not found. Please set the GITHUB_TOKEN secret.")
        return False

    # Sync the full dataframe from the session store before saving
    for i in df.index:
        df.at[i, "label"] = store["label"].get(i, np.nan)
        df.at[i, "reason"] = store["reason"].get(i, "")

    repo = "ttasnim68/LLM_label_app"
    path = "dataset/label_" + selected_user + ".csv"
    return save_data_to_github(df, token, repo, path)


# ---------------------------------------------------------------------------
# One-by-one GitHub-style report viewer
# ---------------------------------------------------------------------------
N = len(df.index)

if N == 0:
    st.warning("No reports to display.")
    st.stop()

# Clamp position
pos = int(st.session_state[pos_key])
pos = max(0, min(pos, N - 1))
st.session_state[pos_key] = pos

# Sidebar progress
labelled = sum(1 for i in df.index if not (isinstance(store["label"].get(i), float) and pd.isna(store["label"].get(i))))
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Progress:** {labelled} / {N} labeled")
st.sidebar.progress(labelled / N if N else 0.0)

# Header line + jump-to control
top_l, top_r = st.columns([3, 2])
with top_l:
    st.markdown(f"#### Report {pos + 1} of {N} &nbsp;·&nbsp; **{selected_user}**")
with top_r:
    jc1, jc2 = st.columns([2, 1])
    with jc1:
        goto = st.number_input("Go to #", min_value=1, max_value=N, value=pos + 1, step=1,
                               key=f"goto::{selected_user}::{pos}")
    with jc2:
        st.write("")
        st.write("")
        if st.button("Jump", use_container_width=True):
            commit_widget_to_store(pos)
            st.session_state[pos_key] = int(goto) - 1
            st.rerun()

# Parse the current report JSON
try:
    data = json.loads(df.at[pos, "body"])
except Exception:
    data = {}

state = data.get("state", "")
is_pr = "pull_request" in data
state_color = "#6f42c1" if state == "closed" else "#2da44e"
state_label = ("PR · " if is_pr else "") + (state or "unknown")

user = data.get("user") or {}
login = html.escape(str(user.get("login", "unknown")))
avatar = user.get("avatar_url", "")
uhtml = user.get("html_url", "#")

title = html.escape(str(data.get("title", "(no title)")))
number = data.get("number", "")
created = (data.get("created_at") or "")[:10]
comments = data.get("comments", 0)
assoc = html.escape(str(data.get("author_association", "")))
project = html.escape(str(df.at[pos, "project"])) if "project" in df.columns else ""
url = df.at[pos, "html_url"] or "#"

pills = ""
for l in (data.get("labels") or []):
    c = (l.get("color") or "ededed")
    name = html.escape(str(l.get("name", "")))
    fg = text_color_for(c)
    pills += (f'<span style="background:#{c};color:{fg};padding:2px 10px;border-radius:12px;'
              f'font-size:12px;margin-right:6px;display:inline-block;margin-bottom:4px;">{name}</span>')

header_html = f"""
<div style="border:1px solid #d0d7de;border-radius:8px;padding:16px;margin-bottom:10px;background:#ffffff;">
  <div style="margin-bottom:8px;">
    <span style="background:{state_color};color:#fff;padding:3px 12px;border-radius:14px;font-size:12px;font-weight:600;">● {state_label}</span>
    <span style="color:#57606a;font-size:13px;margin-left:8px;">{project} &nbsp;·&nbsp; #{number}</span>
  </div>
  <div style="font-size:22px;font-weight:600;line-height:1.3;margin-bottom:10px;color:#1f2328;">{title}</div>
  <div style="display:flex;align-items:center;margin-bottom:10px;flex-wrap:wrap;">
    <img src="{avatar}" style="width:22px;height:22px;border-radius:50%;margin-right:6px;">
    <a href="{uhtml}" target="_blank" style="font-weight:600;text-decoration:none;margin-right:8px;color:#0969da;">{login}</a>
    <span style="color:#57606a;font-size:13px;">opened on {created} &nbsp;·&nbsp; 💬 {comments} comments &nbsp;·&nbsp; {assoc}</span>
  </div>
  <div style="margin-bottom:8px;">{pills}</div>
  <div><a href="{url}" target="_blank" style="font-size:13px;color:#0969da;">🔗 Open on GitHub</a></div>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

# Body (scrollable)
st.markdown("**Description**")
body_text = data.get("body") or "_(no description provided)_"
with st.container(height=400, border=True):
    st.markdown(body_text)

# ---------------------------------------------------------------------------
# Labeling controls
# ---------------------------------------------------------------------------
st.markdown("### Your Label")

cur = store["label"].get(pos)
if cur == 1:
    default_idx = 1
elif cur == 0:
    default_idx = 2
else:
    default_idx = 0

OPTIONS = ["Unlabelled", "Standard", "Not Standard"]
ICONS = {"Unlabelled": "⬜ Unlabelled", "Standard": "✅ Standard", "Not Standard": "❌ Not Standard"}

st.radio(
    "Classification",
    OPTIONS,
    index=default_idx,
    format_func=lambda x: ICONS[x],
    horizontal=True,
    key=f"radio::{selected_user}::{pos}",
)

st.text_area(
    "Reason / notes (optional)",
    value=safe_str(store["reason"].get(pos, "")),
    key=f"reason::{selected_user}::{pos}",
    height=100,
)

# ---------------------------------------------------------------------------
# Navigation buttons
# ---------------------------------------------------------------------------
b1, b2, b3, b4 = st.columns([1, 1, 1.4, 1.4])

with b1:
    prev_clicked = st.button("◀ Previous", use_container_width=True, disabled=(pos == 0))
with b2:
    next_clicked = st.button("Skip / Next ▶", use_container_width=True, disabled=(pos == N - 1))
with b3:
    savenext_clicked = st.button("💾 Save & Next", type="primary", use_container_width=True)
with b4:
    saveall_clicked = st.button("⬆️ Save all to GitHub", use_container_width=True)

if prev_clicked:
    commit_widget_to_store(pos)
    st.session_state[pos_key] = max(0, pos - 1)
    st.rerun()

if next_clicked:
    commit_widget_to_store(pos)
    st.session_state[pos_key] = min(N - 1, pos + 1)
    st.rerun()

if savenext_clicked:
    commit_widget_to_store(pos)
    if push_to_github():
        st.success(f"✅ Saved report {pos + 1}.")
        if pos < N - 1:
            st.session_state[pos_key] = pos + 1
        st.rerun()

if saveall_clicked:
    commit_widget_to_store(pos)
    if push_to_github():
        st.success("✅ All labels saved to GitHub!")
