"""
AI-Based Smart Career Recommendation System
Professional Website-Style Streamlit Application

Compatible with existing model files:
    model/career_model.pkl
    model/scaler.pkl
    model/label_encoders.pkl
    model/target_encoder.pkl
    model/selected_features.pkl

Run:
    python -m streamlit run app.py
"""

import os
import html
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart Career AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# PROFESSIONAL UI CSS
# No custom HTML cards are used in the app body, so raw HTML
# will not appear on screen.
# ============================================================

st.markdown(
    """
    <style>
    /* ---------- Global ---------- */
    :root {
        --bg: #07111f;
        --panel: #0d1b2d;
        --panel-2: #10233a;
        --text: #f8fafc;
        --muted: #9fb0c5;
        --line: rgba(148,163,184,.16);
        --blue: #38bdf8;
        --indigo: #6366f1;
        --purple: #a855f7;
        --green: #22c55e;
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(99,102,241,.18), transparent 28%),
            radial-gradient(circle at 90% 10%, rgba(56,189,248,.14), transparent 25%),
            radial-gradient(circle at 50% 100%, rgba(168,85,247,.12), transparent 30%),
            var(--bg);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(7,17,31,.72);
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1420px;
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
    }

    /* ---------- Typography ---------- */
    h1, h2, h3, h4, p, label, .stMarkdown {
        color: var(--text);
    }

    .muted {
        color: var(--muted) !important;
    }

    /* ---------- Hero ---------- */
    .hero-title {
        font-size: clamp(2.3rem, 5vw, 4.7rem);
        line-height: 1.02;
        font-weight: 850;
        letter-spacing: -0.055em;
        margin: 0;
        background: linear-gradient(90deg, #ffffff, #8be7ff, #a78bfa, #ffffff);
        background-size: 250% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: titleShimmer 7s linear infinite;
    }

    .hero-subtitle {
        color: #aebed1 !important;
        font-size: 1.08rem;
        max-width: 760px;
        line-height: 1.7;
        margin-top: .8rem;
    }

    @keyframes titleShimmer {
        0% { background-position: 0% center; }
        100% { background-position: 250% center; }
    }

    .eyebrow {
        display: inline-block;
        color: #c4b5fd !important;
        background: rgba(99,102,241,.12);
        border: 1px solid rgba(129,140,248,.25);
        border-radius: 999px;
        padding: .42rem .8rem;
        font-size: .78rem;
        font-weight: 750;
        letter-spacing: .08em;
        text-transform: uppercase;
    }

    /* ---------- Native containers ---------- */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(
            145deg,
            rgba(16,35,58,.92),
            rgba(10,25,42,.86)
        );
        border: 1px solid var(--line);
        border-radius: 20px;
        box-shadow: 0 18px 60px rgba(0,0,0,.20);
        transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease;
    }

    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(99,102,241,.34);
        box-shadow: 0 24px 70px rgba(0,0,0,.28);
    }

    /* ---------- Metrics ---------- */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,.035);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: .9rem 1rem;
    }

    [data-testid="stMetricLabel"] {
        color: #91a4ba !important;
    }

    [data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-weight: 800;
    }

    /* ---------- Inputs ---------- */
    [data-baseweb="select"] > div,
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input {
        background: #0a1727 !important;
        border-color: rgba(148,163,184,.20) !important;
        color: #f8fafc !important;
        border-radius: 12px !important;
    }

    [data-testid="stSlider"] [role="slider"] {
        box-shadow: 0 0 0 4px rgba(99,102,241,.10);
    }

    /* ---------- Buttons ---------- */
    .stButton > button,
    .stFormSubmitButton > button {
        border: 0 !important;
        border-radius: 13px !important;
        min-height: 48px !important;
        font-weight: 800 !important;
        color: white !important;
        background: linear-gradient(100deg, #4f46e5, #7c3aed, #0284c7) !important;
        background-size: 220% auto !important;
        box-shadow: 0 12px 30px rgba(79,70,229,.25) !important;
        transition: all .25s ease !important;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        transform: translateY(-2px);
        background-position: 100% center !important;
        box-shadow: 0 18px 40px rgba(79,70,229,.35) !important;
    }

    /* ---------- Result ---------- */
    .result-text {
        text-align: center;
        padding: .7rem 0 1rem 0;
    }

    .result-icon {
        font-size: 4.4rem;
        animation: floatIcon 2.8s ease-in-out infinite;
        filter: drop-shadow(0 12px 20px rgba(56,189,248,.25));
    }

    .result-career {
        font-size: clamp(2rem, 4vw, 3.2rem);
        font-weight: 900;
        letter-spacing: -.035em;
        margin: .25rem 0 .5rem;
        background: linear-gradient(90deg, #67e8f9, #a78bfa, #f0abfc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    @keyframes floatIcon {
        0%,100% { transform: translateY(0) rotate(-2deg); }
        50% { transform: translateY(-9px) rotate(2deg); }
    }

    .result-glow {
        height: 2px;
        border-radius: 99px;
        background: linear-gradient(90deg, transparent, #38bdf8, #8b5cf6, transparent);
        margin: .7rem 0 1.3rem;
        animation: glowLine 2.2s ease-in-out infinite;
    }

    @keyframes glowLine {
        0%,100% { opacity: .45; }
        50% { opacity: 1; }
    }

    /* ---------- Progress ---------- */
    [data-testid="stProgressBar"] > div > div > div > div {
        background: linear-gradient(90deg, #38bdf8, #6366f1, #a855f7);
    }

    /* ---------- Tabs ---------- */
    button[data-baseweb="tab"] {
        color: #9fb0c5 !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #ffffff !important;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: #081321;
        border-right: 1px solid rgba(148,163,184,.12);
    }

    /* ---------- Footer ---------- */
    .footer-note {
        text-align: center;
        color: #71839a !important;
        padding: 1.5rem 0 .5rem;
        font-size: .82rem;
    }

    /* ---------- Small screens ---------- */
    @media (max-width: 900px) {
        [data-testid="stMainBlockContainer"] {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .hero-subtitle {
            font-size: .96rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL LOADING
# ============================================================

MODEL_DIR = "model"


@st.cache_resource(show_spinner=False)
def load_artifacts():
    paths = {
        "model": os.path.join(MODEL_DIR, "career_model.pkl"),
        "scaler": os.path.join(MODEL_DIR, "scaler.pkl"),
        "label_encoders": os.path.join(MODEL_DIR, "label_encoders.pkl"),
        "target_encoder": os.path.join(MODEL_DIR, "target_encoder.pkl"),
        "features": os.path.join(MODEL_DIR, "selected_features.pkl"),
    }

    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing model files:\n" + "\n".join(missing)
        )

    model = joblib.load(paths["model"])
    scaler = joblib.load(paths["scaler"])
    label_encoders = joblib.load(paths["label_encoders"])
    target_encoder = joblib.load(paths["target_encoder"])
    features = joblib.load(paths["features"])

    return model, scaler, label_encoders, target_encoder, features


try:
    model, scaler, label_encoders, target_encoder, selected_features = load_artifacts()
except Exception as exc:
    st.error("Model files could not be loaded.")
    st.code(str(exc))
    st.stop()


# ============================================================
# FEATURE ORDER
# ============================================================

# Prefer the exact order used by the trained model.
if hasattr(model, "feature_names_in_"):
    model_features = list(model.feature_names_in_)
else:
    model_features = list(selected_features)

# Exact numerical columns fitted by StandardScaler.
if hasattr(scaler, "feature_names_in_"):
    scaler_features = list(scaler.feature_names_in_)
else:
    # Fallback based on the user's existing project.
    scaler_features = [
        "Age",
        "CGPA",
        "Communication_Skill",
        "Problem_Solving",
        "Programming_Skill",
        "Leadership",
        "Certification_Count",
        "Projects_Completed",
        "Communication_Score",
        "Problem_Solving_Score",
        "Leadership_Score",
        "Creativity_Score",
        "Teamwork_Score",
    ]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def encoder_options(column_name, fallback):
    """Get exact classes from the saved LabelEncoder."""
    encoder = label_encoders.get(column_name)

    if encoder is not None and hasattr(encoder, "classes_"):
        return [str(x) for x in encoder.classes_]

    return fallback


def safe_encode(df_input):
    """
    Encode categorical columns using the exact saved LabelEncoders.
    Raises a clear error instead of silently changing unseen values.
    """
    result = df_input.copy()

    for col, encoder in label_encoders.items():
        if col not in result.columns:
            continue

        value = result[col].iloc[0]

        classes = list(getattr(encoder, "classes_", []))

        if value not in classes:
            raise ValueError(
                f"'{value}' is not a valid value for '{col}'. "
                f"Allowed values: {classes}"
            )

        result[col] = encoder.transform([value])[0]

    return result


def prepare_input(student_data):
    """Prepare one student row exactly as the model expects."""
    df_input = pd.DataFrame([student_data])

    missing_features = [
        col for col in model_features if col not in df_input.columns
    ]
    if missing_features:
        raise ValueError(
            "Missing model features: " + ", ".join(missing_features)
        )

    df_input = df_input[model_features].copy()

    # Encode categorical fields.
    df_input = safe_encode(df_input)

    # Scale ONLY the columns used to fit StandardScaler.
    missing_scaler = [
        col for col in scaler_features if col not in df_input.columns
    ]
    if missing_scaler:
        raise ValueError(
            "Missing scaler features: " + ", ".join(missing_scaler)
        )

    df_input[scaler_features] = scaler.transform(
        df_input[scaler_features]
    )

    # Restore exact model order.
    return df_input[model_features]


def predict(student_data):
    """Return career, confidence, probabilities and feature importance."""
    prepared = prepare_input(student_data)

    prediction = model.predict(prepared)
    career = target_encoder.inverse_transform(prediction)[0]

    confidence = None
    probabilities = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(prepared)[0]
        confidence = float(np.max(probabilities) * 100)

    importance = None
    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_)

    return career, confidence, probabilities, importance


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🎯 Smart Career AI")
    st.caption("Machine Learning Career Recommendation")

    st.divider()

    st.markdown("### 📌 Model Overview")
    st.write(f"**Model features:** {len(model_features)}")
    st.write(f"**Scaled features:** {len(scaler_features)}")
    st.write(f"**Career classes:** {len(target_encoder.classes_)}")

    st.divider()

    st.markdown("### ⚙️ Pipeline")
    st.write("1. Student profile")
    st.write("2. Label encoding")
    st.write("3. Numerical scaling")
    st.write("4. ML classification")
    st.write("5. Career recommendation")

    st.divider()

    st.caption("MCA Machine Learning Project • 2026")


# ============================================================
# HERO
# ============================================================

st.markdown(
    '<div class="eyebrow">🚀 MACHINE LEARNING • CAREER INTELLIGENCE</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero-title">Find the career that fits your potential.</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-subtitle">
        A professional machine-learning career recommendation system that
        evaluates academic performance, skills, certifications, projects,
        experience and work preferences to suggest a suitable career path.
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# ============================================================
# TOP METRICS
# ============================================================

mc1, mc2, mc3, mc4 = st.columns(4)

with mc1:
    st.metric("Model Features", len(model_features), "Input signals")

with mc2:
    st.metric("Scaled Features", len(scaler_features), "Normalized")

with mc3:
    st.metric("Career Classes", len(target_encoder.classes_), "Recommendations")

with mc4:
    st.metric("Engine", "ML", "Classification")


st.write("")


# ============================================================
# MAIN LAYOUT
# ============================================================

form_col, result_col = st.columns([1.35, 0.85], gap="large")


# ============================================================
# FORM
# ============================================================

with form_col:

    with st.container(border=True):
        st.markdown("## 👤 Student Assessment")
        st.caption(
            "Complete the profile below. The saved ML model will use the same "
            "feature order and preprocessing used during training."
        )

        with st.form("career_form", clear_on_submit=False):

            # ------------------------------------------------
            # PERSONAL
            # ------------------------------------------------
            st.markdown("### 👤 Personal Profile")

            c1, c2 = st.columns(2)

            with c1:
                age = st.number_input(
                    "Age",
                    min_value=18,
                    max_value=35,
                    value=22,
                    step=1,
                )

            with c2:
                gender = st.selectbox(
                    "Gender",
                    encoder_options(
                        "Gender",
                        ["Female", "Male", "Other"],
                    ),
                )

            # ------------------------------------------------
            # ACADEMIC
            # ------------------------------------------------
            st.markdown("### 🎓 Academic Profile")

            c1, c2 = st.columns(2)

            with c1:
                ug_course = st.selectbox(
                    "UG Course",
                    encoder_options(
                        "UG_Course",
                        [
                            "B.Sc Computer Science",
                            "B.Sc IT",
                            "B.Tech",
                            "BCA",
                            "M.Tech",
                            "MCA",
                        ],
                    ),
                )

            with c2:
                ug_specialization = st.selectbox(
                    "UG Specialization",
                    encoder_options(
                        "UG_Specialization",
                        [
                            "AI & ML",
                            "Computer Science",
                            "Cyber Security",
                            "Data Science",
                            "Information Technology",
                            "Software Engineering",
                        ],
                    ),
                )

            cgpa = st.slider(
                "CGPA",
                min_value=5.0,
                max_value=10.0,
                value=7.5,
                step=0.1,
            )

            # ------------------------------------------------
            # CERTIFICATIONS / EXPERIENCE
            # ------------------------------------------------
            st.markdown("### 💼 Certifications & Experience")

            c1, c2 = st.columns(2)

            with c1:
                certification = st.selectbox(
                    "Certification",
                    encoder_options(
                        "Certification",
                        [
                            "AWS Certification",
                            "Cyber Security Certification",
                            "Google Data Analytics",
                            "Machine Learning Certification",
                            "Power BI Certification",
                            "Python Certification",
                            "UI/UX Certification",
                        ],
                    ),
                )

            with c2:
                internship = st.selectbox(
                    "Internship",
                    encoder_options(
                        "Internship",
                        ["No", "Yes"],
                    ),
                )

            internship_exp = st.selectbox(
                "Internship Experience",
                encoder_options(
                    "Internship_Experience",
                    ["0-6 Months", "1+ Year", "6-12 Months"],
                ),
            )

            c1, c2 = st.columns(2)

            with c1:
                cert_count = st.number_input(
                    "Certification Count",
                    min_value=0,
                    max_value=10,
                    value=2,
                    step=1,
                )

            with c2:
                projects = st.number_input(
                    "Projects Completed",
                    min_value=0,
                    max_value=10,
                    value=3,
                    step=1,
                )

            # ------------------------------------------------
            # SKILLS
            # ------------------------------------------------
            st.markdown("### 🧠 Core Skills")
            st.caption("Rate your current skill level from 1 to 5.")

            c1, c2 = st.columns(2)

            with c1:
                comm_skill = st.slider(
                    "Communication Skill",
                    1, 5, 3,
                )
                prog_skill = st.slider(
                    "Programming Skill",
                    1, 5, 3,
                )

            with c2:
                prob_skill = st.slider(
                    "Problem Solving",
                    1, 5, 3,
                )
                leadership = st.slider(
                    "Leadership",
                    1, 5, 3,
                )

            # ------------------------------------------------
            # DETAILED SCORES
            # ------------------------------------------------
            st.markdown("### 📊 Detailed Skill Scores")
            st.caption("Rate each area from 1 to 10.")

            c1, c2 = st.columns(2)

            with c1:
                comm_score = st.slider(
                    "Communication Score",
                    1, 10, 6,
                )
                prob_score = st.slider(
                    "Problem Solving Score",
                    1, 10, 6,
                )
                lead_score = st.slider(
                    "Leadership Score",
                    1, 10, 6,
                )

            with c2:
                creativity_score = st.slider(
                    "Creativity Score",
                    1, 10, 6,
                )
                teamwork_score = st.slider(
                    "Teamwork Score",
                    1, 10, 6,
                )

            # ------------------------------------------------
            # PREFERENCES
            # ------------------------------------------------
            st.markdown("### 🚀 Career Preferences")

            c1, c2 = st.columns(2)

            with c1:
                work_pref = st.selectbox(
                    "Work Preference",
                    encoder_options(
                        "Work_Preference",
                        ["Hybrid", "On-site", "Remote"],
                    ),
                )

            with c2:
                work_env = st.selectbox(
                    "Preferred Work Environment",
                    encoder_options(
                        "Preferred_Work_Environment",
                        ["Hybrid", "Office", "Remote"],
                    ),
                )

            masters_field = st.selectbox(
                "Masters Field",
                encoder_options(
                    "Masters_Field",
                    [
                        "AI & ML",
                        "Business Analytics",
                        "Computer Science",
                        "Cyber Security",
                        "Data Science",
                    ],
                ),
            )

            st.write("")

            submitted = st.form_submit_button(
                "🔮  ANALYZE MY CAREER",
                use_container_width=True,
            )


# ============================================================
# RESULT
# ============================================================

with result_col:

    if not submitted:
        with st.container(border=True):
            st.markdown("## 🔮 AI Recommendation")
            st.caption("Your result will appear here after the assessment.")

            st.markdown(
                """
                <div class="result-text">
                    <div class="result-icon">🧭</div>
                    <h3>Ready when you are.</h3>
                    <p class="muted">
                        Complete your student profile and run the ML analysis
                        to discover your recommended career.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.info(
                "The application uses your saved model and preprocessing "
                "artifacts. It does not retrain the model during prediction."
            )

    else:

        student_data = {
            "Age": age,
            "Gender": gender,
            "UG_Course": ug_course,
            "UG_Specialization": ug_specialization,
            "CGPA": cgpa,
            "Certification": certification,
            "Internship": internship,
            "Communication_Skill": comm_skill,
            "Problem_Solving": prob_skill,
            "Programming_Skill": prog_skill,
            "Leadership": leadership,
            "Preferred_Work_Environment": work_env,
            "Masters_Field": masters_field,
            "Internship_Experience": internship_exp,
            "Certification_Count": cert_count,
            "Projects_Completed": projects,
            "Communication_Score": comm_score,
            "Problem_Solving_Score": prob_score,
            "Leadership_Score": lead_score,
            "Creativity_Score": creativity_score,
            "Teamwork_Score": teamwork_score,
            "Work_Preference": work_pref,
        }

        with st.spinner("🧠 Analyzing your profile..."):

            try:
                career, confidence, probabilities, importance = predict(
                    student_data
                )

                st.session_state["last_prediction"] = career
                st.session_state["last_confidence"] = confidence

            except Exception as exc:
                st.error("Prediction could not be completed.")
                st.code(str(exc))
                st.stop()

        # Celebration
        st.balloons()

        with st.container(border=True):

            st.markdown("## ✨ Your AI Recommendation")

            st.markdown(
                f"""
                <div class="result-text">
                    <div class="result-icon">🏆</div>
                    <div class="muted">RECOMMENDED CAREER</div>
                    <div class="result-career">
                        {html.escape(str(career))}
                    </div>
                    <div class="result-glow"></div>
                    <p class="muted">
                        Your profile has been matched against the trained
                        career classification model.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if confidence is not None:

                st.markdown("### 🎯 Career Match")

                st.metric(
                    "Prediction Confidence",
                    f"{confidence:.1f}%",
                )

                st.progress(
                    min(max(confidence / 100, 0.0), 1.0)
                )

            st.divider()

            # ------------------------------------------------
            # TOP CAREER PROBABILITIES
            # ------------------------------------------------
            if probabilities is not None:

                st.markdown("### 📈 Career Probability")

                class_names = list(target_encoder.classes_)

                probability_df = pd.DataFrame(
                    {
                        "Career": class_names,
                        "Probability": probabilities * 100,
                    }
                ).sort_values(
                    "Probability",
                    ascending=False,
                ).head(5)

                for _, row in probability_df.iterrows():

                    st.write(
                        f"**{row['Career']}** — "
                        f"{row['Probability']:.1f}%"
                    )

                    st.progress(
                        min(
                            max(
                                float(row["Probability"]) / 100,
                                0.0,
                            ),
                            1.0,
                        )
                    )

            # ------------------------------------------------
            # FEATURE IMPORTANCE
            # ------------------------------------------------
            if importance is not None:

                st.divider()
                st.markdown("### 🔑 Top Model Factors")

                importance_df = pd.DataFrame(
                    {
                        "Feature": model_features,
                        "Importance": importance,
                    }
                ).sort_values(
                    "Importance",
                    ascending=False,
                ).head(7)

                for _, row in importance_df.iterrows():

                    score = float(row["Importance"])

                    st.write(
                        f"**{row['Feature']}**  "
                        f"• {score * 100:.1f}%"
                    )

                    st.progress(
                        min(max(score, 0.0), 1.0)
                    )

        # ----------------------------------------------------
        # PROFILE SUMMARY
        # ----------------------------------------------------

        with st.expander("👤 View Submitted Student Profile"):

            profile_df = pd.DataFrame(
                {
                    "Feature": list(student_data.keys()),
                    "Value": list(student_data.values()),
                }
            )

            st.dataframe(
                profile_df,
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# BOTTOM INFORMATION
# ============================================================

st.write("")

tab1, tab2, tab3 = st.tabs(
    ["💡 How It Works", "🧠 Model Inputs", "🎓 Career Classes"]
)

with tab1:
    st.write(
        "The application collects the student's academic, skill, "
        "experience and preference information. Categorical values are "
        "encoded with the saved LabelEncoders, numerical values are "
        "scaled with the saved StandardScaler, and the trained classifier "
        "produces the final career recommendation."
    )

with tab2:
    st.write(
        "The trained model expects these features:"
    )
    st.code(", ".join(model_features))

with tab3:
    careers = list(target_encoder.classes_)
    st.write("Career classes available in the saved target encoder:")
    st.write(" • ".join(careers))


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer-note">
        🎯 <b>AI-Based Smart Career Recommendation System</b><br>
        MCA Machine Learning Project • Python • Pandas • Scikit-learn • Streamlit<br>
        <br>
        Designed for educational and career exploration purposes.
    </div>
    """,
    unsafe_allow_html=True,
)
