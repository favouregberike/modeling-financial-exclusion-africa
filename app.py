import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Inclusion Predictor · Africa",
    page_icon="🌍",
    layout="centered",
)

# ── Constants (from Findex 2021 Sub-Saharan Africa distributions) ─────────────
COUNTRIES = [
    "Nigeria", "Ghana", "Kenya", "South Africa", "Ethiopia", "Tanzania",
    "Uganda", "Rwanda", "Senegal", "Côte d'Ivoire", "Cameroon", "Zimbabwe",
    "Zambia", "Mozambique", "Mali", "Burkina Faso", "Niger", "Chad",
    "Madagascar", "Malawi", "Angola", "Togo", "Benin", "Congo, Rep.",
    "Congo, Dem. Rep.", "Guinea"
]


EDUCATION_LEVELS = [
    "No formal education",
    "Primary",
    "Secondary",
    "Tertiary or above",
]

INCOME_QUINTILES = [
    "Poorest 20%",
    "Second 20%",
    "Middle 20%",
    "Fourth 20%",
    "Richest 20%",
]

EMPLOYMENT_STATUS = [
    "Employed (full-time)",
    "Employed (part-time)",
    "Self-employed",
    "Out of workforce",
    "Student",
]

# Inclusion rates by country (Findex 2021 actuals for context labels)
COUNTRY_INCLUSION_RATE = {
    "Nigeria": 0.45, "Ghana": 0.57, "Kenya": 0.79, "South Africa": 0.85,
    "Ethiopia": 0.46, "Tanzania": 0.52, "Uganda": 0.48, "Rwanda": 0.93,
    "Senegal": 0.56, "Côte d'Ivoire": 0.41, "Cameroon": 0.36,
    "Zimbabwe": 0.53, "Zambia": 0.45, "Mozambique": 0.34, "Mali": 0.35,
    "Burkina Faso": 0.43, "Niger": 0.22, "Chad": 0.22, "Madagascar": 0.18,
    "Malawi": 0.36, "Angola": 0.43, "Togo": 0.42, "Benin": 0.38,
    "Congo, Rep.": 0.26, "Congo, Dem. Rep.": 0.26, "Guinea": 0.23
}
# Feature importance weights (derived from Findex research + notebook findings)
# Used to build a well-calibrated synthetic model
FEATURE_WEIGHTS = {
    "mobile_owned": 1.8,
    "income_quintile": 1.5,
    "internet_use": 1.3,
    "education": 1.2,
    "employed": 0.9,
    "urban": 0.8,
    "age_working": 0.7,
    "gender_male": 0.4,
    "country_ke_za": 1.1,
}


@st.cache_resource
def build_model():
    """
    Trains a GradientBoostingClassifier on synthetic data calibrated to
    World Bank Findex 2021 Sub-Saharan Africa inclusion patterns.
    Replace with: model = pickle.load(open('model.pkl','rb')) once you
    export the trained model from your notebook.
    """
    rng = np.random.default_rng(42)
    n = 4000

    countries  = rng.choice(COUNTRIES, n)
    female     = rng.integers(0, 2, n)           # 0=male, 1=female
    age        = rng.integers(15, 70, n)
    edu        = rng.integers(0, 4, n)            # 0–3
    inc_q      = rng.integers(0, 5, n)            # 0–4
    employed   = rng.integers(0, 2, n)
    urban      = rng.integers(0, 2, n)
    mobile     = rng.integers(0, 2, n)
    internet   = rng.integers(0, 2, n)

    # Calibrated inclusion probability
    base = np.array([COUNTRY_INCLUSION_RATE[c] for c in countries])
    logit = (
        np.log(base / (1 - base))
        + 1.8 * mobile
        + 1.5 * (inc_q / 4)
        + 1.3 * internet
        + 1.2 * (edu / 3)
        + 0.9 * employed
        + 0.8 * urban
        + 0.7 * np.where((age >= 25) & (age <= 55), 1, 0)
        - 0.4 * female
    )
    prob = 1 / (1 + np.exp(-logit))
    label = (rng.random(n) < prob).astype(int)

    country_enc = LabelEncoder().fit_transform(countries)

    X = np.column_stack([
        country_enc, female, age, edu, inc_q,
        employed, urban, mobile, internet
    ])

    model = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                        learning_rate=0.05, random_state=42)
    model.fit(X, label)

    country_le = LabelEncoder().fit(COUNTRIES)
    return model, country_le


model, country_le = build_model()


def predict(country, female, age, education, income_quintile,
            employed, urban, mobile_owned, internet_use):
    country_enc = country_le.transform([country])[0]
    X = np.array([[
        country_enc, int(female), age,
        EDUCATION_LEVELS.index(education),
        INCOME_QUINTILES.index(income_quintile),
        int(employed), int(urban),
        int(mobile_owned), int(internet_use)
    ]])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]
    return pred, prob


def feature_contributions(country, female, age, education, income_quintile,
                            employed, urban, mobile_owned, internet_use):
    """Returns a simple breakdown of which factors drive the prediction."""
    factors = []
    if mobile_owned:
        factors.append(("📱 Mobile phone ownership", "high", "+"))
    else:
        factors.append(("📱 No mobile phone", "high", "−"))

    inc_idx = INCOME_QUINTILES.index(income_quintile)
    if inc_idx >= 3:
        factors.append(("💰 Income level (top 40%)", "high", "+"))
    elif inc_idx <= 1:
        factors.append(("💰 Income level (bottom 40%)", "high", "−"))

    if internet_use:
        factors.append(("🌐 Internet access", "medium", "+"))
    else:
        factors.append(("🌐 No internet access", "medium", "−"))

    edu_idx = EDUCATION_LEVELS.index(education)
    if edu_idx >= 2:
        factors.append(("🎓 Secondary+ education", "medium", "+"))
    else:
        factors.append(("🎓 Below secondary education", "medium", "−"))

    if employed:
        factors.append(("💼 Employed", "medium", "+"))

    if urban:
        factors.append(("🏙️ Urban residence", "low", "+"))
    else:
        factors.append(("🌾 Rural residence", "low", "−"))

    nat_rate = COUNTRY_INCLUSION_RATE[country]
    factors.append((f"🌍 {country} national rate: {nat_rate:.0%}", "low",
                    "+" if nat_rate > 0.6 else "−"))

    return factors


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🌍 Financial Inclusion Predictor")
st.caption("Sub-Saharan Africa · World Bank Findex 2021 · XGBoost model")
st.markdown(
    "Predict whether an individual is likely to have access to a formal "
    "financial account (bank or mobile money) based on their demographic "
    "and socioeconomic profile."
)

# Context banner
with st.expander("📊 Why this matters"):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nigeria inclusion rate", "45%", help="Findex 2021")
    col2.metric("Ghana inclusion rate", "57%", help="Findex 2021")
    col3.metric("Kenya inclusion rate", "79%", help="Findex 2021")
    col4.metric("South Africa inclusion rate", "85%", help="Findex 2021")
    st.caption("Account ownership at a financial institution or mobile money provider. Source: World Bank Global Findex 2021.")

st.divider()
st.subheader("Profile inputs")

col1, col2 = st.columns(2)
with col1:
    country = st.selectbox("Country", COUNTRIES)
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    age = st.slider("Age", min_value=15, max_value=70, value=30)
    education = st.selectbox("Highest education level", EDUCATION_LEVELS, index=2)

with col2:
    income_quintile = st.selectbox("Household income quintile", INCOME_QUINTILES, index=2)
    employment = st.selectbox("Employment status", EMPLOYMENT_STATUS)
    urban = st.radio("Residential area", ["Urban", "Rural"], horizontal=True)
    mobile_owned = st.radio("Owns a mobile phone?", ["Yes", "No"], horizontal=True)
    internet_use = st.radio("Uses the internet?", ["Yes", "No"], horizontal=True)

st.divider()

if st.button("Predict financial inclusion likelihood", type="primary", use_container_width=True):
    female = gender == "Female"
    employed = employment in ["Employed (full-time)", "Employed (part-time)", "Self-employed"]
    is_urban = urban == "Urban"
    has_mobile = mobile_owned == "Yes"
    uses_internet = internet_use == "Yes"

    pred, prob = predict(
        country, female, age, education, income_quintile,
        employed, is_urban, has_mobile, uses_internet
    )

    # Result
    if pred == 1:
        st.success(f"### ✅ Likely financially included")
        st.markdown(
            f"This profile has a **{prob:.0%} probability** of having access to a "
            f"formal financial account."
        )
    else:
        st.error(f"### ❌ At risk of financial exclusion")
        st.markdown(
            f"This profile has only a **{prob:.0%} probability** of financial inclusion — "
            f"below the threshold for this model."
        )

    # Progress bar
    st.progress(float(prob), text=f"Inclusion probability: {prob:.0%}")

    st.divider()

    # Feature contributions
    st.markdown("**Key factors driving this prediction:**")
    factors = feature_contributions(
        country, female, age, education, income_quintile,
        employed, is_urban, has_mobile, uses_internet
    )

    for label, importance, direction in factors:
        if direction == "+":
            st.markdown(f"🟢 &nbsp; {label}")
        else:
            st.markdown(f"🔴 &nbsp; {label}")

    st.divider()

    # Policy insight
    st.markdown("**Policy insight**")
    if not has_mobile:
        st.info(
            "📱 **Mobile ownership is the single strongest predictor** of financial inclusion "
            "in Sub-Saharan Africa. Expanding mobile infrastructure and reducing handset "
            "costs could unlock inclusion for millions of excluded individuals with similar profiles."
        )
    elif INCOME_QUINTILES.index(income_quintile) <= 1:
        st.info(
            "💰 **Income level is the second strongest barrier** for this profile. "
            "Targeted programmes such as government-to-person transfers, savings groups, "
            "and subsidised account products are most effective for low-income populations."
        )
    elif not uses_internet:
        st.info(
            "🌐 **Internet access amplifies mobile money adoption.** USSD-based mobile "
            "money (like M-Pesa) can bridge this gap without requiring internet connectivity."
        )
    else:
        st.info(
            f"📊 The overall financial inclusion rate in **{country}** is "
            f"{COUNTRY_INCLUSION_RATE[country]:.0%}. This profile sits "
            f"{'above' if prob > COUNTRY_INCLUSION_RATE[country] else 'below'} the national average."
        )

    with st.expander("Model details"):
        st.markdown("""
**Model:** Gradient Boosting Classifier (scikit-learn)  
**Training data:** Synthetic data calibrated to World Bank Findex 2021 SSA distributions  
**Countries:** Nigeria, Ghana, Kenya, South Africa  
**Features:** Country, gender, age, education, income quintile, employment, urbanisation, mobile ownership, internet use  
**Top predictors:** Mobile ownership, income quintile, internet access, education  
**Note:** In production, replace the synthetic model with one trained directly on the Findex microdata from [data.worldbank.org](https://microdata.worldbank.org)
        """)

st.divider()
st.caption(
    "Built by [Favour Egberike](https://github.com/favouregberike) · "
    "[View full project →](https://github.com/favouregberike/modeling-financial-exclusion-africa)"
)
