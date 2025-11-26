import streamlit as st
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
import shap
# shap.initjs()

st.set_page_config(layout="wide")
st.title("Red Blood Cell Distribution Width as a Risk Factor for 30/90‑Day Mortality in Patients with Gastrointestinal Bleeding")

# ======================================================
# LOAD CSV
# ======================================================
FILE_PATH = "clean_with_duration.csv"
try:
    df = pd.read_csv(FILE_PATH)
    st.success(f"Loaded file: {FILE_PATH}")
except Exception as e:
    st.error(f"Cannot load {FILE_PATH}: {e}")
    st.stop()

# ensure expected columns exist (warn if missing)
expected = ['rdw_max','intime','dod']
for col in expected:
    if col not in df.columns:
        st.warning(f"Warning: column '{col}' not found in CSV (may be fine if not used).")

# thiết lập nhóm RDW (nếu có rdw_group, giữ; nếu không, create from rdw_max using same bins)
desired_order = ['Q1', 'Q2', 'Q3', 'Q4']
if 'rdw_group' not in df.columns and 'rdw_max' in df.columns:
    bins = [12.4, 14.5, 16.0, 18.0, 25.1]
    labels = desired_order
    df['rdw_group'] = pd.cut(df['rdw_max'], bins=bins, labels=labels, include_lowest=True)
if 'rdw_group' in df.columns:
    df['rdw_group'] = pd.Categorical(df['rdw_group'], categories=desired_order, ordered=True)

# ======================================================
# Create main tabs (4 tabs)
# ======================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Kaplan–Meier",
    "📉 ROC Curve (RDW)",
    "📊 ROC Comparison (AIMS65 / SOFA / RDW)",
    "🔶 RCS–Cox (Restricted Cubic Splines)",
    "📊 Model Comparison – ROC Curves",
    "🚀 Gradient Boosting – Feature Selection + Training + Evaluation"
])

# --------------------
# Helper functions for ROC
# --------------------
def bootstrap_auc_ci(y, y_score, n_boot=1000, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    y = np.asarray(y)
    y_score = np.asarray(y_score)
    n = len(y)
    aucs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        y_s = y[idx]
        score_s = y_score[idx]
        if y_s.min() == y_s.max():
            continue
        try:
            aucs.append(roc_auc_score(y_s, score_s))
        except Exception:
            continue
    aucs = np.array(aucs)
    if aucs.size == 0:
        return np.nan, (np.nan, np.nan)
    return aucs.mean(), (np.percentile(aucs, 2.5), np.percentile(aucs, 97.5))

def bootstrap_roc_band(y, y_score, n_boot=1000, fpr_grid=None, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    y = np.asarray(y)
    y_score = np.asarray(y_score)
    n = len(y)
    if fpr_grid is None:
        fpr_grid = np.linspace(0,1,200)
    tprs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        y_s = y[idx]
        score_s = y_score[idx]
        if y_s.min() == y_s.max():
            continue
        fpr_s, tpr_s, _ = roc_curve(y_s, score_s)
        tpr_interp = np.interp(fpr_grid, fpr_s, tpr_s)
        tprs.append(tpr_interp)
    tprs = np.array(tprs)
    if tprs.size == 0:
        return fpr_grid, np.zeros_like(fpr_grid), np.zeros_like(fpr_grid), np.zeros_like(fpr_grid)
    return fpr_grid, tprs.mean(axis=0), np.percentile(tprs, 2.5, axis=0), np.percentile(tprs, 97.5, axis=0)

def youden_threshold(y, y_score):
    fpr, tpr, thr = roc_curve(y, y_score)
    J = tpr - fpr
    idx = np.argmax(J)
    sens = tpr[idx]
    spec = 1 - fpr[idx]
    return thr[idx], sens, spec

# ======================================================
# TAB 1: Kaplan–Meier
# ======================================================
with tab1:
    st.subheader("Kaplan–Meier Survival Curves")
    durations = ['duration_30d', 'duration_90d']
    events = ['event_30d', 'event_90d']
    titles = ['30 days', '90 days']

    fig, axes = plt.subplots(1, 2, figsize=(14,6))
    axes = axes.flatten()

    for i in range(len(durations)):
        ax = axes[i]
        models = []
        for group in desired_order:
            if 'rdw_group' not in df.columns:
                continue
            mask = df['rdw_group'] == group
            if mask.sum() == 0:
                continue
            kmf = KaplanMeierFitter()
            kmf.fit(df.loc[mask, durations[i]], df.loc[mask, events[i]], label=group)
            kmf.plot(ax=ax)
            models.append(kmf)

        # log-rank (safe if columns exist)
        if durations[i] in df.columns and events[i] in df.columns and 'rdw_group' in df.columns:
            try:
                results = multivariate_logrank_test(df[durations[i]], df['rdw_group'], df[events[i]])
                p = results.p_value
                p_text = "p < 0.001" if p < 0.001 else f"p = {p:.4f}"
                ax.text(0.05, 0.05, p_text, transform=ax.transAxes)
            except Exception:
                pass

        # if models:
        #     add_at_risk_counts(*models, ax=ax)

        ax.set_title(titles[i])
        ax.set_ylim(0,1)
        if i == 1:
            ax.set_xlim(0, 90)

    st.pyplot(fig)

# ======================================================
# TAB 2: ROC Curve (RDW)
# ======================================================
with tab2:
    st.subheader("ROC Curve with Bootstrap 95% CI (RDW continuous)")

    outcomes = [
        ('mortality_30d', '30-day mortality'),
        ('mortality_90d', '90-day mortality')
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14,6))
    axes = axes.flatten()

    for ax, (ycol, title) in zip(axes, outcomes):
        if ycol not in df.columns or 'rdw_max' not in df.columns:
            ax.axis('off')
            ax.text(0.5,0.5,f"Missing '{ycol}' or 'rdw_max'", ha='center')
            continue

        mask = df[[ycol, 'rdw_max']].notna().all(axis=1)
        y = df.loc[mask, ycol].astype(int).values
        score = df.loc[mask, 'rdw_max'].values

        if len(y) == 0 or y.min() == y.max():
            ax.text(0.5,0.5,"Not enough variation", ha='center', va='center')
            ax.axis('off')
            continue

        fpr, tpr, _ = roc_curve(y, score)
        auc = roc_auc_score(y, score)
        auc_boot, (ci_low, ci_high) = bootstrap_auc_ci(y, score, n_boot=500, seed=42)
        fpr_grid, tpr_mean, tpr_low, tpr_high = bootstrap_roc_band(y, score, n_boot=500, seed=42)
        thr, sens, spec = youden_threshold(y, score)

        ax.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})", linewidth=2)
        ax.fill_between(fpr_grid, tpr_low, tpr_high, alpha=0.2)
        ax.plot([0,1],[0,1],'--', color='gray', alpha=0.6)
        ax.scatter([1-spec],[sens], c='black')
        ax.annotate(f"Thr={thr:.2f}\nSens={sens:.2f}\nSpec={spec:.2f}",
                    xy=(1-spec, sens), xytext=(20,-20), textcoords='offset points',
                    bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.8))

        ax.set_xlabel("1 - Specificity")
        ax.set_ylabel("Sensitivity")
        ax.set_title(title)
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.grid(True)
        ax.text(0.6,0.12, f"AUC 95% CI:\n({ci_low:.3f}, {ci_high:.3f})", transform=ax.transAxes,
                bbox=dict(facecolor="white", alpha=0.8))

    st.pyplot(fig)

# ======================================================
# TAB 3: ROC Comparison
# ======================================================
with tab3:
    st.subheader("ROC Comparison: AIMS65 / SOFA / RDW / Combined Models")

    N_BOOT = 500
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)

    outcomes = [
        ('event_30d', '30-day ICU mortality'),
        ('event_90d', '90-day ICU mortality')
    ]

    models = {
        'AIMS65': {'predictors': ['aims65_score'], 'linestyle': '-', 'color': 'blue'},
        'RDW': {'predictors': ['rdw_max'], 'linestyle': '--', 'color': 'orange'},
        'AIMS65+RDW': {'predictors': ['aims65_score', 'rdw_max'], 'linestyle': '-.', 'color': 'green'},
        'SOFA': {'predictors': ['sofa'], 'linestyle': ':', 'color': 'purple'},
        'SOFA+RDW': {'predictors': ['sofa', 'rdw_max'], 'linestyle': '-.', 'color': 'red'},
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes = axes.flatten()

    for ax_idx, (ycol, title) in enumerate(outcomes):
        ax = axes[ax_idx]

        for name, config in models.items():
            predictors = config["predictors"]
            # check columns
            if any([p not in df.columns for p in predictors]) or ycol not in df.columns:
                continue

            mask = df[[ycol] + predictors].notna().all(axis=1)
            y = df.loc[mask, ycol].astype(int).values
            X = df.loc[mask, predictors].values

            if len(y) < 20 or y.min() == y.max():
                continue

            if predictors == ["rdw_max"] or predictors == ["aims65_score"]:
                scores = X.flatten()
            else:
                log_reg = LogisticRegression(solver='liblinear', random_state=RANDOM_SEED)
                log_reg.fit(X, y)
                scores = log_reg.predict_proba(X)[:, 1]

            fpr, tpr, _ = roc_curve(y, scores)
            auc = roc_auc_score(y, scores)
            _, (ci_low, ci_high) = bootstrap_auc_ci(y, scores, n_boot=N_BOOT, seed=RANDOM_SEED)

            if name == 'AIMS65+RDW':
                fpr_grid, tpr_mean, tpr_low, tpr_high = bootstrap_roc_band(y, scores, n_boot=300, seed=RANDOM_SEED)
                ax.fill_between(fpr_grid, tpr_low, tpr_high, alpha=0.08, color=config['color'])

            ax.plot(fpr, tpr, linestyle=config['linestyle'], color=config['color'], linewidth=2,
                    label=f"{name} AUC={auc:.3f} (95% CI {ci_low:.3f}–{ci_high:.3f})")

        ax.plot([0, 1], [0, 1], '--', color='gray')
        ax.set_title(title)
        ax.set_xlabel("1 - Specificity")
        ax.set_ylabel("Sensitivity")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle=':', linewidth=0.5)
        ax.legend(loc="lower right", fontsize='small')

    st.pyplot(fig)

# ======================================================
# TAB 4: RCS–Cox (robust implementation tuned to your pipeline)
# ======================================================
COVARIATES_RCS = [
    'age', 'gender', 'bmi', 'sofa', 'sapsii', 
    'inr_max', 'pt_max', 'ptt_max', 
    'hemoglobin_min', 'platelets_min', 
    'has_mv', 'has_vaso', 'has_crrt', 
    'myocardial_infarct', 'congestive_heart_failure',
    'cerebrovascular_disease', 'chronic_pulmonary_disease',
    'renal_disease', 'malignant_cancer', 
    'diabetes', 'liver_disease'
]
OUTCOME_LIST = [
    ("30-day mortality", "duration_30d", "event_30d"),
    ("90-day mortality", "duration_90d", "event_90d"),
    ("ICU mortality", "los_icu", "icu_mortality"),
    ("Hospital mortality", "los_hosp", "hosp_mortality")
]

def rcs(x, knots):
    knots = np.sort(knots)
    k = len(knots)
    if k < 3:
        raise ValueError("RCS requires at least 3 knots.")

    def pos(y):
        return np.where(y > 0, y, 0)

    k_K = knots[-1]
    k_K_1 = knots[-2]
    
    bases = {}
    for j in range(k - 2):
        k_j = knots[j]
        term1 = pos(x - k_j)**3
        term2 = pos(x - k_K_1)**3 * ((k_K - k_j) / (k_K - k_K_1))
        term3 = pos(x - k_K)**3 * ((k_K_1 - k_j) / (k_K - k_K_1))
        
        B_j = (term1 - term2 + term3)
        bases[f"rcs_basis_{j+1}"] = B_j

    return bases

with tab4:

    st.subheader("Restricted Cubic Spline Cox models (RCS–Cox)")

    # Covariates có trong df
    COVS_AVAILABLE = [c for c in COVARIATES_RCS if c in df.columns]

    # ---- Chạy Cox ----
    def run_rcs(df, dur, evt, covs):
        df2 = df.copy()

        if 'gender' in df2.columns:
            df2['gender'] = np.where(df2['gender'].astype(str).str.startswith("M"),1,0)

        df2 = df2.dropna(subset=['rdw_max', dur, evt])

        knots = df2['rdw_max'].quantile([0.05,0.50,0.95]).tolist()

        spline = rcs(df2['rdw_max'].values, knots)
        for k,v in spline.items():
            df2[k] = v
        spline_vars = list(spline.keys())

        covs = [c for c in covs if c in df2.columns]
        FINAL = [dur, evt] + spline_vars + covs
        df_fit = df2[FINAL].dropna()

        cph = CoxPHFitter()
        cph.fit(df_fit, duration_col=dur, event_col=evt)
        return cph, spline_vars, knots, df2

    # ---- Plot HR ----
    def plot_hr(ax, cph, df2, spline_vars, knots, covs, title):
        x_min, x_max = df2['rdw_max'].min(), df2['rdw_max'].max()
        xs = np.linspace(x_min, x_max, 200)

        sp = rcs(xs, knots)
        pred = pd.DataFrame({"rdw_max": xs})
        for k,v in sp.items():
            pred[k] = v

        for c in covs:
            if df2[c].nunique() > 2:
                pred[c] = df2[c].median()
            else:
                pred[c] = df2[c].mode().iloc[0]

        params = list(cph.params_.index)
        for p in params:
            if p not in pred.columns:
                pred[p] = 0.0

        X = pred[params]
        beta = cph.params_.values
        log_hr = X.dot(beta)

        mid = df2['rdw_max'].median()
        idx0 = np.argmin(abs(xs - mid))
        log_hr = log_hr - log_hr[idx0]

        HR = np.exp(log_hr)

        var = cph.variance_matrix_.reindex(index=params, columns=params).fillna(0).values
        Xv = X.values
        SE = np.sqrt(np.sum((Xv @ var) * Xv, axis=1))

        HR_low = np.exp(log_hr - 1.96*SE)
        HR_up  = np.exp(log_hr + 1.96*SE)

        ax.plot(xs, HR, color='brown', lw=2)
        ax.fill_between(xs, HR_low, HR_up, alpha=0.25, color='brown')
        ax.axhline(1.0, ls='--', color='gray')
        ax.set_xlabel("RDW")
        ax.set_ylabel("Hazard Ratio")
        ax.set_title(title)
        ax.grid(alpha=0.3)

    # ---- Tạo 1 figure với 4 outcome ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, (name, dur, evt) in enumerate(OUTCOME_LIST):
        try:
            cph, spline_vars, knots, df2 = run_rcs(df, dur, evt, COVS_AVAILABLE)
            plot_hr(axes[i], cph, df2, spline_vars, knots, COVS_AVAILABLE, name)
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Model failed\n{e}", ha='center', va='center')
            axes[i].set_title(name)

    plt.tight_layout()
    st.pyplot(fig)
# ======================================================
# TAB 5: Model Comparison – ROC Curves (30-day & 90-day Mortality)
# ======================================================
with tab5:

    st.header("📊 Model Comparison – ROC Curves (30-day & 90-day Mortality)")

    BASE_DIR = os.path.dirname(__file__)

    # ================================
    # LOAD DATA
    # ================================
    @st.cache_resource
    def load_data():
        # Đảm bảo bạn đã nhập 'read_csv'
        try:
             X = pd.read_csv(os.path.join(BASE_DIR, "X_test.csv"))
             Y = pd.read_csv(os.path.join(BASE_DIR, "Y_test.csv"))
        except NameError:
             import pandas as pd
             X = pd.read_csv(os.path.join(BASE_DIR, "X_test.csv"))
             Y = pd.read_csv(os.path.join(BASE_DIR, "Y_test.csv"))
        return X, Y

    # ================================
    # LOAD MODELS (.joblib) - ĐÃ SỬA ĐỔI
    # ================================
    @st.cache_resource
    def load_model():
        # Tải bộ mô hình 30 ngày
        models_30d = {
            "Ada Boost": joblib.load(os.path.join(BASE_DIR, "AdaBoost_mortality_30d.joblib")),
            "Extra Trees": joblib.load(os.path.join(BASE_DIR, "ExtraTrees_mortality_30d.joblib")),
            "Gradient Boosting": joblib.load(os.path.join(BASE_DIR, "GradientBoosting_mortality_30d.joblib")),
            "Random Forest": joblib.load(os.path.join(BASE_DIR, "RandomForest_mortality_30d.joblib")),
        }
        # Tải bộ mô hình 90 ngày
        models_90d = {
            "Ada Boost": joblib.load(os.path.join(BASE_DIR, "AdaBoost_mortality_90d.joblib")),
            "Extra Trees": joblib.load(os.path.join(BASE_DIR, "ExtraTrees_mortality_90d.joblib")),
            "Gradient Boosting": joblib.load(os.path.join(BASE_DIR, "GradientBoosting_mortality_90d.joblib")),
            "Random Forest": joblib.load(os.path.join(BASE_DIR, "RandomForest_mortality_90d.joblib")),
        }
        # Trả về cả hai bộ mô hình
        return models_30d, models_90d

    X_test, Y_test = load_data()
    # Nhận hai bộ mô hình
    models_30d, models_90d = load_model()

    # ================================
    # PLOT ROC – only 30d & 90d
    # ================================
    mortalities = ["mortality_30d", "mortality_90d"]
    titles = ["30-Day Mortality", "90-Day Mortality"]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8), )

    for i in range(len(mortalities)):
        target = mortalities[i]
        y_test = Y_test[target]
        
        # CHỌN BỘ MÔ HÌNH PHÙ HỢP
        if target == "mortality_30d":
            current_models = models_30d
        else:
            current_models = models_90d # mortality_90d

        for name, model in current_models.items():
            RocCurveDisplay.from_estimator(
                estimator=model,
                X=X_test,
                y=y_test,
                ax=axes[i],
                name=f"{name} (AUC = {model.score(X_test, y_test):.2f})"
            )

        axes[i].set_title(titles[i], fontsize=16)
        axes[i].grid(True)
        # Thêm nhãn trục để biểu đồ dễ đọc hơn
        axes[i].set_xlabel('False Positive Rate (Positive label: 1)')
        axes[i].set_ylabel('True Positive Rate (Positive label: 1)')

    st.pyplot(fig)

# ======================================================
# TAB 6: Gradient Boosting – Mortality Prediction
# ======================================================
with tab6:

    st.header("🧠 Gradient Boosting – Mortality Prediction")

    BASE_DIR = os.path.dirname(__file__)

    # ================================
    # LOAD MODEL + X_train
    # ================================
    @st.cache_resource
    def load_gb_model():
        model_path = os.path.join(BASE_DIR, "gb_model.joblib")
        xtrain_path = os.path.join(BASE_DIR, "gb_X_train.joblib")

        if not os.path.exists(model_path):
            st.error("❌ Missing file: gb_model.joblib")
            st.stop()

        if not os.path.exists(xtrain_path):
            st.error("❌ Missing file: gb_X_train.joblib")
            st.stop()

        model = joblib.load(model_path)
        X_train = joblib.load(xtrain_path)
        return model, X_train

    gb_model, X_train_tab6 = load_gb_model()

    # SHAP Explainer
    explainer = shap.Explainer(gb_model, X_train_tab6)

    # ================================
    # USER INPUT
    # ================================
    st.subheader("📋 Input patient values")

    col1, col2 = st.columns(2)

    vars = {}
    with col1:
        vars['age'] = st.number_input('Age', 16, 100, 60)
        vars['bmi'] = st.number_input('BMI', 10.0, 60.0, 22.5)
        vars['temperature_max'] = st.number_input('Max Temperature', 33.0, 42.0, 37.0)
        vars['sofa'] = st.number_input('SOFA', 0, 24, 4)
        vars['sapsii'] = st.number_input('SAPSII', 0, 200, 35)
        vars['malignant_cancer'] = st.selectbox('Malignant cancer', [0, 1], index=0)

    with col2:
        vars['has_mv'] = st.selectbox('Mechanical Ventilation', [0, 1], index=0)
        vars['has_vaso'] = st.selectbox('Vasopressor', [0, 1], index=0)
        vars['inr_max'] = st.number_input('INR max', 0.5, 10.0, 1.2)
        vars['platelets_max'] = st.number_input('Platelets max', 1.0, 1500.0, 250.0)
        vars['chloride_max'] = st.number_input('Chloride max', 70.0, 140.0, 104.0)
        vars['aims65_score'] = st.number_input('AIMS65 Score', 0, 5, 1)

    # ================================
    # PREDICT BUTTON
    # ================================
    arr = ['Survival', 'Died']

    if st.button("🔮 Predict"):

        df_pred = pd.DataFrame([vars])

        pred = gb_model.predict(df_pred)[0]
        prob = gb_model.predict_proba(df_pred)[0][1]

        st.subheader(f"### 🧾 Prediction: **{arr[pred]}**")
        st.info(f"**Probability of Death:** {prob:.3f}")

        # ================================
        # SHAP VALUES
        # ================================
        shap_values = explainer(df_pred)

        # ---------- WATERFALL PLOT ----------
        st.subheader("🌊 SHAP Waterfall Plot")

        fig_water, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig_water)







