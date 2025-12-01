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
# CSS tùy chỉnh để tăng kích thước phông chữ của tab
st.markdown("""
<style>

/* --- STYLE TAB ĐẸP – hiện đại --- */
button[role="tab"] > div {
    font-size: 18px !important;
    font-weight: 700 !important;
    padding: 10px 18px !important;
}

/* Container của toàn bộ tabs */
.stTabs [role="tablist"] {
    gap: 6px !important;           /* Khoảng cách giữa các tab */
    padding-bottom: 6px !important;
}

/* Tab chưa được chọn */
button[role="tab"] {
    border-radius: 8px !important;     /* Góc bo mềm */
    background-color: #f1f3f6 !important;
    border: 1px solid #dce0e5 !important;
    color: #303030 !important;
    transition: all 0.25s ease !important;
}

/* Hover tab */
button[role="tab"]:hover {
    background-color: #e3e7ed !important;
    border-color: #c7ccd3 !important;
}

/* Tab đang được chọn */
button[role="tab"][aria-selected="true"] {
    background-color: #cfe3ff !important;    /* Xanh nhạt */
    border: 1px solid #4a90e2 !important;    /* Border xanh highlight */
    color: #003366 !important;
}

/* Text trong tab đang chọn */
button[role="tab"][aria-selected="true"] > div {
    font-weight: 900 !important;
}

</style>
""", unsafe_allow_html=True)
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Kaplan–Meier",
    "📉 ROC Curve (RDW)",
    "📊 ROC Comparison (AIMS65 / SOFA / RDW)",
    "🔶 RCS–Cox (Restricted Cubic Splines)",
    "⭐ Model Comparison – ROC Curves",
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
    events = ['mortality_30d', 'mortality_90d']
    titles = ['30 days', '90 days']
    desired_order = ['Q1', 'Q2', 'Q3', 'Q4']

    # Ép lại thứ tự nhóm RDW (rất quan trọng để legend và plot đúng thứ tự)
    df['rdw_group'] = pd.Categorical(df['rdw_group'], categories=desired_order, ordered=True)

    # Chỉ giữ các nhóm thực sự có trong dữ liệu
    groups = [g for g in desired_order if g in df['rdw_group'].unique()]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
    axes = axes.flatten()

    for i in range(len(durations)):
        ax = axes[i]
        models = []

        # Vẽ từng nhóm theo đúng thứ tự groups
        for group in groups:
            df_g = df[df['rdw_group'] == group]
            if df_g.empty:
                continue

            kmf = KaplanMeierFitter()
            kmf.fit(
                durations=df_g[durations[i]],
                event_observed=df_g[events[i]],
                label=group
            )
            kmf.plot(ax=ax)
            models.append(kmf)

        # Log-rank test
        try:
            results = multivariate_logrank_test(
                event_durations=df[durations[i]],
                groups=df['rdw_group'],
                event_observed=df[events[i]]
            )
            pval = results.p_value
            p_text = "Log-rank p < 0.001" if pval < 0.001 else f"Log-rank p = {pval:.4f}"
            ax.text(0.05, 0.05, p_text, transform=ax.transAxes,
                    fontsize=14, verticalalignment='bottom')
        except Exception as e:
            ax.text(0.05, 0.05, "Log-rank: error", transform=ax.transAxes)

        # Sắp xếp legend theo đúng thứ tự groups
        handles, lbls = ax.get_legend_handles_labels()
        label2h = dict(zip(lbls, handles))
        ordered_handles = [label2h[g] for g in groups if g in label2h]
        ordered_labels = [g for g in groups if g in label2h]
        if ordered_handles:
            ax.legend(ordered_handles, ordered_labels)

        # Title, axis labels
        ax.set_title(titles[i])
        ax.set_ylabel('Survival probability')
        ax.set_xlabel('Time (days)')
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_ylim(0, 1)

        # X-axis limit cho 90 ngày
        if durations[i] == "duration_90d":
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

        # =====================================================
        #        P-VALUE: LOGISTIC REGRESSION (statsmodels)
        # =====================================================
        import statsmodels.api as sm
        X_sm = sm.add_constant(score)
        logit = sm.Logit(y, X_sm)
        res = logit.fit(disp=False)

        pval = res.pvalues[1]
        p_text = "P-value < 0.001" if pval < 0.001 else f"P = {pval:.3g}"

        # Hiển thị p-value ở góc trên trái
        ax.text(
            0.02, 0.98,
            p_text,
            transform=ax.transAxes,
            fontsize=12,
            fontweight='bold',
            va='top'
        )
        # =====================================================

        # ROC
        fpr, tpr, _ = roc_curve(y, score)
        auc = roc_auc_score(y, score)

        # Bootstrap AUC + ROC band
        auc_boot, (ci_low, ci_high) = bootstrap_auc_ci(y, score, n_boot=500, seed=42)
        fpr_grid, tpr_mean, tpr_low, tpr_high = bootstrap_roc_band(y, score, n_boot=500, seed=42)

        # Youden index
        thr, sens, spec = youden_threshold(y, score)

        # Draw ROC
        ax.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})", linewidth=2)
        ax.fill_between(fpr_grid, tpr_low, tpr_high, alpha=0.2)

        ax.plot([0,1],[0,1],'--', color='gray', alpha=0.6)

        # Youden point
        ax.scatter([1-spec],[sens], c='black')
        ax.annotate(
            f"Thr={thr:.2f}\nSens={sens:.2f}\nSpec={spec:.2f}",
            xy=(1-spec, sens),
            xytext=(20,-20),
            textcoords='offset points',
            bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.8)
        )

        # Labels & CI box
        ax.set_xlabel("1 - Specificity")
        ax.set_ylabel("Sensitivity")
        ax.set_title(title)
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.grid(True)

        ax.text(
            0.60, 0.12,
            f"AUC 95% CI:\n({ci_low:.3f}, {ci_high:.3f})",
            transform=ax.transAxes,
            bbox=dict(facecolor="white", alpha=0.8)
        )
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

        # ==========================================================
        # 👉 (A) Tính p-value chung từ model AIMS65 + RDW
        # ==========================================================
        predictors_all = ['aims65_score', 'rdw_max']

        mask_all = df[[ycol] + predictors_all].notna().all(axis=1)
        y_all = df.loc[mask_all, ycol].astype(int).values
        X_all = df.loc[mask_all, predictors_all].values

        try:
            X_sm = sm.add_constant(X_all)
            logit = sm.Logit(y_all, X_sm)
            res = logit.fit(disp=False)

            max_p = float(np.max(res.pvalues))

            if max_p < 0.05:
                p_text = "all P < 0.05"
            else:
                p_text = f"all P = {max_p:.3g}"
        except:
            p_text = "P-value: N/A"

        # Hiển thị p-value ở góc trên trái
        ax.text(
            0.02, 0.98, p_text,
            ha='left', va='top',
            transform=ax.transAxes,
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.6)
        )
        # ==========================================================

        # ==========================================================
        # 👉 (B) Vẽ ROC cho tất cả các model
        # ==========================================================
        for name, config in models.items():
            predictors = config["predictors"]

            if any([p not in df.columns for p in predictors]) or ycol not in df.columns:
                continue

            mask = df[[ycol] + predictors].notna().all(axis=1)
            y = df.loc[mask, ycol].astype(int).values
            X = df.loc[mask, predictors].values

            if len(y) < 20 or y.min() == y.max():
                continue

            if predictors in [["rdw_max"], ["aims65_score"]]:
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

            ax.plot(
                fpr, tpr,
                linestyle=config['linestyle'],
                color=config['color'],
                linewidth=2,
                label=f"{name} AUC={auc:.3f} (95% CI {ci_low:.3f}–{ci_high:.3f})"
            )

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
    def run_rcs_cox(df, duration_col, event_col, covariates):
        df_rcs = df.copy()
        df_rcs['gender'] = np.where(df_rcs['gender']=='M', 1, 0)
        df_rcs = df_rcs.dropna(subset=['rdw_max', duration_col, event_col])

        knots = df_rcs['rdw_max'].quantile([0.05,0.50,0.95]).tolist()

        spline = rcs(df_rcs['rdw_max'].values, knots)
        for k,v in spline.items():
            df_rcs[k] = v

        spline_vars = list(spline.keys())
        FINAL = [duration_col, event_col] + spline_vars + covariates

        df_fit = df_rcs[FINAL].dropna()

        cph = CoxPHFitter()
        cph.fit(df_fit, duration_col=duration_col, event_col=event_col)

        return cph, spline_vars, knots, df_rcs


    # ---- Plot HR ----
    def plot_rcs_subplot(ax, cph, df_rcs, spline_vars, knots, covariates, title):

        # ---------------------------
        # 1. RDW range: dùng 1–99% để tránh outlier
        # ---------------------------
        rdw_range = np.linspace(
            df_rcs['rdw_max'].quantile(0.01),
            df_rcs['rdw_max'].quantile(0.99),
            200
        )

        # ---------------------------
        # 2. Build prediction dataframe (ĐẦY ĐỦ BIẾN)
        # ---------------------------
        spline_pred = rcs(rdw_range, knots)
        pred_df = pd.DataFrame({'rdw_max': rdw_range})

        # thêm spline vào pred_df
        for k, v in spline_pred.items():
            pred_df[k] = v

        # thêm tất cả covariates
        for col in covariates:
            if df_rcs[col].nunique() > 2:
                pred_df[col] = df_rcs[col].median()
            else:
                pred_df[col] = df_rcs[col].mode()[0]

        # ---------------------------
        # 3. Linear predictor theo đúng thứ tự mô hình
        # ---------------------------
        X = pred_df[cph.params_.index]      # thứ tự KHÔNG được sai
        beta = cph.params_.values
        LP = X.dot(beta)

        # ---------------------------
        # 4. Normalize HR = 1 tại median RDW
        # ---------------------------
        rdw_median = df_rcs['rdw_max'].median()
        idx_ref = (abs(rdw_range - rdw_median)).argmin()
        LP0 = LP.iloc[idx_ref]

        log_hr = LP - LP0
        HR = np.exp(log_hr)

        # ---------------------------
        # 5. Đúng công thức SE = sqrt(X Σ Xᵀ)
        # ---------------------------
        var_beta = cph.variance_matrix_.values
        SE = np.sqrt(np.einsum("ij,jk,ik->i", X, var_beta, X))

        HR_low = np.exp(log_hr - 1.96 * SE)
        HR_up  = np.exp(log_hr + 1.96 * SE)

        # ---------------------------
        # 6. Plot
        # ---------------------------
        ax.plot(rdw_range, HR, color='brown', lw=2)
        ax.fill_between(rdw_range, HR_low, HR_up, color='brown', alpha=0.25)

        ax.axhline(1.0, linestyle='--', color='black')
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("RDW")
        ax.set_ylabel("Hazard Ratio")
        ax.grid(alpha=0.3)


    # ---- Tạo 1 figure với 4 outcome ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, (name, dur, evt) in enumerate(OUTCOME_LIST):
        try:
            cph, spline_vars, knots, df2 = run_rcs_cox(df, dur, evt, COVS_AVAILABLE)
            plot_rcs_subplot(axes[i], cph, df2, spline_vars, knots, COVS_AVAILABLE, name)
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
        X = pd.read_csv(os.path.join(BASE_DIR, "X_test.csv"))
        Y = pd.read_csv(os.path.join(BASE_DIR, "Y_test.csv"))
        return X, Y

    # ================================
    # LOAD MODELS (.joblib)
    # ================================
    @st.cache_resource
    def load_model():
        models_30d = {
            "Ada Boost": joblib.load(os.path.join(BASE_DIR, "AdaBoost_mortality_30d.joblib")),
            "Extra Trees": joblib.load(os.path.join(BASE_DIR, "ExtraTrees_mortality_30d.joblib")),
            "Gradient Boosting": joblib.load(os.path.join(BASE_DIR, "GradientBoosting_mortality_30d.joblib")),
            "Random Forest": joblib.load(os.path.join(BASE_DIR, "RandomForest_mortality_30d.joblib")),
        }

        models_90d = {
            "Ada Boost": joblib.load(os.path.join(BASE_DIR, "AdaBoost_mortality_90d.joblib")),
            "Extra Trees": joblib.load(os.path.join(BASE_DIR, "ExtraTrees_mortality_90d.joblib")),
            "Gradient Boosting": joblib.load(os.path.join(BASE_DIR, "GradientBoosting_mortality_90d.joblib")),
            "Random Forest": joblib.load(os.path.join(BASE_DIR, "RandomForest_mortality_90d.joblib")),
        }

        return models_30d, models_90d

    X_test, Y_test = load_data()
    models_30d, models_90d = load_model()

    # ================================
    # PLOT ROC
    # ================================
    mortalities = ["mortality_30d", "mortality_90d"]
    titles = ["30-Day Mortality", "90-Day Mortality"]

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    for i, target in enumerate(mortalities):

        y_test = Y_test[target]

        current_models = models_30d if target == "mortality_30d" else models_90d

        for name, model in current_models.items():

            # Không đưa AUC vào name nữa — để RocCurveDisplay tự hiển thị 1 lần
            RocCurveDisplay.from_estimator(
                estimator=model,
                X=X_test,
                y=y_test,
                ax=axes[i],
                name=name        # chỉ hiển thị tên model
            )

        axes[i].set_title(titles[i], fontsize=16)
        axes[i].grid(True)
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
        vars['temperature_max'] = st.number_input('Temperature', 33.0, 42.0, 37.0)
        vars['sofa'] = st.number_input('SOFA', 0, 24, 4)
        vars['sapsii'] = st.number_input('SAPSII', 0, 200, 35)
        vars['malignant_cancer'] = st.selectbox('Malignant cancer', [0, 1], index=0)

    with col2:
        vars['has_mv'] = st.selectbox('Mechanical Ventilation', [0, 1], index=0)
        vars['has_vaso'] = st.selectbox('Vasopressor', [0, 1], index=0)
        vars['inr_max'] = st.number_input('INR', 0.5, 10.0, 1.2)
        vars['platelets_max'] = st.number_input('Platelets', 1.0, 1500.0, 250.0)
        vars['chloride_max'] = st.number_input('Chloride', 70.0, 140.0, 104.0)
        vars['aims65_score'] = st.number_input('AIMS65 Score', 0, 5, 1)

    # ================================
    # PREDICT BUTTON
    # ================================
    arr = ['Survival', 'Died']

    if st.button("🔮 Predict"):

        df_pred = pd.DataFrame([vars])

        pred = gb_model.predict(df_pred)[0]
        prob = gb_model.predict_proba(df_pred)[0][1]

        st.subheader(f"🧾 Prediction: **{arr[pred]}**")
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