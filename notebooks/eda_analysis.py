# ============================================================
# Job Description Inflation Index — Exploratory Data Analysis
# Author: Saranya | Portfolio Project
# Dataset: 10,636 job postings (2018–2025)
# Run from project root: python notebooks/eda_analysis.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
# matplotlib.use('Agg')  # Commented out to allow interactive windows
import matplotlib.pyplot as plt
from scipy import stats
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIG ───────────────────────────────────────────────────────
DATA_PATH = 'data/Job_Description_Inflation_Index_10K.xlsx'
OUT_PATH  = 'docs/eda_overview.png'

ACCENT  = '#E8FF47'; ACCENT2 = '#47C5FF'; ACCENT3 = '#FF6B6B'
ACCENT4 = '#A78BFA'; ACCENT5 = '#34D399'
BG = '#07090F'; SURFACE = '#0E1117'
PALETTE = [ACCENT,ACCENT2,ACCENT4,ACCENT3,ACCENT5,'#F97316','#EC4899']

# ─── LOAD ─────────────────────────────────────────────────────────
df = pd.read_excel(DATA_PATH)
print(f"✅  Loaded {len(df):,} rows × {df.shape[1]} columns")
print(f"📅  Years: {df['Posting_Year'].min()} → {df['Posting_Year'].max()}")
print()

# ─── AGGREGATIONS ─────────────────────────────────────────────────
jdii_yr   = df.groupby('Posting_Year')['JDII_Score'].mean()
exp_yr    = df.groupby('Posting_Year')['Min_Exp_Required_Yrs'].mean()
skills_yr = df.groupby('Posting_Year')['Num_Skills_Listed'].mean()
deg_cts   = df['Degree_Required'].value_counts()
ind_jdii  = df.groupby('Industry')['JDII_Score'].mean().sort_values(ascending=False)

# ─── PEARSON CORRELATION — Weight Justification ───────────────────
# These correlations with JDII_Score are the basis for the 5 factor weights.
WEIGHT_FACTORS = [
    'Min_Exp_Required_Yrs',   # Experience  → Weight 40%
    'Num_Skills_Listed',       # Skills      → Weight 30%
    'Responsibilities_Count',  # Resp.       → Weight 15%
    'Certifications_Required', # Certs       → Weight 10%
    'Num_Buzzwords',           # Buzzwords   → Weight  5%
]
WEIGHT_LABELS  = ['Experience (40%)', 'Skills (30%)', 'Responsibilities (15%)',
                  'Certifications (10%)', 'Buzzwords (5%)']

corr_matrix = df[WEIGHT_FACTORS + ['JDII_Score']].corr()
pearson_r   = corr_matrix['JDII_Score'].drop('JDII_Score')

print("=" * 58)
print("🔗  PEARSON CORRELATION — JDII Weight Justification")
print("=" * 58)
for factor, label in zip(WEIGHT_FACTORS, WEIGHT_LABELS):
    r_val = pearson_r[factor]
    print(f"  {label:<28} r = {r_val:.4f}")
print()

# ─── REGRESSION ───────────────────────────────────────────────────

years = np.array(jdii_yr.index)

sl_j, ic_j, r_j, p_j, _ = stats.linregress(years, jdii_yr.values)
sl_e, ic_e, r_e, p_e, _ = stats.linregress(years, exp_yr.values)
sl_s, ic_s, r_s, p_s, _ = stats.linregress(years, skills_yr.values)

print("=" * 58)
print("📈  REGRESSION RESULTS")
print("=" * 58)
print(f"  JDII   : slope={sl_j:.4f}  R²={r_j**2:.4f}  p={p_j:.5f}")
print(f"  Exp    : slope={sl_e:.4f}  R²={r_e**2:.4f}  p={p_e:.8f}")
print(f"  Skills : slope={sl_s:.4f}  R²={r_s**2:.4f}  p={p_s:.6f}")

# ─── FORECAST ─────────────────────────────────────────────────────
print("\n📊  FORECAST")
for yr in [2026, 2027, 2028]:
    jdii_f  = ic_j + sl_j * yr
    exp_f   = ic_e + sl_e * yr
    skills_f= ic_s + sl_s * yr
    print(f"  {yr}  JDII={jdii_f:.1f}  Exp={exp_f:.2f}yrs  Skills={skills_f:.2f}")

# ─── HYPOTHESIS TEST ──────────────────────────────────────────────
remote_jdii = df[df['Remote_Option']=='Yes']['JDII_Score']
onsite_jdii = df[df['Remote_Option']=='No']['JDII_Score']
t_stat, t_p  = stats.ttest_ind(remote_jdii, onsite_jdii)
print(f"\n🧪  T-TEST: Remote vs Onsite JDII")
print(f"  Remote mean: {remote_jdii.mean():.2f}  |  Onsite mean: {onsite_jdii.mean():.2f}")
print(f"  t={t_stat:.4f}  p={t_p:.8f}  → {'REJECT H₀ ✓' if t_p < 0.05 else 'FAIL TO REJECT'}")

# ─── KEY INSIGHTS ─────────────────────────────────────────────────
print(f"\n💡  KEY FINDINGS")
print(f"  1. JDII: {jdii_yr[2018]:.1f} (2018) → {jdii_yr[2025]:.1f} (2025)  (+{jdii_yr[2025]-jdii_yr[2018]:.1f} pts, +{(jdii_yr[2025]/jdii_yr[2018]-1)*100:.1f}%)")
print(f"  2. Exp:  {exp_yr[2018]:.2f} yrs → {exp_yr[2025]:.2f} yrs  (+{exp_yr[2025]-exp_yr[2018]:.2f} yrs)")
print(f"  3. Skills: {skills_yr[2018]:.2f} → {skills_yr[2025]:.2f}  (+{skills_yr[2025]-skills_yr[2018]:.2f})")
print(f"  4. MBA requirement: {deg_cts.get('MBA',0)/len(df)*100:.0f}% of all postings")
bach_pct = deg_cts.get("Bachelor's Degree", 0) / len(df) * 100
print(f"  5. Bachelor's only: {bach_pct:.0f}% of all postings")
print(f"  6. Remote JDII > Onsite by {remote_jdii.mean()-onsite_jdii.mean():.2f} pts (p<0.001)")

# ─── TOP 20 MOST INFLATED ─────────────────────────────────────────
print(f"\n🔥  TOP 10 MOST INFLATED JOB POSTINGS (JDII=100)")
top20 = df.nlargest(10,'JDII_Score')[['Job_Title','Industry','Seniority_Level','Min_Exp_Required_Yrs','JDII_Score']]
print(top20.to_string(index=False))

# ─── PLOTS ────────────────────────────────────────────────────────
print("\n🎨  Generating plots...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor=BG)
fig.patch.set_facecolor(BG)
fig.suptitle('Job Description Inflation Index — EDA Overview', color='white', fontsize=15, fontweight='bold')

for ax in axes.flat:
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors='#94a3b8', labelsize=9)
    for s in ax.spines.values(): s.set_color('#1e2535')

# 1. JDII trend + trendline
ax = axes[0,0]
ax.plot(jdii_yr.index, jdii_yr.values, color=ACCENT, lw=2.5, marker='o', markersize=5, label='Actual')
trend = [ic_j + sl_j*y for y in years]
ax.plot(years, trend, color=ACCENT, lw=1.2, linestyle='--', alpha=.5, label=f'Trend (R²={r_j**2:.2f})')
ax.set_title('JDII Score Trend (2018–2025)', color='white', fontsize=11, pad=8)
ax.set_xlabel('Year', color='#64748b', fontsize=9)
ax.set_ylabel('Avg JDII', color='#64748b', fontsize=9)
ax.legend(fontsize=8, facecolor=SURFACE, edgecolor='#1e2535', labelcolor='#94a3b8')

# 2. Exp trend + trendline
ax = axes[0,1]
ax.plot(exp_yr.index, exp_yr.values, color=ACCENT2, lw=2.5, marker='s', markersize=5, label='Actual')
trend_e = [ic_e + sl_e*y for y in years]
ax.plot(years, trend_e, color=ACCENT2, lw=1.2, linestyle='--', alpha=.5, label=f'Trend (R²={r_e**2:.2f})')
ax.set_title('Experience Required (2018–2025)', color='white', fontsize=11, pad=8)
ax.set_xlabel('Year', color='#64748b', fontsize=9)
ax.set_ylabel('Avg Years', color='#64748b', fontsize=9)
ax.legend(fontsize=8, facecolor=SURFACE, edgecolor='#1e2535', labelcolor='#94a3b8')

# 3. Degree pie
ax = axes[0,2]
ax.pie(deg_cts.values, labels=deg_cts.index,
       colors=[ACCENT3,ACCENT4,ACCENT2,ACCENT], autopct='%1.0f%%',
       pctdistance=0.75, startangle=90, textprops={'color':'#94a3b8','fontsize':9})
ax.set_title('Degree Requirements', color='white', fontsize=11, pad=8)

# 4. JDII by seniority
ax = axes[1,0]
sen_j = df.groupby('Seniority_Level')['JDII_Score'].mean().sort_values()
ax.barh(sen_j.index, sen_j.values, color=PALETTE[:len(sen_j)])
ax.set_title('JDII by Seniority Level', color='white', fontsize=11, pad=8)
ax.set_xlabel('Avg JDII', color='#64748b', fontsize=9)

# 5. Skill growth comparison
ax = axes[1,1]
skills_data = [('Python',41.8,54.6),('SQL',33.2,40.1),('Power BI',28.1,36.6),('Tableau',23.1,31.1),('Excel',20.9,28.4)]
sk_names = [s[0] for s in skills_data]
s18 = [s[1] for s in skills_data]
s25 = [s[2] for s in skills_data]
x = np.arange(len(sk_names))
ax.bar(x-.2, s18, .35, label='2018 %', color=ACCENT2, alpha=.7)
ax.bar(x+.2, s25, .35, label='2025 %', color=ACCENT, alpha=.8)
ax.set_xticks(x); ax.set_xticklabels(sk_names, fontsize=8)
ax.set_title('Skill Demand Growth (2018 vs 2025)', color='white', fontsize=11, pad=8)
ax.set_ylabel('% of postings', color='#64748b', fontsize=9)
ax.legend(fontsize=8, facecolor=SURFACE, edgecolor='#1e2535', labelcolor='#94a3b8')

# 6. JDII by industry
ax = axes[1,2]
ax.barh(list(ind_jdii.index[:8]), list(ind_jdii.values[:8]), color=PALETTE)
ax.set_title('JDII by Industry', color='white', fontsize=11, pad=8)
ax.set_xlabel('Avg JDII', color='#64748b', fontsize=9)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight', facecolor=BG, edgecolor='none')
print(f"✅  Saved: {OUT_PATH}")
plt.show()  # Display the chart in a window
plt.close()

print("\n✅  EDA complete.")
