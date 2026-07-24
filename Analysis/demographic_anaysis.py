"""
Demographic & Cyber Awareness Survey Analysis
Local version (originally built for Kaggle).

Usage:
    python demograpghic_amnaysis.py --csv path/to/Data.csv --out charts

Requires: pandas, matplotlib, numpy  (see requirements.txt)
"""

import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 12

BLUE = '#2563EB'; PINK = '#EC4899'; GREEN = '#10B981'
ORANGE = '#F59E0B'; RED = '#EF4444'; PURPLE = '#8B5CF6'
COLORS = [BLUE, PINK, GREEN, ORANGE, PURPLE, RED]
BLUES4 = ['#DBEAFE', '#60A5FA', '#2563EB', '#1E3A5F']


def normalize_likert(series):
    mapping = {}
    for v in series.dropna().unique():
        vl = v.strip().lower()
        if 'strongly' in vl and 'agree' in vl and 'dis' not in vl:
            mapping[v] = 'Strongly Agree'
        elif ('strongly' in vl and 'dis' in vl) or 'stongly' in vl:
            mapping[v] = 'Strongly Disagree'
        elif 'disagree' in vl:
            mapping[v] = 'Disagree'
        elif 'neutral' in vl:
            mapping[v] = 'Neutral'
        elif 'agree' in vl:
            mapping[v] = 'Agree'
    out = series.map(mapping)
    out = out.fillna('Neutral')
    return out


def positive_pct(series):
    clean = normalize_likert(series)
    pos = clean.isin(['Agree', 'Strongly Agree']).sum()
    return round(pos / len(clean) * 100, 1)


def likert_bar(ax, df, col):
    ORDER = ['Strongly Disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree']
    COLORS_L = [RED, '#FCA5A5', '#D1D5DB', '#93C5FD', BLUE]
    clean = normalize_likert(df[col])
    mapped = clean.value_counts()
    vals = [mapped.get(o, 0) for o in ORDER]
    total = sum(vals)
    pcts = [v / total * 100 for v in vals]
    bars = ax.barh(ORDER, pcts, color=COLORS_L, edgecolor='white')
    for bar, pct, val in zip(bars, pcts, vals):
        if pct > 2:
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                     f'{pct:.1f}% ({val})', va='center', fontsize=11)
    ax.set_xlim(0, 82)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Percentage (%)', fontsize=12)


def save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  saved {path}")
    plt.close(fig)


def require_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"[error] Missing expected column(s): {missing}")
        print("Available columns:")
        for c in df.columns:
            print(f"  - {c}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Generate demographic & cyber-awareness charts from survey CSV.")
    parser.add_argument("--csv", default="Data.csv", help="Path to the survey CSV file (default: Data.csv)")
    parser.add_argument("--out", default="charts", help="Output directory for charts (default: charts)")
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"[error] CSV not found: {args.csv}")
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.csv)
    OUT = args.out

    print(f"Loaded {len(df)} rows from {args.csv}")
    print(f"Saving charts to: {os.path.abspath(OUT)}\n")

    # 3.1.1 Gender pie
    require_columns(df, ['2. Gender'])
    fig, ax = plt.subplots(figsize=(8, 5))
    gender = df['2. Gender'].value_counts()
    wedges, texts = ax.pie(gender.values, colors=[BLUE, PINK, GREEN], startangle=90,
                            wedgeprops=dict(edgecolor='none'))
    ax.set_aspect('equal')
    legend_labels = [f'{g} ({n})' for g, n in gender.items()]
    ax.legend(wedges, legend_labels, loc='lower right', bbox_to_anchor=(1.3, 0), fontsize=11)
    plt.tight_layout()
    save(fig, OUT, '3_1_1_gender_pie.png')

    # 3.1.2 College bar
    require_columns(df, ['College name'])
    fig, ax = plt.subplots(figsize=(9, 5))
    college = df['College name'].value_counts()
    bars = ax.bar(college.index, college.values, color=COLORS[:len(college)], edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, college.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8, str(val),
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_xlabel('College', fontsize=12)
    ax.set_ylabel('Number of Respondents', fontsize=12)
    ax.set_ylim(0, college.values.max() + 14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save(fig, OUT, '3_1_2_college_bar.png')

    # 3.1.3 Age bar
    require_columns(df, ['1. Age'])
    fig, ax = plt.subplots(figsize=(8, 5))
    age = df['1. Age'].value_counts().sort_index()
    bars = ax.bar(age.index.astype(str), age.values, color=BLUE, edgecolor='white', linewidth=1.5, width=0.6)
    for bar, val in zip(bars, age.values):
        pct = val / len(df) * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{val}\n({pct:.1f}%)',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_xlabel('Age (Years)', fontsize=12)
    ax.set_ylabel('Number of Respondents', fontsize=12)
    ax.set_ylim(0, age.values.max() + 22)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save(fig, OUT, '3_1_3_age_bar.png')

    # 3.1.4 Gender x College grouped bar
    fig, ax = plt.subplots(figsize=(10, 5))
    cross = pd.crosstab(df['College name'], df['2. Gender'])[['Male', 'Female']]
    colleges = cross.index.tolist()
    x = np.arange(len(colleges))
    width = 0.35
    b1 = ax.bar(x - width / 2, cross['Male'], width, label='Male', color=BLUE, edgecolor='white')
    b2 = ax.bar(x + width / 2, cross['Female'], width, label='Female', color=PINK, edgecolor='white')
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4, str(int(bar.get_height())),
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(colleges, fontsize=12)
    ax.set_xlabel('College', fontsize=12)
    ax.set_ylabel('Number of Respondents', fontsize=12)
    ax.legend(fontsize=12)
    ax.set_ylim(0, cross.values.max() + 12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save(fig, OUT, '3_1_4_gender_college_grouped.png')

    # 3.1.5 Internet usage pie
    require_columns(df, ['4. Daily internet usage'])
    fig, ax = plt.subplots(figsize=(8, 5))
    usage_order = ['0-2 Hours', '2-4 Hours', '4-6 Hours', 'More than 6 Hours']
    usage = df['4. Daily internet usage'].value_counts().reindex(usage_order)
    wedges, texts = ax.pie(usage.values, colors=BLUES4, startangle=90, wedgeprops=dict(edgecolor='none'))
    ax.set_aspect('equal')
    legend_labels = [f'{u} ({n})' for u, n in zip(usage.index, usage.values)]
    ax.legend(wedges, legend_labels, loc='lower right', bbox_to_anchor=(1.3, 0), fontsize=11)
    plt.tight_layout()
    save(fig, OUT, '3_1_5_internet_usage_pie.png')

    # 3.1.6 Technical vs Non-Technical pie
    require_columns(df, ['Scheme'])
    fig, ax = plt.subplots(figsize=(8, 5))
    scheme = df['Scheme'].value_counts()
    wedges, texts = ax.pie(scheme.values, colors=[BLUE, PINK], startangle=90, wedgeprops=dict(edgecolor='none'))
    ax.set_aspect('equal')
    legend_labels = [f'{s} ({n / scheme.sum() * 100:.1f}%)' for s, n in scheme.items()]
    ax.legend(wedges, legend_labels, loc='lower right', bbox_to_anchor=(1.3, 0), fontsize=11)
    plt.tight_layout()
    save(fig, OUT, '3_1_6_scheme_pie.png')

    # 3.2 Awareness - separate charts
    awareness_cols = {
        '3_2_1_awareness_cyber_threats.png': '10. I know the cyber threats',
        '3_2_2_awareness_unknown_links.png': '11. I understand risks of unknown links',
        '3_2_3_awareness_strong_passwords.png': '14. I know strong passwords are important',
        '3_2_4_awareness_2fa.png': '27. I use two-factor authentication (2FA) and secure privacy settings',
        '3_2_5_awareness_spot_suspicious_messages.png': '17. I can spot suspicious messages',
        '3_2_6_awareness_public_wifi_risk.png': '15. I know public Wi-Fi is risky',
    }
    for fname, col in awareness_cols.items():
        require_columns(df, [col])
        fig, ax = plt.subplots(figsize=(8, 4))
        likert_bar(ax, df, col)
        plt.tight_layout()
        save(fig, OUT, fname)

    # 3.3 safety practices grid
    safety_cols = [
        '23. I use strong passwords',
        '24. I avoid unknown links.',
        "25. I don't share OTP, passwords and sensitive information to other",
        '27. I use two-factor authentication (2FA) and secure privacy settings',
    ]
    require_columns(df, safety_cols)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    likert_bar(axes[0, 0], df, safety_cols[0])
    likert_bar(axes[0, 1], df, safety_cols[1])
    likert_bar(axes[1, 0], df, safety_cols[2])
    likert_bar(axes[1, 1], df, safety_cols[3])
    plt.tight_layout()
    save(fig, OUT, '3_3_safety_practices_grid.png')

    # 3.4 Summary comparison bar
    SUMMARY_QS = {
        'Know Cyber Threats': '10. I know the cyber threats',
        'Understand Link Risks': '11. I understand risks of unknown links',
        'Know Strong Passwords': '14. I know strong passwords are important',
        'Know Public Wi-Fi Risk': '15. I know public Wi-Fi is risky',
        'Spot Suspicious Messages': '17. I can spot suspicious messages',
        'Know 2FA': '27. I use two-factor authentication (2FA) and secure privacy settings',
        'Use Strong Passwords': '23. I use strong passwords',
        'Avoid Unknown Links': '24. I avoid unknown links.',
        "Don't Share OTP": "25. I don't share OTP, passwords and sensitive information to other",
    }
    require_columns(df, list(SUMMARY_QS.values()))
    pcts = {l: positive_pct(df[c]) for l, c in SUMMARY_QS.items()}
    labels = list(pcts.keys())
    values = list(pcts.values())
    colors = [BLUE if v >= 60 else ORANGE if v >= 40 else RED for v in values]
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(labels, values, color=colors, edgecolor='white', height=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, f'{val}%',
                 va='center', fontsize=12, fontweight='bold')
    ax.axvline(60, color='gray', linestyle='--', alpha=0.5, label='60% reference line')
    ax.set_xlim(0, 108)
    ax.set_xlabel('% Agree or Strongly Agree', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=12)
    plt.tight_layout()
    save(fig, OUT, '3_summary_bar.png')

    # Faculty distribution
    require_columns(df, ['Faculty'])
    fig, ax = plt.subplots(figsize=(9, 5))
    faculty = df['Faculty'].value_counts()
    faculty_pct = faculty / len(df) * 100
    bars = ax.bar(faculty.index, faculty_pct.values, color=COLORS[:len(faculty)], edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, faculty_pct.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{val:.1f}%',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_xlabel('Faculty', fontsize=12)
    ax.set_ylabel('Percentage of Respondents (%)', fontsize=12)
    ax.set_ylim(0, faculty_pct.values.max() + 8)
    ax.tick_params(axis='x', labelrotation=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save(fig, OUT, 'faculty_distribution.png')

    # Risky behaviors
    RISKY_QS = {
        'Same Password on Multiple Platforms': '33. I keep same password on multiple platforms',
        'Click Links Without Checking': '34. I click on links without checking them',
        'Accept T&C Without Reading': '35. I accept terms and conditions of software/apps without studying properly',
        'Ignore Risks Even When Known': '32. Even if I know risks, I ignore them.',
    }
    require_columns(df, list(RISKY_QS.values()))
    risky_pcts = {l: positive_pct(df[c]) for l, c in RISKY_QS.items()}
    labels = list(risky_pcts.keys())
    values = list(risky_pcts.values())
    colors = [RED if v >= 50 else ORANGE if v >= 30 else GREEN for v in values]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels, values, color=colors, edgecolor='white', height=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, f'{val}%',
                 va='center', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_xlabel('% Agree or Strongly Agree (Higher = More Risky)', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save(fig, OUT, 'risky_behaviors.png')

    # Cyber exposure
    EXPOSURE_QS = {
        'Received Suspicious Messages/Links': '37. I receive suspicious messages or links',
        'Faced Cyber Risks (Fraud/Scams/Hacking)': '38. I have faced cyber risks (fraud, scams, hacking)',
        'Incidents Affected Me or Someone I Know': '39. Cyber incidents affected me or someone I know',
        'Feel Unsafe Using Internet Sometimes': '40. I feel unsafe using the internet sometimes',
    }
    require_columns(df, list(EXPOSURE_QS.values()))
    exp_pcts = {l: positive_pct(df[c]) for l, c in EXPOSURE_QS.items()}
    labels = list(exp_pcts.keys())
    values = list(exp_pcts.values())
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels, values, color=[ORANGE, RED, RED, PURPLE], edgecolor='white', height=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, f'{val}%',
                 va='center', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_xlabel('% Agree or Strongly Agree', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save(fig, OUT, 'cyber_exposure.png')

    print("\nDone. All charts saved to:", os.path.abspath(OUT))


if __name__ == "__main__":
    main()