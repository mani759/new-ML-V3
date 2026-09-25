"""
MPLADS Risk Engine v3 - real eSAKSHI data (18th Lok Sabha + Rajya Sabha).

One set of functions is used for BOTH batch scoring (all works) and live
scoring (one new work), so the live form and the dashboard always agree.

  bench = build_benchmarks(works, payments, as_of)     # once, saved to bench.pkl
  scored = score_frame(works, payments, bench, models, as_of)
  one = score_one(work_dict, bench, models)            # live "Analyze" form

Design and thresholds: see "Model design (agreed)" in the data-understanding doc.
"""
import re
import numpy as np
import pandas as pd

MODEL_VERSION = '3.0.0'
WEIGHT = {'high': 75, 'medium': 40, 'low': 12}
ML_WEIGHT = 30
EARLY_STAGES = ('Sanction', 'Time Estimation', 'Vendor Identification')
OPEN_STAGES = ('Sanction', 'Time Estimation', 'Vendor Identification', 'Work partially Completed',
               'Physical Inspection')

_CIVIL = re.compile(r'construct|road|drain|building|hall|bridge|culvert|boundary wall|compound wall|'
                    r'\bwall\b|shed|toilet|pond|renovat|repair|bhawan|bhavan|nirman|pathway|paving|'
                    r'interlock|check dam|platform|stadium|godown|centre building|center building', re.I)
_PURCHASE = re.compile(r'purchase|supply|procure|books|furniture|equipment|vehicle|ambulance|computer|'
                       r'smart board|projector|tanker|bus\b|van\b|machine|kit\b|laptop', re.I)
_BUNDLED = re.compile(r'various|different (?:places|locations|blocks|villages|sites)|'
                      r'\d+\s*(?:places|locations|villages|blocks|sites|nos? of places)|multiple|'
                      r'as per (?:the )?(?:attach|list|enclos)|attached list', re.I)
_VAGUE = re.compile(r'as per (?:the )?attach|see (?:the )?attach|letter as per|as per letter|enclosed', re.I)


# ---------------------------------------------------------------- helpers
def _norm_text(s: pd.Series) -> pd.Series:
    return (s.fillna('').str.lower().str.replace(r'\btq\b|\btaluka\b|\btaluk\b', 'tq', regex=True)
            .str.replace(r'\bdist\b|\bdistrict\b', 'dist', regex=True)
            .str.replace(r'[^a-z0-9]', '', regex=True))


def _norm_mp(s: pd.Series) -> pd.Series:
    return (s.fillna('').str.lower().str.replace(r'\(\d{4}-\d{2,4}\)', '', regex=True)
            .str.replace(r'^(shri|smt|dr|ms|mr|adv|km)\.?\s+', '', regex=True)
            .str.replace(r'[^a-z ]', '', regex=True).str.split().str.join(' '))


def _num(x):
    """JSON-safe number: NaN/inf -> None."""
    return None if x is None or not np.isfinite(float(x)) else float(x)


def _rs(x) -> str:
    x = float(x)
    return f"Rs {x / 1e7:.2f} Cr" if abs(x) >= 1e7 else f"Rs {x / 1e5:.1f} L"


def prepare(works: pd.DataFrame) -> pd.DataFrame:
    """Parse dates, derive text/type columns. Safe to call on one row."""
    w = works.copy()
    for c in ('recommendation_date', 'sanction_date', 'completion_date', 'first_payment_date', 'last_payment_date'):
        if c in w:
            w[c] = pd.to_datetime(w[c], errors='coerce')
    for c, d in (('total_paid', 0.0), ('payment_count', 0), ('vendor_count', 0),
                 ('in_recommended_list', True), ('in_sanctioned_list', False), ('is_completed', False)):
        if c not in w:
            w[c] = d
        w[c] = w[c].fillna(d)
    sanc = w['sanction_amount'] if 'sanction_amount' in w else pd.Series(np.nan, index=w.index)
    w['amount'] = pd.to_numeric(sanc, errors='coerce').fillna(pd.to_numeric(w['recommended_amount'], errors='coerce'))
    text = w['activity_type'].fillna('') + ' ' + w['work_description'].fillna('')
    w['kind'] = np.where(text.str.contains(_CIVIL), 'construction',
                         np.where(text.str.contains(_PURCHASE), 'purchase', 'other'))
    d = w['work_description'].fillna('')
    w['is_bundled'] = d.str.contains(_BUNDLED)
    w['is_vague'] = d.str.contains(_VAGUE) | (d.str.strip().str.len() < 15)
    w['desc_key'] = _norm_text(d)
    w['mp_key'] = _norm_mp(w['mp_name'])
    w['rejected'] = w.get('stage', pd.Series('', index=w.index)).eq('Rejected/Withdrawn')
    return w


# ---------------------------------------------------------------- benchmarks
def build_benchmarks(works: pd.DataFrame, payments: pd.DataFrame, as_of: str,
                     allocations: pd.DataFrame = None) -> dict:
    w = prepare(works)
    w = w[~w['rejected'] & (w['amount'] > 100)]
    w['la'] = np.log(w['amount'])
    peer = {}
    for keys in (['activity_type', 'kind', 'state'], ['activity_type', 'kind'], ['activity_type'], []):
        if keys:
            g = w.groupby(keys)['la']
            st = pd.DataFrame({'med': g.median(), 'mad': g.apply(lambda x: (x - x.median()).abs().median()),
                               'n': g.size()})
            peer[tuple(keys)] = st[st['n'] >= 20].to_dict('index')
        else:
            peer[()] = {(): {'med': w['la'].median(), 'mad': (w['la'] - w['la'].median()).abs().median(),
                             'n': len(w)}}
    s = w[w['in_sanctioned_list'] & w['recommendation_date'].notna() & w['sanction_date'].notna()]
    lag = (s['sanction_date'] - s['recommendation_date']).dt.days
    g = lag.groupby(s['ida'])
    ida_lag = pd.DataFrame({'p90': g.quantile(0.9), 'med': g.median(), 'n': g.size()})
    bench = {
        'model_version': MODEL_VERSION, 'as_of': as_of,
        'peer': peer,
        'ida_lag': ida_lag[ida_lag['n'] >= 30][['p90', 'med']].to_dict('index'),
        'nat_lag_p90': float(lag.quantile(0.9)), 'nat_lag_med': float(lag.median()),
    }
    # stored-work index for live duplicate checks
    idx = w[w['desc_key'].str.len() >= 25][['house', 'mp_key', 'ida', 'desc_key', 'work_id', 'amount', 'total_paid', 'mp_name']]
    bench['dup_index'] = idx.reset_index(drop=True)
    bench['batch_keys'] = set(map(tuple, w.groupby(['house', 'mp_key', 'desc_key']).size()
                                  .loc[lambda x: x >= 5].index.tolist()))
    bench['mp'] = mp_level(works, payments, allocations, as_of)
    return bench


def _peer_stats(w: pd.DataFrame, bench: dict):
    med = pd.Series(np.nan, index=w.index)
    mad = pd.Series(np.nan, index=w.index)
    for keys in (('activity_type', 'kind', 'state'), ('activity_type', 'kind'), ('activity_type',), ()):
        table = bench['peer'][keys]
        need = med.isna()
        if not need.any():
            break
        if keys:
            k = list(zip(*[w.loc[need, c] for c in keys])) if len(keys) > 1 else list(w.loc[need, keys[0]])
        else:
            k = [()] * int(need.sum())
        vals = [table.get(x) for x in k]
        med.loc[need] = [v['med'] if v else np.nan for v in vals]
        mad.loc[need] = [v['mad'] if v else np.nan for v in vals]
    return med, mad


# ---------------------------------------------------------------- features
def features(works: pd.DataFrame, payments: pd.DataFrame, bench: dict, as_of) -> pd.DataFrame:
    w = prepare(works)
    ref = pd.Timestamp(as_of).normalize()
    med, mad = _peer_stats(w, bench)
    la = np.log(w['amount'].clip(lower=1))
    w['peer_median'] = np.exp(med)
    w['cost_ratio'] = w['amount'] / w['peer_median']
    w['cost_z'] = (la - med) / (1.4826 * mad.where(mad > 0))
    lag_tab = bench['ida_lag']
    w['ida_lag_p90'] = w['ida'].map(lambda x: lag_tab.get(x, {}).get('p90')).fillna(bench['nat_lag_p90'])
    w['ida_lag_med'] = w['ida'].map(lambda x: lag_tab.get(x, {}).get('med')).fillna(bench['nat_lag_med'])
    w['sanction_lag'] = (w['sanction_date'] - w['recommendation_date']).dt.days
    w['pending_days'] = np.where(~w['in_sanctioned_list'], (ref - w['recommendation_date']).dt.days, np.nan)
    w['is_open'] = w['in_sanctioned_list'] & ~w['is_completed'] & ~w['rejected']
    w['days_since_sanction'] = (ref - w['sanction_date']).dt.days
    w['days_since_payment'] = (ref - w['last_payment_date']).dt.days
    w['completion_days'] = (w['completion_date'] - w['sanction_date']).dt.days
    w['paid_frac'] = (w['total_paid'] / w['amount']).where(w['amount'] > 0).fillna(0).clip(0, 1)
    mp = bench['mp'].set_index(['house', 'mp_key'])
    k = list(zip(w['house'], w['mp_key']))
    w['mp_top_vendor_share'] = [mp['top_vendor_share'].get(x, np.nan) for x in k]
    if payments is not None and len(payments):
        key = ['house', 'work_id', 'vendor_name', 'payment_date', 'amount']
        rep = payments[payments.duplicated(subset=key, keep=False)]
        rep_ids = set(zip(rep['house'], rep['work_id']))
        w['repeated_payment'] = [x in rep_ids for x in zip(w['house'], w['work_id'])]
    else:
        w['repeated_payment'] = False
    return w


# ---------------------------------------------------------------- duplicates
def duplicate_flags(w: pd.DataFrame, bench: dict = None) -> pd.DataFrame:
    """Batch: groups inside w. Live (bench given, w has 1 row): compare with stored works."""
    out = pd.DataFrame({'dup_level': None, 'dup_with': None, 'cross_mp': False}, index=w.index)
    spec = (w['desc_key'].str.len() >= 25) & ~w['rejected']
    if bench is not None:
        idx = bench['dup_index']
        for i, r in w[spec].iterrows():
            if (r['house'], r['mp_key'], r['desc_key']) in bench['batch_keys']:
                continue
            same = idx[(idx['house'] == r['house']) & (idx['mp_key'] == r['mp_key']) &
                       (idx['desc_key'] == r['desc_key']) & (idx['work_id'] != r.get('work_id'))]
            same = same[(same['amount'] - r['amount']).abs() <= 0.10 * np.maximum(same['amount'], r['amount'])]
            if len(same):
                all_paid = r['total_paid'] > 0 and bool((same['total_paid'] > 0).all())
                out.at[i, 'dup_level'] = 'high' if all_paid else 'medium'
                out.at[i, 'dup_with'] = ', '.join(map(str, same['work_id'].head(5)))
            cm = idx[(idx['ida'] == r['ida']) & (idx['desc_key'] == r['desc_key']) & (idx['mp_key'] != r['mp_key'])]
            if len(cm):
                out.at[i, 'cross_mp'] = True
                out.at[i, 'dup_with'] = ', '.join(map(str, cm['work_id'].head(5)))
        return out
    s = w[spec].copy()
    s['n'] = s.groupby(['house', 'mp_key', 'desc_key'])['desc_key'].transform('size')
    cand = s[s['n'].between(2, 4)].copy()
    g = cand.groupby(['house', 'mp_key', 'desc_key'])
    cand['amt_spread'] = g['amount'].transform(lambda x: (x.max() - x.min()) / x.max() if x.max() else 1)
    cand = cand[cand['amt_spread'] <= 0.10]
    g = cand.groupby(['house', 'mp_key', 'desc_key'])
    all_paid = g['total_paid'].transform(lambda x: (x > 0).all())
    ids = g['work_id'].transform(lambda x: ', '.join(map(str, x)))
    out.loc[cand.index, 'dup_level'] = np.where(all_paid, 'high', 'medium')
    out.loc[cand.index, 'dup_with'] = ids
    c = w[spec].copy()
    c['mps'] = c.groupby(['ida', 'desc_key'])['mp_key'].transform('nunique')
    cm = c[c['mps'] >= 2]
    out.loc[cm.index, 'cross_mp'] = True
    out.loc[cm.index, 'dup_with'] = out.loc[cm.index, 'dup_with'].fillna(
        cm.groupby(['ida', 'desc_key'])['work_id'].transform(lambda x: ', '.join(map(str, x))))
    return out


# ---------------------------------------------------------------- rules
def apply_rules(f: pd.DataFrame, dup: pd.DataFrame) -> pd.Series:
    """Returns, per work, a list of (flag, severity, reason)."""
    res = [[] for _ in range(len(f))]
    pos = {ix: n for n, ix in enumerate(f.index)}

    def add(mask, flag, sev, text_fn):
        for ix, r in f[mask.fillna(False).astype(bool)].iterrows():
            s = sev(r) if callable(sev) else sev
            res[pos[ix]].append((flag, s, text_fn(r)))

    live = ~f['rejected']
    # --- delays
    overdue = f['is_open'] & (f['days_since_sanction'] > 365)
    stalled = (f['payment_count'] == 0) | (f['days_since_payment'] > 180)
    add(overdue & stalled, 'overdue_stalled', 'high', lambda r:
        f"Sanctioned {int(r.days_since_sanction)} days ago (guideline: complete within 365), still at "
        f"'{r.stage}', and " + ("no payment has ever been made." if r.payment_count == 0 else
                               f"no payment for {int(r.days_since_payment)} days."))
    add(overdue & ~stalled, 'overdue', 'medium', lambda r:
        f"Sanctioned {int(r.days_since_sanction)} days ago, past the 365-day completion guideline; "
        f"payments are still being made (last {int(r.days_since_payment)} days ago).")
    pend = live & ~f['in_sanctioned_list'] & (f['pending_days'] > 45) & (f['pending_days'] > f['ida_lag_p90'])
    add(pend, 'sanction_pending', lambda r: 'high' if r.pending_days > 365 else 'medium', lambda r:
        f"Waiting {int(r.pending_days)} days for sanction; the guideline is 45 days, and 90% of works in "
        f"{r.ida.split('(')[0].title()} were sanctioned within {int(r.ida_lag_p90)} days.")
    # --- cost
    # Option B (agreed): over 4x peer median AND robust z over 2 (top ~2-3% of similar works)
    exp_ = live & (f['cost_ratio'] > 4) & (f['cost_z'] > 2) & ~f['is_bundled']
    add(exp_, 'unusually_expensive', 'medium', lambda r:
        f"{_rs(r.amount)} is {r.cost_ratio:.1f}x the median of similar works "
        f"({r.activity_type}, {r.kind}; median {_rs(r.peer_median)}).")
    add(live & f['is_bundled'] & (f['cost_ratio'] > 3), 'bundled_work', 'low', lambda r:
        f"One work covering many locations ({_rs(r.amount)}); cost per location cannot be verified "
        f"from the public record.")
    add(live & f['is_vague'] & (f['cost_ratio'] > 1) & ~f['is_bundled'], 'vague_description', 'low', lambda r:
        f"Description is only '{str(r.work_description)[:60]}' for {_rs(r.amount)}; the public record does "
        f"not say what is being built or bought.")
    add(f['is_completed'] & (f['actual_cost'] < 0.5 * f['amount']), 'completed_far_below_sanction', 'medium',
        lambda r: f"Completed for {_rs(r.actual_cost)}, under half of the sanctioned {_rs(r.amount)}.")
    # --- duplicates
    d_hi = dup['dup_level'].eq('high')
    d_md = dup['dup_level'].eq('medium')
    add(d_hi, 'duplicate_paid', 'high', lambda r:
        f"Same MP, same location-specific description and amount within 10% as works {dup.at[r.name, 'dup_with']}; "
        f"all copies have been paid.")
    add(d_md, 'possible_duplicate', 'medium', lambda r:
        f"Same MP, same description and amount within 10% as works {dup.at[r.name, 'dup_with']}; "
        f"check before sanctioning or paying.")
    add(dup['cross_mp'] & ~d_hi & ~d_md, 'same_work_other_mp', 'medium', lambda r:
        f"The same work in the same district is also recommended by another MP (works {dup.at[r.name, 'dup_with']}); "
        f"confirm it is co-funding, not double funding.")
    # --- misuse / norms
    add(f['is_open'] & f['stage'].isin(EARLY_STAGES) & (f['paid_frac'] >= 0.9) & (f['days_since_payment'] > 180),
        'paid_but_stalled', 'high', lambda r:
        f"{r.paid_frac:.0%} of {_rs(r.amount)} paid, but the work is still at '{r.stage}' and nothing has "
        f"been paid for {int(r.days_since_payment)} days.")
    fast = f['is_completed'] & f['kind'].eq('construction') & (f['completion_days'] >= 0)
    add(fast & (f['completion_days'] == 0), 'construction_same_day', 'high', lambda r:
        f"Construction work ({r.activity_type}) marked completed on the same day it was sanctioned.")
    add(fast & f['completion_days'].between(1, 7), 'construction_within_7_days', 'medium', lambda r:
        f"Construction work ({r.activity_type}) marked completed {int(r.completion_days)} day(s) after sanction.")
    add(f['in_sanctioned_list'] & ~f['in_recommended_list'], 'no_recommendation_record', 'medium', lambda r:
        "Sanctioned, but the work has no record in the MP-recommended list.")
    add(f['is_completed'] & (f['payment_count'] == 0), 'completed_without_payment', 'medium', lambda r:
        "Marked completed, but no vendor payment is recorded for it.")
    add(f['repeated_payment'], 'repeated_payment_entries', 'low', lambda r:
        "The same payment (vendor, date and amount) is recorded more than once; total paid is still within "
        "the sanction, so these are probably equal instalments.")
    return pd.Series(res, index=f.index)


# ---------------------------------------------------------------- ML features
ML_FEATURES = ['cost_z_c', 'lag_rel', 'age_rel', 'paid_frac', 'payment_count', 'idle_rel',
               'vendor_count', 'mp_top_vendor_share_f', 'completion_rel', 'is_completed_f']


def ml_matrix(f: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=f.index)
    x['cost_z_c'] = f['cost_z'].clip(-6, 8).fillna(0)
    x['lag_rel'] = (f['sanction_lag'] / f['ida_lag_med'].clip(lower=1)).clip(0, 10).fillna(1)
    x['age_rel'] = np.where(f['is_open'], f['days_since_sanction'] / 365, 0).clip(0, 4)
    x['paid_frac'] = f['paid_frac']
    x['payment_count'] = f['payment_count'].clip(0, 20)
    x['idle_rel'] = np.where(f['is_open'] & (f['payment_count'] > 0), f['days_since_payment'] / 180, 0).clip(0, 5)
    x['vendor_count'] = f['vendor_count'].clip(0, 10)
    x['mp_top_vendor_share_f'] = f['mp_top_vendor_share'].fillna(0.3)
    x['completion_rel'] = np.where(f['is_completed'], f['completion_days'] / 365, 0).clip(0, 4)
    x['is_completed_f'] = f['is_completed'].astype(float)
    return x[ML_FEATURES].astype(float)


def ml_reason(row) -> str:
    parts = []
    if abs(row['cost_z_c']) >= 2:
        parts.append(f"cost {row['cost_z_c']:.1f} standard deviations from similar works")
    if row['lag_rel'] >= 3:
        parts.append(f"sanction took {row['lag_rel']:.1f}x the district's usual time")
    if row['idle_rel'] >= 2:
        parts.append(f"no payment for {row['idle_rel'] * 180:.0f} days")
    if row['payment_count'] >= 8:
        parts.append(f"{int(row['payment_count'])} separate payments")
    if row['vendor_count'] >= 3:
        parts.append(f"{int(row['vendor_count'])} different vendors")
    if row['mp_top_vendor_share_f'] >= 0.6:
        parts.append(f"the MP's largest vendor receives {row['mp_top_vendor_share_f']:.0%} of payments")
    tail = f" Main factors: {'; '.join(parts)}." if parts else ""
    return ("No single rule is broken, but the combination of cost, timing, payments and vendors is "
            "unusual compared with other MPLADS works." + tail)


# ---------------------------------------------------------------- scoring
def combine(flags: list, ml_flag: bool) -> tuple:
    p = 1.0
    for _, sev, _ in flags:
        p *= 1 - WEIGHT[sev] / 100
    if ml_flag:
        p *= 1 - ML_WEIGHT / 100
    score = int(round(100 * (1 - p)))
    severity = 'high' if score >= 70 else 'medium' if score >= 40 else 'low' if score >= 12 else 'none'
    return score, severity


def score_frame(works: pd.DataFrame, payments: pd.DataFrame, bench: dict, models: dict, as_of) -> pd.DataFrame:
    f = features(works, payments, bench, as_of)
    dup = duplicate_flags(f)
    rules = apply_rules(f, dup)
    x = ml_matrix(f)
    iso = models['iforest']
    raw = -iso.score_samples(x.values)
    ml_flag = (raw > models['iforest_threshold']) & rules.apply(len).eq(0) & ~f['rejected']
    out = f[['house', 'work_id', 'state', 'ida', 'mp_name', 'constituency', 'activity_type', 'kind', 'stage',
             'amount', 'total_paid']].copy()
    scores = [combine(fl, m) for fl, m in zip(rules, ml_flag)]
    out['risk_score'] = [s for s, _ in scores]
    out['severity'] = [s for _, s in scores]
    out['flags'] = [[a for a, _, _ in fl] + (['statistical_anomaly'] if m else []) for fl, m in zip(rules, ml_flag)]
    out['reasons'] = [[c for _, _, c in fl] + ([ml_reason(x.loc[i])] if m else [])
                      for i, fl, m in zip(f.index, rules, ml_flag)]
    out['ml_anomaly_score'] = raw.round(4)
    out['delay_risk'] = delay_risk(f, models)
    out['as_of'] = pd.Timestamp(as_of).strftime('%Y-%m-%d')
    out['model_version'] = MODEL_VERSION
    return out


def score_one(work: dict, bench: dict, models: dict, as_of=None) -> dict:
    """Live scoring of one work (no payments table needed)."""
    as_of = as_of or pd.Timestamp.now().strftime('%Y-%m-%d')
    w = pd.DataFrame([work])
    for c in ('house', 'state', 'ida', 'mp_name', 'constituency', 'activity_type', 'work_description', 'stage',
              'recommendation_date', 'sanction_date', 'completion_date', 'last_payment_date', 'actual_cost', 'work_id'):
        if c not in w:
            w[c] = None
    w['stage'] = w['stage'].fillna('Pending for Sanction')
    if 'work_category' not in w or w['work_category'].isna().all():
        w['work_category'] = 'Normal/Others'
    f = features(w, None, bench, as_of)
    dup = duplicate_flags(f, bench)
    rules = apply_rules(f, dup)
    x = ml_matrix(f)
    raw = float(-models['iforest'].score_samples(x.values)[0])
    ml_flag = bool(raw > models['iforest_threshold'] and len(rules.iloc[0]) == 0)
    score, sev = combine(rules.iloc[0], ml_flag)
    fl = rules.iloc[0]
    r = f.iloc[0]
    return {
        'work_id': work.get('work_id'), 'risk_score': score, 'severity': sev,
        'flags': [a for a, _, _ in fl] + (['statistical_anomaly'] if ml_flag else []),
        'reasons': [c for _, _, c in fl] + ([ml_reason(x.iloc[0])] if ml_flag else []),
        'features': {'amount': float(r['amount']), 'peer_median': _num(round(float(r['peer_median']), 0)),
                     'cost_ratio': _num(round(float(r['cost_ratio']), 2)), 'kind': r['kind'],
                     'district_sanction_p90_days': int(r['ida_lag_p90']),
                     'days_since_sanction': _num(r['days_since_sanction']), 'paid_frac': round(float(r['paid_frac']), 3)},
        'ml_anomaly_score': round(raw, 4), 'delay_risk': _num(delay_risk(f, models)[0]),
        'as_of': as_of, 'model_version': MODEL_VERSION,
    }


# ---------------------------------------------------------------- delay prediction
DELAY_CAT = ['activity_type', 'kind', 'state', 'house', 'work_category']
DELAY_NUM = ['log_amount', 'cost_z_c', 'sanction_lag', 'ida_lag_med', 'ida_hist_dur', 'sanction_month']


def delay_frame(f: pd.DataFrame, ida_dur: dict, nat_dur: float) -> pd.DataFrame:
    x = pd.DataFrame(index=f.index)
    for c in DELAY_CAT:
        x[c] = f[c].fillna('NA').astype(str)
    x['log_amount'] = np.log(f['amount'].clip(lower=1))
    x['cost_z_c'] = f['cost_z'].clip(-6, 8).fillna(0)
    x['sanction_lag'] = f['sanction_lag'].fillna(f['ida_lag_med'])
    x['ida_lag_med'] = f['ida_lag_med']
    x['ida_hist_dur'] = f['ida'].map(ida_dur).fillna(nat_dur)
    x['sanction_month'] = f['sanction_date'].dt.month.fillna(0)
    return x


def delay_risk(f: pd.DataFrame, models: dict) -> np.ndarray:
    """Probability an OPEN work under 365 days old will pass 365 days. NaN otherwise."""
    m = models.get('delay')
    out = np.full(len(f), np.nan)
    if m is None:
        return out
    target = (f['is_open'] & (f['days_since_sanction'] <= 365)).values
    if target.any():
        x = delay_frame(f[target], m['ida_dur'], m['nat_dur'])
        for c in DELAY_CAT:
            x[c] = pd.Categorical(x[c], categories=m['cats'][c])
        out[target] = m['clf'].predict_proba(x[DELAY_CAT + DELAY_NUM])[:, 1].round(3)
    return out


# ---------------------------------------------------------------- MP level
def mp_level(works: pd.DataFrame, payments: pd.DataFrame, allocations: pd.DataFrame, as_of) -> pd.DataFrame:
    w = works.copy()
    w['mp_key'] = _norm_mp(w['mp_name'])
    w = w[w.get('stage', '') != 'Rejected/Withdrawn']
    mp = w.groupby(['house', 'mp_key']).agg(mp_name=('mp_name', 'first'), state=('state', 'first'),
                                            works=('work_id', 'size'), recommended=('recommended_amount', 'sum'),
                                            paid=('total_paid', 'sum')).reset_index()
    if payments is not None and len(payments):
        p = payments.copy()
        p['mp_key'] = _norm_mp(p['mp_name'])
        tot = p.groupby(['house', 'mp_key']).agg(pay_total=('amount', 'sum'), paid_works=('work_id', 'nunique'))
        top = (p.groupby(['house', 'mp_key', 'vendor_name'])['amount'].sum().reset_index()
               .sort_values('amount', ascending=False).drop_duplicates(['house', 'mp_key'])
               .set_index(['house', 'mp_key']).rename(columns={'vendor_name': 'top_vendor', 'amount': 'top_vendor_amt'}))
        mp = mp.join(tot, on=['house', 'mp_key']).join(top, on=['house', 'mp_key'])
        mp['top_vendor_share'] = mp['top_vendor_amt'] / mp['pay_total']
    if allocations is not None:
        a = allocations.copy()
        a['mp_key'] = _norm_mp(a['mp_name'])
        mp = a[['house', 'mp_key', 'mp_name', 'state', 'allocated', 'term_start']].merge(
            mp.drop(columns=['mp_name', 'state']), on=['house', 'mp_key'], how='left')
        mp[['works', 'recommended', 'paid']] = mp[['works', 'recommended', 'paid']].fillna(0)
        mp['utilisation'] = mp['recommended'] / mp['allocated']
    alerts = [[] for _ in range(len(mp))]
    for i, r in mp.iterrows():
        if r.get('paid_works', 0) >= 10 and r.get('top_vendor_share', 0) > 0.6:
            alerts[i].append(('vendor_concentration', 'medium',
                              f"{r['top_vendor']} received {r['top_vendor_share']:.0%} of this MP's payments "
                              f"({_rs(r['top_vendor_amt'])} of {_rs(r['pay_total'])})."))
        eligible = r['house'] == 'LS' or (pd.notna(r.get('term_start')) and r.get('term_start') <= 2025)
        if allocations is not None and eligible and pd.notna(r.get('allocated')) and r['utilisation'] < 0.25:
            alerts[i].append(('low_fund_utilisation', 'medium',
                              f"Recommended {_rs(r['recommended'])} of {_rs(r['allocated'])} allocated "
                              f"({r['utilisation']:.0%})" + ("; no works on eSAKSHI." if r['works'] == 0 else ".")))
    mp['alerts'] = alerts
    return mp
