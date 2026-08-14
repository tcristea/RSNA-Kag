# V40 notebook — fixes, cell by cell

Cell numbers follow the notebook's own order (0-indexed, 32 cells total).
Each fix shows exact code: either a full cell replacement or a precise old → new edit.

---

## FIX 1 — Cell 31: demote the receipt from gatekeeper to witness

**Problem:** raises on E10 promotion status, config name, hash, and even on a missing
`rad_e10_audit.json` — killing an already-valid `submission.csv`. It also makes the
e9/e9b fallback branches in cell 30 unreachable-in-success (they write differently
named audit files).

**Full replacement for cell 31:**

```python
# V40 runtime receipt: fail-closed on submission validity, log-only on arm status.
import hashlib as _v40_hashlib
import json as _v40_json
from pathlib import Path as _V40Path
import numpy as _v40_np
import pandas as _v40_pd

_v40_work = _V40Path('/kaggle/working')
_v40_primary = _v40_work / 'submission.csv'
_v40_parent = _v40_work / 'submission_e2_preserved.csv'

# Use the primary pipeline's ROOT; COMP (defined by the A5 block) may be None.
_v40_data = _V40Path(str(ROOT)) if 'ROOT' in globals() else COMP
_v40_test = _v40_pd.read_csv(_v40_data / 'test.csv', dtype={'StudyInstanceUID': str})
_v40_sub = _v40_pd.read_csv(_v40_primary, dtype={'StudyInstanceUID': str})

# ---- HARD checks: these protect the submission itself. Keep raising. ----
_v40_expected = ['StudyInstanceUID', *TARGETS]
if _v40_sub.columns.tolist() != _v40_expected:
    raise RuntimeError('V40 submission schema drift')
if _v40_sub.StudyInstanceUID.tolist() != _v40_test.StudyInstanceUID.astype(str).tolist():
    raise RuntimeError('V40 dynamic test identity/order drift')
_v40_values = _v40_sub[TARGETS].to_numpy(float)
if not _v40_np.isfinite(_v40_values).all() or _v40_values.min() < 0 or _v40_values.max() > 1:
    raise RuntimeError('V40 invalid submission values')

# ---- SOFT checks: arm provenance. Record, never raise. ----
_v40_sha = lambda p: _v40_hashlib.sha256(p.read_bytes()).hexdigest()
_v40_arm = {'e10_status': 'AUDIT_FILE_MISSING', 'audit_file': None}
for _v40_name in ('rad_e10_audit.json', 'rad_e9b_audit.json', 'rad_e9_audit.json'):
    _v40_p = _v40_work / _v40_name
    if _v40_p.is_file():
        _v40_a = _v40_json.loads(_v40_p.read_text())
        _v40_arm = {
            'audit_file': _v40_name,
            'e10_status': _v40_a.get('status'),
            'configuration': _v40_a.get('configuration'),
            'promoted_hash_matches': (
                _v40_a.get('selected_sha256') == _v40_sha(_v40_primary)
                if _v40_a.get('selected_sha256') else None
            ),
            'error': _v40_a.get('error'),
        }
        break

_v40_receipt = {
    'status': 'VALID_SUBMISSION',
    'test_studies': len(_v40_sub),
    'dynamic_test_ids_exact': True,
    'schema_exact': True,
    'finite_in_range': True,
    'rad_arm': _v40_arm,
    'a5_blend': globals().get('A5_RECEIPT', 'not recorded'),
    'parent_preserved_exists': _v40_parent.is_file(),
    'parent_sha256': _v40_sha(_v40_parent) if _v40_parent.is_file() else None,
    'submission_sha256': _v40_sha(_v40_primary),
}
(_v40_work / 'v40_runtime_audit.json').write_text(
    _v40_json.dumps(_v40_receipt, indent=2, sort_keys=True) + '\n'
)
print(_v40_json.dumps(_v40_receipt, indent=2, sort_keys=True))
```

---

## FIX 2 — Cell 30: make the E10 audit tell the truth about its parent

**Problem:** E10 was gated against pure-E2 OOF ranks but deploys over the post-A5
file, and its audit claims the E2 parent it doesn't have.

**Minimal honest edit** (inside `main_v52_e10`, the `audit = {...}` dict):

```python
# OLD
"parent": "E2 captured 20-member DINOv2 rank ensemble",

# NEW
"parent": (
    "V40 primary at deploy time: DINOv2 public frontier + cross-series DINOv3, "
    "A5 rank blend at 0.45 (NOT the pure E2 parent the OOF gate was computed on)"
),
```

**And record the transfer risk explicitly** — add one line right below it:

```python
"gate_transfer_caveat": (
    "alpha ladder was audited on E2-only OOF ranks; deployment parent includes "
    "the A5 arm, so audited per-target gains are not guaranteed to transfer"
),
```

**The real decision (pick one, no code can dodge it):**
- **(a) Accept the risk consciously** — ship with the caveat above. Cheapest.
- **(b) Re-audit against the true parent** — requires A5 OOF predictions on the
  58 gold studies to rebuild the ladder baseline. Correct but new work.
- **(c) Reorder the arms** — run E10 before A5 so it deploys on the parent it was
  audited on, then let A5 blend last. One-cell move, but then A5 dilutes the
  audited E10 correction instead. Only do this if you trust A5 less than E10.

---

## FIX 3 — Cells 25–29: give A5 a safety net

### 3a. Cell 25 — asserts become a flag

```python
# OLD
assert COMP is not None, 'competition data not attached'
assert CKPT is not None, 'fold weights not attached'
assert (COMP / 'sample_submission.csv').exists(), f'no competition data at {COMP}'
assert list(CKPT.glob('*_f*.pt')), f'no checkpoints at {CKPT}'

# NEW
A5_ENABLED, _a5_reasons = True, []
if COMP is None:
    A5_ENABLED = False; _a5_reasons.append('competition data not attached')
elif not (COMP / 'sample_submission.csv').exists():
    A5_ENABLED = False; _a5_reasons.append(f'no competition data at {COMP}')
if CKPT is None:
    A5_ENABLED = False; _a5_reasons.append('fold weights not attached')
elif not list(CKPT.glob('*_f*.pt')):
    A5_ENABLED = False; _a5_reasons.append(f'no checkpoints at {CKPT}')
if not A5_ENABLED:
    print('a5: DISABLED —', '; '.join(_a5_reasons))
```

Then wrap the **bodies** of cells 26, 27, and 28 in:

```python
if not A5_ENABLED:
    print('a5: disabled, cell skipped')
else:
    <existing cell body, indented>
```

One exception: the globals-restore loop at the very end of cell 28
(`for _a5k, _a5v in _A5_SAVED.items(): ...`) must stay **outside** the `else`,
at top level — it has to run even when A5 is disabled, because cell 25 always
takes the snapshot.

### 3b. Cell 26 — coerce the fat-suppression flag like the rest of the notebook

```python
# ADD near the top of the cell:
def _a5_flag(v):
    if isinstance(v, str):
        return v.strip().lower() in ('1', 'true', 't', 'yes', 'y')
    try:
        return bool(int(v))
    except Exception:
        return bool(v)

# OLD (inside build_study)
sub = rows[(rows.Anatomical_Plane == plane) & (rows.Fat_Suppression == fs)]

# NEW
sub = rows[(rows.Anatomical_Plane.astype(str) == plane)
           & (rows.Fat_Suppression.map(_a5_flag) == bool(fs))]
```

### 3c. Cell 28 — time budget on the decode loop (FIX 4 lives here too)

```python
# ADD before the inference loop (T0 is the primary pipeline's start time,
# same convention main_v52_e10 already uses):
_a5_wall_start = float(globals().get('T0', time.time()))
A5_DEADLINE = _a5_wall_start + 8.72 * 3600 - 55 * 60   # leave E10 its 45-min window + margin
A5_ABORTED = False

# OLD
    for c0 in range(0, len(studies), CHUNK):
        block = studies[c0:c0 + CHUNK]

# NEW
    for c0 in range(0, len(studies), CHUNK):
        if time.time() > A5_DEADLINE:
            print(f'a5: time budget exhausted at {done:,}/{len(studies):,}; '
                  f'aborting decode, remaining studies stay NaN')
            A5_ABORTED = True
            break
        block = studies[c0:c0 + CHUNK]
```

Aborted/undecoded studies remain NaN in `preds`; the rewritten cell 29 below
turns that into a per-row or full skip instead of a crash.

Also in cell 28, keep `A5_LABELS = list(LABELS)` and the `A5_PREDS = ...` line
where they are (before the restore loop) — they survive the restore because
they're new keys. `A5_W` moves to cell 29.

### 3d. Cell 29 — full replacement (never assert, never crash)

```python
import numpy as _np
import pandas as _pd

A5_W = 0.45
A5_MIN_COVERAGE = 0.98   # below this, skip the blend entirely rather than mix scales
A5_RECEIPT = {'applied': False, 'reason': None, 'coverage': None, 'weight': A5_W}

try:
    if not globals().get('A5_ENABLED', False) or 'A5_PREDS' not in globals():
        raise RuntimeError('A5 arm disabled or produced no predictions')
    if A5_W <= 0:
        raise RuntimeError('A5_W is 0')

    _a5_sub = _pd.read_csv('/kaggle/working/submission.csv',
                           dtype={'StudyInstanceUID': str})
    if _a5_sub.columns.tolist()[1:] != A5_LABELS:
        raise RuntimeError('submission schema drift')

    _a5_ours = _np.stack([
        A5_PREDS.get(_u, _np.full(len(A5_LABELS), _np.nan, _np.float32))
        for _u in _a5_sub['StudyInstanceUID'].astype(str)
    ])
    _row_ok = _np.isfinite(_a5_ours).all(axis=1)
    _cov = float(_row_ok.mean()) if len(_row_ok) else 0.0
    A5_RECEIPT['coverage'] = round(_cov, 4)
    if _cov < A5_MIN_COVERAGE:
        raise RuntimeError(f'coverage {_cov:.1%} below {A5_MIN_COVERAGE:.0%}')

    _base_rank = _a5_sub[A5_LABELS].rank(method='average', pct=True).to_numpy()
    _ours_rank = _pd.DataFrame(_a5_ours, columns=A5_LABELS)\
        .rank(method='average', pct=True).to_numpy()
    _blend = _base_rank.copy()
    _blend[_row_ok] = ((1.0 - A5_W) * _base_rank[_row_ok]
                       + A5_W * _ours_rank[_row_ok])
    if not _np.isfinite(_blend).all():
        raise RuntimeError('non-finite values after blend')

    _a5_sub[A5_LABELS] = _blend
    _a5_sub.to_csv('/kaggle/working/submission.csv', index=False)
    A5_RECEIPT.update(applied=True, reason='ok')
    print(f'a5: blended at {A5_W} over {int(_row_ok.sum()):,}/{len(_row_ok):,} studies')
except Exception as _e:
    A5_RECEIPT['reason'] = f'{type(_e).__name__}: {_e}'
    print(f'a5: SKIPPED ({A5_RECEIPT["reason"]}); submission untouched')
```

Notes:
- Per-row fallback (base ranks kept for the few NaN rows) is only allowed above
  98% coverage; below that the calibration mismatch between blended and
  unblended rows would distort cross-row ordering, so the arm steps aside whole.
- `A5_RECEIPT` is what the rewritten cell 31 picks up and logs.

---

## FIX 5 — not a patch: the dress rehearsal

No code change in the notebook. Before submitting, run once with:
1. `studies` padded to hidden-test scale (duplicate the visible studies ×N) to
   measure real decode throughput against `A5_DEADLINE`;
2. one study's DICOM directory emptied and one file truncated on purpose —
   cell 29 must print `a5: SKIPPED`/per-row fallback and cell 31 must still
   print `VALID_SUBMISSION`;
3. `test_series.csv` re-saved with `Fat_Suppression` as strings
   (`"True"/"False"`) — cell 26's `_a5_flag` must keep masks non-zero.

If all three pass, every arm now either improves the file or steps aside.
