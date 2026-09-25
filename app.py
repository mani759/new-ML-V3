"""
MPLADS Risk Scoring Service v3 (FastAPI) - real eSAKSHI data.

Endpoints
  GET  /                health + model version
  GET  /version         model version, data snapshot date, validation summary
  POST /score           score ONE work instantly (live "Analyze a work" form). Read-only.
  POST /score/batch     score a list of works (same logic), for small batches / tests

Bulk scoring of all works is NOT done over HTTP: run `python pipeline.py score`
(see HANDOFF.md) and load output/scored_works.csv + output/mp_alerts.csv into the database.

Auth: set env ML_SERVICE_KEY and send header X-Service-Key. Unset = open.
"""
import json
import os
from typing import List, Optional

import joblib
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, model_validator

import risk_engine as RE

_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH = joblib.load(os.path.join(_DIR, 'artifacts', 'bench.pkl'))
MODELS = joblib.load(os.path.join(_DIR, 'artifacts', 'models.pkl'))
_REPORT = os.path.join(_DIR, 'output', 'validation_report.json')
REPORT = json.load(open(_REPORT)) if os.path.exists(_REPORT) else {}
_KEY = os.environ.get('ML_SERVICE_KEY')

app = FastAPI(title='MPLADS Risk Scoring Service', version=RE.MODEL_VERSION)


def check_key(x_service_key: Optional[str] = Header(default=None)):
    if _KEY and x_service_key != _KEY:
        raise HTTPException(status_code=401, detail='Invalid or missing X-Service-Key')


class WorkIn(BaseModel):
    house: str                              # 'LS' or 'RS'
    state: str
    ida: str                                # District Authority, exactly as in eSAKSHI
    mp_name: str
    # activity_type / work_description / recommended_amount may be null for real eSAKSHI works
    # (pending works named 'NA-...', works with no description, works sanctioned without a
    # recommended-list record). Nulls are scored exactly as the batch pipeline scores them.
    activity_type: Optional[str] = None     # one of the 115 eSAKSHI activity types, or null if not recorded
    work_description: Optional[str] = None
    recommended_amount: Optional[float] = None  # at least one of recommended_amount / sanction_amount is required
    recommendation_date: str                # YYYY-MM-DD
    work_id: Optional[int] = None           # omit for a new work
    constituency: Optional[str] = None
    # Omitted -> 'Normal/Others' (unchanged documented default for the live form).
    # Explicit null -> kept null all the way to the engine (batch treats it as 'NA').
    work_category: Optional[str] = 'Normal/Others'
    sanction_date: Optional[str] = None
    sanction_amount: Optional[float] = None
    stage: Optional[str] = None             # defaults to 'Pending for Sanction'
    in_recommended_list: bool = True
    in_sanctioned_list: bool = False
    is_completed: bool = False
    completion_date: Optional[str] = None
    actual_cost: Optional[float] = None
    total_paid: float = 0
    payment_count: int = 0
    vendor_count: int = 0
    last_payment_date: Optional[str] = None
    as_of: Optional[str] = None             # scoring date; omit = today

    @model_validator(mode='after')
    def _need_an_amount(self):
        # The engine scores amount = sanction_amount, else recommended_amount. With neither there is
        # nothing real to score, and no amount is invented.
        if self.recommended_amount is None and self.sanction_amount is None:
            raise ValueError('recommended_amount or sanction_amount is required')
        return self


class BatchIn(BaseModel):
    works: List[WorkIn]
    as_of: Optional[str] = None


@app.get('/')
def health():
    return {'status': 'ok', 'service': 'mplads-risk-scoring', 'model_version': RE.MODEL_VERSION}


@app.get('/version')
def version():
    return {'model_version': RE.MODEL_VERSION, 'data_snapshot': BENCH.get('as_of'),
            'validation': {k: REPORT.get(k) for k in ('planted_anomaly_validation', 'delay_model_test')}}


def _one(w: WorkIn, as_of=None):
    d = w.model_dump()
    # Only fall back to a REAL recommended amount; with recommended_amount=null the sanction amount stays as given
    # (overwriting it with None made the scored amount NaN and the response un-serialisable).
    if d.get('in_sanctioned_list') and not d.get('sanction_amount') and d.get('recommended_amount') is not None:
        d['sanction_amount'] = d['recommended_amount']
    return RE.score_one(d, BENCH, MODELS, as_of=as_of or d.pop('as_of', None))


@app.post('/score', dependencies=[Depends(check_key)])
def score(work: WorkIn):
    try:
        return _one(work)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post('/score/batch', dependencies=[Depends(check_key)])
def score_batch(body: BatchIn):
    out = []
    for w in body.works:
        try:
            out.append(_one(w, body.as_of))
        except Exception as e:
            out.append({'work_id': w.work_id, 'error': str(e)})
    return {'results': out, 'count': len(out)}
