"""Executions API: serves rule execution sessions from the audit log."""

from fastapi import APIRouter, HTTPException

import rules as rls
import services as svc

router = APIRouter(prefix="/executions", tags=["executions"])


@router.get("")
async def list_executions(domain: str) -> dict:
    """Return execution sessions for a domain, newest first.

    Args:
        domain: Domain key (e.g. "claim_appeal").

    Returns:
        JSON with domain and sessions list. Each session includes trace_id,
        timestamp, triggered rule IDs and detail, ordered evaluations (one entry
        per rule evaluated with outcome triggered/skipped_precondition/skipped_no_match),
        entity snapshots (claim and customer objects at execution time), output
        keys, and eligibility.
    """
    all_domains = {r.domain for r in rls.RuleRegistry().all()}
    if domain not in all_domains:
        raise HTTPException(status_code=404, detail=f"Domain '{domain}' not found")

    records = svc.RuleExecutionAuditService().list(domain)
    sessions = []
    for r in records:
        meta = r.get("metadata", {})
        triggered = r.get("triggered", [])
        outputs = r.get("outputs", {})
        sessions.append({
            "trace_id": meta.get("trace_id", ""),
            "executed_timestamp": meta.get("executed_timestamp", ""),
            "executed_by": meta.get("executed_by", ""),
            "triggered_ids": [rule["id"] for rule in triggered],
            "triggered_rules": [
                {
                    "id": rule["id"],
                    "reason": rule.get("reason", ""),
                    "subject": rule.get("subject", ""),
                    "attribute": rule.get("attribute", ""),
                    "operator": rule.get("operator", ""),
                    "threshold": rule.get("threshold", ""),
                    "input": rule.get("input", []),
                    "output": rule.get("output", []),
                }
                for rule in triggered
            ],
            "evaluations": r.get("evaluations", []),
            "entities": r.get("entities", {}),
            "output_keys": list(outputs.keys()),
            "eligible": "appeal.disqualified" not in outputs,
        })
    return {"domain": domain, "sessions": sessions}
