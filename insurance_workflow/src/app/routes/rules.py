"""Rules data API: serves domain list and per-domain DAG as JSON."""

from fastapi import APIRouter, HTTPException

import rules as rls

router = APIRouter(prefix="/rules", tags=["rules"])


@router.get("/history/{domain}/{rule_id}")
async def get_rule_history(domain: str, rule_id: str) -> dict:
    """Return all versions of a rule, newest first.

    Args:
        domain: Domain key (e.g. "claim_appeal").
        rule_id: Rule identifier.

    Returns:
        JSON with versions list (all versions, newest first) and a diff list
        mapping each version to the fields that changed relative to the next
        older version.

    Raises:
        HTTPException 404: rule not found in the domain.
    """
    registry = rls.RuleRegistry()
    versions = sorted(
        [r for r in registry.get_by_key(domain) if r.id == rule_id],
        key=lambda r: r.metadata.version,
        reverse=True,
    )
    if not versions:
        raise HTTPException(status_code=404, detail=f"Rule '{rule_id}' not found in domain '{domain}'")

    dicts = [r.to_dict() for r in versions]

    # compute per-version diff against the previous (older) version
    diffs: list[list[str]] = []
    for i, d in enumerate(dicts):
        if i == len(dicts) - 1:
            diffs.append([])          # oldest version, no prior to diff against
            continue
        older = dicts[i + 1]
        changed = [
            k for k in d
            if k not in ("metadata",) and d.get(k) != older.get(k)
        ]
        diffs.append(changed)

    return {
        "domain":  domain,
        "rule_id": rule_id,
        "versions": [{"rule": d, "changed_fields": diff} for d, diff in zip(dicts, diffs)],
    }


@router.get("/domains")
async def get_domains() -> dict:
    """Return all domains currently loaded in the rule registry."""
    registry = rls.RuleRegistry()
    domains = sorted({r.domain for r in registry.all()})
    return {"domains": domains}


@router.get("/dag/{domain}")
async def get_dag(domain: str, group: str = "",
                  replace_with_new: bool = False) -> dict:
    """Return DAG nodes and edges for a domain, suitable for vis-network rendering.

    Args:
        domain: Domain key (e.g. "claim_appeal").
        group: Optional group filter within the domain.
        replace_with_new: If True, bypasses the cache and rebuilds the DAG from
            the current registry contents before returning.

    Returns:
        JSON with node_count, edge_count, nodes (rule dicts), and edges
        ({from, to, field}) for every producer→consumer relationship in the DAG.

    Raises:
        HTTPException 404: domain not found in the registry.
    """
    registry = rls.RuleRegistry()
    all_domains = {r.domain for r in registry.all()}
    if domain not in all_domains:
        raise HTTPException(status_code=404, detail=f"Domain '{domain}' not found")

    G = registry.get_dag(domain, group, replace_with_new=replace_with_new)

    ordered = registry.get_active(domain, group)
    nodes = [r.to_dict() for r in ordered if r.id in G.nodes]
    edges = [
        {"from": u, "to": v, "fields": data.get("fields", [])}
        for u, v, data in G.edges(data=True)
    ]

    return {
        "domain": domain,
        "group": group,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
    }
