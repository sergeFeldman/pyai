"""Rule registry backed by the generic KeyedRegistry."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import networkx as nx

import shared.core as shd_core
import shared.data as shd_data

from .rule import Rule
from .rule_factory import RuleFactory

logger = logging.getLogger(__name__)


@dataclass
class RuleExecutionResult(shd_core.SerializableMixin):
    """Result of executing a rule domain through RuleRegistry.execute().

    Carries execution provenance (metadata), which rules triggered,
    the full ordered evaluation log (evaluations), entity snapshots captured
    at execution time (entities), and the outputs produced (outputs).

    Attributes:
        metadata: Trace and audit metadata (trace ID, executor, timestamp).
        domain: The rule domain that was executed, e.g. "claim_appeal".
        triggered: Rules that matched and triggered, in topological execution order.
        evaluations: Ordered record of every rule visited during execution. Each entry is
            {"rule_id": str, "outcome": "triggered"|"skipped_no_match"|"pruned"|"skipped_precondition"}.
            triggered means preconditions met and condition matched;
            skipped_no_match means the rule was evaluated but did not match;
            pruned means a required internal input field was unreachable because all its
            producers failed or were themselves pruned; the rule was never evaluated;
            skipped_precondition means a required external field was absent from context.
        entities: Snapshot of domain objects at execution time, keyed by entity
            type e.g. {"claim": {...}, "customer": {...}}.
        outputs: Intermediate and terminal outputs produced during execution,
            e.g. {"appeal.fraud_flagged": True, "appeal.disqualified": True}.
    """

    metadata: shd_core.ExecutionMetadata
    domain: str
    triggered: list[Rule]
    evaluations: list[dict] = field(default_factory=list)
    entities: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)


class RuleRegistry(shd_core.KeyedRegistry[Rule], metaclass=shd_core.Singleton):
    """Registry for all rule versions, keyed by domain.

    Extends KeyedRegistry with rule-specific queries and a DAG layer built from
    the input/output field declarations on each Rule.

    Storage:
        _registry (from KeyedRegistry): all rules, all versions, all domains.
        _dags: lazy cache of nx.DiGraph per domain (or domain:group), built on
            first get_dag() call and cleared whenever load() is called.

    Rule resolution:
        get_latest() resolves the highest-versioned rule for a given id.
        get_active() returns the latest active version of each rule in a domain
        in topological execution order; rules that produce outputs consumed by
        other rules are always returned before their dependents.

    DAG:
        get_dag() returns a cached nx.DiGraph for a domain. Nodes are rule IDs
        with the Rule object attached; edges connect producer rules to consumer
        rules via shared input/output field names. All NetworkX graph queries
        (ancestors, descendants, topological_sort, etc.) are available on the
        returned graph directly.

    Execution:
        execute() walks the DAG in topological generation order. Before evaluating
        each rule it checks whether all producers of its required internal inputs
        have failed or been pruned. If so, the rule is pruned without calling
        ready() or evaluate(), making the traversal true DAG execution rather than
        a linear sweep. Rules that pass the pruning check are evaluated via
        rule.ready()/rule.evaluate(); outputs are propagated to context, enabling
        downstream rules. Returns a RuleExecutionResult with provenance, triggered
        rules, full evaluation log (including pruned entries), and outputs.

    Audit trail:
        The registry is append-only; add() never replaces an existing rule.
        All versions of a rule coexist so history is fully preserved.
    """

    def __init__(self) -> None:
        super().__init__(Rule, key_field="domain")
        self._dags: dict[str, nx.DiGraph] = {}  # lazy cache, mirrors ConfigurableObjectFactory._objects

    def load(self, items: list[Rule]) -> None:
        """Replace registry contents and invalidate the DAG cache.

        Overrides KeyedRegistry.load() to clear _dags on every reload so
        callers never receive a stale graph after rules change.

        Args:
            items: Rule instances to load into the registry.
        """
        super().load(items)
        self._dags.clear()

    @classmethod
    def load_from(cls, *file_paths: str) -> RuleRegistry:
        """Create and populate a registry from one or more rule data files.

        Deserializes all rules from every file via RuleFactory, loads them into
        the registry in a single pass, then validates that every domain's active
        rule graph is acyclic. Raises at startup before any request is served if
        a cycle is found.

        Args:
            file_paths: One or more paths to JSON rule data files. All files are
                combined before loading so cycle detection covers every domain.

        Returns:
            RuleRegistry populated with rules from all provided files.

        Raises:
            ValueError: If any domain's active rule graph contains a cycle.
        """
        all_rules = []
        for file_path in file_paths:
            storage = shd_data.JsonDataStorage(
                shd_data.JsonDataStorageConfig(model_class=Rule, key_field="id", file_path=file_path)
            )
            all_rules.extend(RuleFactory().from_dict(raw) for raw in storage.read_as_dicts())

        registry = cls()
        registry.load(all_rules)

        for domain in {r.domain for r in registry.all()}:
            if not nx.is_directed_acyclic_graph(registry.get_dag(domain)):
                raise ValueError(f"Cycle detected in rules for domain: {domain}")

        return registry

    def _get_active_rules(self, domain: str, group: str = "") -> list[Rule]:
        """Resolve deduplicated active rules without building a graph.

        Internal helper shared by get_dag(). Selects the latest version per
        rule id, applies the optional group filter, and returns rules that
        pass is_active. Does not perform any graph construction.

        Args:
            domain: Domain to retrieve rules for.
            group: Optional execution cluster filter within the domain.

        Returns:
            Active Rule instances, one per id at the latest version, unordered.
        """
        rules = self.get_by_key(domain)
        if group:
            rules = [r for r in rules if r.group == group]

        by_id: dict[str, Rule] = {}
        for r in rules:
            if r.id not in by_id or r.metadata.version > by_id[r.id].metadata.version:
                by_id[r.id] = r

        return [r for r in by_id.values() if r.is_active]

    def _build_graph(self, rules: list[Rule]) -> nx.DiGraph:
        """Build a DiGraph from a rule list using input/output field overlap as edges.

        An edge A → B is added for every field that appears in both A.output and
        B.input. External inputs (fields with no producer in the rule set) make
        their rule a root node with in-degree zero.

        Args:
            rules: Rule instances to include as nodes.

        Returns:
            nx.DiGraph with nodes keyed by rule.id (Rule object on each node as
            node data "rule") and edges annotated with all shared field names as
            a list under the "fields" key.
        """
        G = nx.DiGraph()
        producers: dict[str, list[str]] = {}

        for rule in rules:
            G.add_node(rule.id, rule=rule)
            for field in rule.output:
                producers.setdefault(field, []).append(rule.id)

        edge_fields: dict[tuple[str, str], list[str]] = {}
        for rule in rules:
            for field in rule.input:
                for producer_id in producers.get(field, []):
                    edge_fields.setdefault((producer_id, rule.id), []).append(field)

        for (src, dst), fields in edge_fields.items():
            G.add_edge(src, dst, fields=fields)

        return G

    def get_dag(self, domain: str, group: str = "",
                replace_with_new: bool = False) -> nx.DiGraph:
        """Return the cached DiGraph for the domain, building it on first access.

        Follows ConfigurableObjectFactory.get_obj() pattern: lazy build on cache
        miss, replace_with_new bypasses the cache and forces a rebuild, and
        logger.info() is emitted on both hit and miss.

        The returned graph exposes all NetworkX queries directly:
            nx.topological_sort(G)
            nx.ancestors(G, rule_id)
            nx.descendants(G, rule_id)
            nx.is_directed_acyclic_graph(G)
            G.nodes[rule_id]["rule"]       # Rule object

        Args:
            domain: Domain to retrieve the graph for.
            group: Optional execution cluster filter within the domain.
            replace_with_new: If True, rebuilds and re-caches the graph.

        Returns:
            nx.DiGraph of active rules for the domain.
        """
        key = f"{domain}:{group}" if group else domain
        if key not in self._dags or replace_with_new:
            self._dags[key] = self._build_graph(self._get_active_rules(domain, group))
            logger.info(f"DAG built for domain: {key}")
        else:
            logger.info(f"DAG returned from cache for domain: {key}")
        return self._dags[key]

    def get_latest(self, id: str, domain: str) -> Rule | None:
        """Return the highest-versioned rule matching id within the given domain.

        Args:
            id: Rule identifier.
            domain: Domain to search within.

        Returns:
            Rule with the highest metadata.version, or None if not found.
        """
        matches = [r for r in self.get_by_key(domain) if r.id == id]
        return max(matches, key=lambda r: r.metadata.version) if matches else None

    def get_active(self, domain: str, group: str = "") -> list[Rule]:
        """Return active rules for a domain in topological execution order.

        Delegates to get_dag() for the cached graph, then returns rules in
        topological sort order so that every rule producing an output consumed
        by another rule is guaranteed to appear before its dependents. Within
        the same topological level, rules are sorted by priority descending
        (higher value executes first), with rule id as a secondary tiebreaker
        to guarantee fully deterministic ordering.

        Args:
            domain: Domain to retrieve rules for.
            group: Optional execution cluster filter within the domain.

        Returns:
            Active Rule instances in topological order, one per id at the
            latest version.
        """
        G = self.get_dag(domain, group)
        result: list[Rule] = []
        for generation in nx.topological_generations(G):
            rules = [G.nodes[rid]["rule"] for rid in generation]
            rules.sort(key=lambda r: (-r.priority, r.id))
            result.extend(rules)
        return result

    def execute(self, domain: str, context: dict,
                trace_id: str = "", executed_by: str = "",
                group: str = "", entities: dict | None = None) -> RuleExecutionResult:
        """Execute all active rules for a domain against a shared context dict.

        Walks the DAG in topological generation order (same traversal as
        get_active()). Within each generation, rules are evaluated in priority
        descending order. For each rule:
          1. rule.ready(context) checks all declared input preconditions.
          2. rule.evaluate(context) tests the condition and returns outputs to write.
          3. On match, outputs are merged into context, enabling downstream rules.

        Every rule is recorded in evaluations with an outcome:
          - triggered: preconditions met and condition matched.
          - skipped_no_match: preconditions met but condition did not match.
          - pruned: a required internal input field was unreachable because all its
            producers either failed or were themselves pruned; rule never evaluated.
          - skipped_precondition: a required external field was absent from context
            (caller did not seed it); rule never evaluated.

        Callers seed context with domain object fields using dot-notation keys
        (e.g. "claim.amount", "customer.tenure_years") before calling execute().
        Only outputs produced during execution are included in the result;
        all seeded caller fields are excluded.

        Args:
            domain: Rule domain to execute, e.g. "claim_appeal".
            context: Mutable dict seeded with subject field values. Modified
                in place as rules trigger and write their outputs.
            trace_id: Workflow request trace ID for the result metadata.
            executed_by: Agent or process triggering execution, e.g. "claim_appeal_agent".
            group: Optional execution cluster filter within the domain.
            entities: Snapshot of domain objects at execution time, e.g.
                {"claim": claim.to_dict(), "customer": customer.to_dict()}.

        Returns:
            RuleExecutionResult with metadata, triggered rules, evaluations,
            entities, and outputs.
        """
        G = self.get_dag(domain, group)
        seeded_keys = set(context.keys())
        triggered: list[Rule] = []
        evaluations: list[dict] = []

        # Map each output field to the rule(s) that produce it, for pruning.
        producers: dict[str, list[str]] = {}
        for node_id in G.nodes:
            for f in G.nodes[node_id]["rule"].output:
                producers.setdefault(f, []).append(node_id)

        failed: set[str] = set()   # rules that did not produce output
        pruned: set[str] = set()   # rules whose required inputs are unreachable

        for generation in nx.topological_generations(G):
            rules = [G.nodes[rid]["rule"] for rid in generation]
            rules.sort(key=lambda r: (-r.priority, r.id))
            for rule in rules:
                # Prune if every producer of any required internal input has failed or been pruned.
                if any(
                    f in producers and all(p in failed or p in pruned for p in producers[f])
                    for f in rule.input
                ):
                    pruned.add(rule.id)
                    evaluations.append({"rule_id": rule.id, "outcome": "pruned"})
                    continue

                if not rule.ready(context):
                    failed.add(rule.id)
                    evaluations.append({"rule_id": rule.id, "outcome": "skipped_precondition"})
                    continue

                matched, rule_outputs = rule.evaluate(context)
                if matched:
                    context.update(rule_outputs)
                    triggered.append(rule)
                    evaluations.append({"rule_id": rule.id, "outcome": "triggered"})
                else:
                    failed.add(rule.id)
                    evaluations.append({"rule_id": rule.id, "outcome": "skipped_no_match"})

        outputs = {k: v for k, v in context.items() if k not in seeded_keys}
        return RuleExecutionResult(
            metadata=shd_core.ExecutionMetadata(trace_id=trace_id, executed_by=executed_by),
            domain=domain,
            triggered=triggered,
            evaluations=evaluations,
            entities=entities or {},
            outputs=outputs,
        )

    def all(self) -> list[Rule]:
        """Return all rule versions across all domains, sorted by id then version descending.

        Sorted so all versions of the same rule are grouped together with the
        latest version first - consistent ordering for the output JSON file.

        Returns:
            Flat list of all Rule instances across all domains and versions.
        """
        rules = [r for items in self._registry.values() for r in items]
        return sorted(rules, key=lambda r: (r.id, -r.metadata.version))
