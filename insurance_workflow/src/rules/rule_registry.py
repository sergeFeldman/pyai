"""Rule registry backed by the generic KeyedRegistry."""

from __future__ import annotations

import logging
from collections import defaultdict, deque
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
        evaluations: Ordered record of rules that entered the ready queue or were cascade-pruned.
            Rules still waiting on a producer cut off by early exit never became ready and get
            no entry. Each entry is {"rule_id": str, "outcome": str} where outcome is one of:
            "triggered"|"skipped_no_match"|"pruned"|"skipped_precondition"|"not_evaluated".
            triggered means preconditions met and condition matched;
            skipped_no_match means the rule was evaluated but did not match;
            pruned means a required internal input field was unreachable because all its
            producers failed or were themselves pruned; the rule was never evaluated;
            skipped_precondition means a required external field was absent from context;
            not_evaluated means the rule was in the ready queue when early exit fired and
            was never dequeued.
        entities: Snapshot of domain objects at execution time, keyed by entity
            type e.g. {"claim": {...}, "customer": {...}}.
        outputs: Intermediate and terminal outputs produced during execution,
            e.g. {"appeal.risk_flagged": True, "appeal.disqualified": True}.
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
        get_effective() returns the latest active version of each rule in a domain
        in topological execution order; rules that produce outputs consumed by
        other rules are always returned before their dependents.

    DAG:
        get_dag() returns a cached nx.DiGraph for a domain. Nodes are rule IDs
        with the Rule object attached; edges connect producer rules to consumer
        rules via shared input/output field names. All NetworkX graph queries
        (ancestors, descendants, topological_sort, etc.) are available on the
        returned graph directly.

    Execution:
        execute() uses a ready-queue model: a rule enters the queue only when every
        upstream producer has been decided. When a rule is decided, its consumers are
        cascaded immediately; any consumer whose last producer just settled is either
        cascade-pruned or enqueued. Execution stops as soon as a terminal output
        (a field produced by a rule but consumed by none) appears in context.

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

    def _get_effective(self, domain: str, group: str = "") -> list[Rule]:
        """Resolve deduplicated active rules without building a graph.

        Selects the latest version per rule id, applies the optional group filter, 
        and returns rules that pass is_effective. Does not perform any graph construction.

        Args:
            domain: Domain to retrieve rules for.
            group: Optional execution cluster filter within the domain.

        Returns:
            Active Rule instances, one per id at the latest version, unordered.
        """
        rules = self.get_by_key(domain)
        if group:
            rules = [r for r in rules if r.group == group]

        latest: dict[str, Rule] = {}
        for rule in rules:
            prev = latest.get(rule.id)
            if prev is None or rule.metadata.version > prev.metadata.version:
                latest[rule.id] = rule
        return [rule for rule in latest.values() if rule.is_effective]

    def _build_graph(self, rules: list[Rule]) -> nx.DiGraph:
        """Build a DiGraph from a rule list using input/output field overlap as edges.

        An edge A -> B is added for every field that appears in both A.output and
        B.input. External inputs (fields with no producer in the rule set) make
        their rule a root node with in-degree zero.

        Four structures are precomputed and stored on the graph for use by execute():
            G.graph["producers"]: field -> [rule_ids] that produce it.
            G.graph["consumers"]: rule_id -> [rule_ids] that consume any of its outputs.
            G.graph["terminal_outputs"]: fields produced but consumed by no rule;
                presence in context signals early exit during execution.
            G.graph["undecided_template"]: rule_id -> frozenset of upstream producer IDs;
                execute() copies this per call to seed its mutable undecided tracking dict.

        Args:
            rules: Rule instances to include as nodes.

        Returns:
            nx.DiGraph with nodes keyed by rule.id (Rule object on each node as
            node data "rule") and edges annotated with all shared field names as
            a list under the "fields" key.
        """
        G = nx.DiGraph()
        producers: defaultdict[str, list[str]] = defaultdict(list)
        all_outputs: set[str] = set()

        # Register nodes and build producers; 
        # must complete before edges and consumers can be derived.
        for rule in rules:
            G.add_node(rule.id, rule=rule)
            for field_name in rule.output:
                producers[field_name].append(rule.id)
                all_outputs.add(field_name)

        # Connect producers to consumers via shared fields; 
        # accumulate all_inputs for terminal output detection.
        # (two passes are required because producers must be fully populated 
        # before any edges can be added.)
        consumers: defaultdict[str, set[str]] = defaultdict(set)
        all_inputs: set[str] = set()
        for rule in rules:
            for field_name in rule.input:
                all_inputs.add(field_name)
                # One edge per producer-consumer pair; 
                # append the shared field if the edge already exists.
                for producer_id in producers.get(field_name, []):
                    consumers[producer_id].add(rule.id)
                    edge_data = G.get_edge_data(producer_id, rule.id)
                    if edge_data is not None:
                        edge_data["fields"].append(field_name)
                    else:
                        G.add_edge(producer_id, rule.id, fields=[field_name])

        # Store all structures on the graph so execute() reads them in O(1), avoiding per-call rebuilds.
        G.graph["producers"] = dict(producers)  # plain dict: prevents defaultdict __missing__ side effects in execute()
        G.graph["consumers"] = {k: list(v) for k, v in consumers.items()}
        G.graph["terminal_outputs"] = all_outputs - all_inputs  # fields with no consumer; triggers early exit in execute()
        # Per-rule upstream producer sets; execute() copies these per call instead of recomputing from the graph.
        G.graph["undecided_template"] = {
            rule.id: frozenset(p for f in rule.input for p in producers.get(f, []))
            for rule in rules
        }

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
            self._dags[key] = self._build_graph(self._get_effective(domain, group))
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

    def get_effective(self, domain: str, group: str = "") -> list[Rule]:
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

        Uses a ready-queue model (Kahn's variant): a rule enters the queue only when
        every upstream producer rule has been decided. When a rule is decided, its
        consumers are cascaded immediately; any consumer whose last producer just
        settled is either cascade-pruned or enqueued. Execution stops as soon as a
        terminal output appears in context.

        A rule appears in evaluations if and only if it entered the ready queue or was
        cascade-pruned. Rules still waiting on a producer cut off by early exit never became
        ready and get no entry. Of rules that do appear, each gets one of five outcomes:
          - triggered: preconditions met and condition matched; outputs written to context.
          - skipped_no_match: preconditions met but condition did not match.
          - pruned: a required internal input is absent and all its producers failed or
            were pruned; rule never evaluated.
          - skipped_precondition: a required external field is absent from context;
            rule never evaluated.
          - not_evaluated: rule was in the ready queue when early exit fired; never dequeued.

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

        # Precomputed at build time; reading here avoids rebuilding field maps on every execute() call.
        producers: dict[str, list[str]] = G.graph["producers"]
        consumers: dict[str, list[str]] = G.graph["consumers"]
        terminal_outputs: set[str] = G.graph["terminal_outputs"]

        # Track decided rules so _is_prunable can test whether inputs are permanently unreachable.
        failed: set[str] = set()
        pruned: set[str] = set()

        def _is_prunable(rule: Rule) -> bool:
            # True if any required internal input is absent and all its producers failed/pruned.
            return any(
                field_name not in context
                and field_name in producers
                and all(p in failed or p in pruned for p in producers[field_name])
                for field_name in rule.input
            )

        # Per-rule mutable copy of upstream producer IDs not yet decided; empties as producers settle.
        undecided: dict[str, set[str]] = {
            k: set(v) for k, v in G.graph["undecided_template"].items()
        }

        # Seed the queue with rules that have no internal inputs (generation 0).
        initial = sorted(
            [rid for rid, s in undecided.items() if not s],
            key=lambda rid: (-G.nodes[rid]["rule"].priority, rid)
        )
        queue: deque[str] = deque(initial)

        def _cascade(decided_id: str) -> None:
            # Iterative cascade: after a rule is decided, immediately settle consumers
            # whose last producer just resolved. Prunable consumers are cascade-pruned
            # without entering the queue; others are enqueued in priority order.
            stack = [decided_id]
            while stack:
                src = stack.pop()
                newly_ready = []
                for consumer_id in consumers.get(src, []):
                    undecided[consumer_id].discard(src)
                    if undecided[consumer_id]:
                        continue  # still waiting on other producers
                    rule = G.nodes[consumer_id]["rule"]
                    if _is_prunable(rule):
                        pruned.add(consumer_id)
                        evaluations.append({"rule_id": consumer_id, "outcome": "pruned"})
                        stack.append(consumer_id)
                    else:
                        newly_ready.append(consumer_id)
                # Preserve priority ordering within each cascade wave.
                newly_ready.sort(key=lambda rid: (-G.nodes[rid]["rule"].priority, rid))
                queue.extend(newly_ready)

        # Evaluate each ready rule in priority order; a rule enters only after all its producers have settled,
        # so its inputs are final by the time it runs.
        while queue:
            rule_id = queue.popleft()
            rule = G.nodes[rule_id]["rule"]

            # Defensive prune check: _cascade() detects prunable consumers and records them
            # before they are enqueued; this guard catches the edge case where a rule
            # reaches the queue in a prunable state despite that.
            if _is_prunable(rule):
                pruned.add(rule_id)
                evaluations.append({"rule_id": rule_id, "outcome": "pruned"})
                _cascade(rule_id)  # pruned is a decision; consumers waiting on this rule must be settled
                continue

            # Precondition check: all declared inputs present in context?
            if not rule.ready(context):
                failed.add(rule_id)
                evaluations.append({"rule_id": rule_id, "outcome": "skipped_precondition"})
                _cascade(rule_id)  # failure is a decision; settle consumers so they are not left waiting
                continue

            # Evaluate and propagate outputs to context.
            matched, rule_outputs = rule.evaluate(context)
            if matched:
                context.update(rule_outputs)
                triggered.append(rule)
                evaluations.append({"rule_id": rule_id, "outcome": "triggered"})
            else:
                failed.add(rule_id)
                evaluations.append({"rule_id": rule_id, "outcome": "skipped_no_match"})

            _cascade(rule_id)  # rule evaluated; settle downstream consumers regardless of match outcome

            # Early exit: terminal output produced; remaining rules cannot change the result.
            if terminal_outputs & context.keys():
                break

        # Record rules still in the queue when early exit fired; they were ready but never reached.
        for remaining_id in queue:
            evaluations.append({"rule_id": remaining_id, "outcome": "not_evaluated"})

        # Strip caller-seeded keys; the result contains only fields written during this execution.
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
