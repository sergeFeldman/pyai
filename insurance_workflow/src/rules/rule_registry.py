"""Rule registry backed by the generic KeyedRegistry."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field

import networkx as nx

import shared.core as shd_core
import shared.data as shd_data

import models as mdl
from .rule import Rule
from .rule_factory import RuleFactory
from .rule_graph_validator import RuleGraphFinding, RuleGraphValidationReport, RuleGraphValidator, RuleValidationSeverity

logger = logging.getLogger(__name__)

_FAILED_OUTCOMES: frozenset[str] = frozenset({"skipped_no_match", "skipped_precondition", "pruned"})


@dataclass
class RuleExecutionResult(shd_core.SerializableMixin):
    """Result of executing a rule domain through RuleRegistry.execute_async().

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
            triggered - preconditions met and condition matched;
            skipped_no_match - the rule was evaluated but did not match;
            pruned - a required internal input field was unreachable because all its producers 
                     failed or were themselves pruned; the rule was never evaluated;
            skipped_precondition - a required external field was absent from context;
            not_evaluated - the rule was in the ready queue when early exit fired and
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
        execute_async() uses a wave-based ready-queue model: all rules whose upstream
        producers have settled form a wave and are evaluated concurrently via
        asyncio.gather. Outputs are merged to context after the wave completes, then
        consumers are cascaded. Execution stops as soon as a terminal output appears
        in context after a wave.

    Audit trail:
        The registry is append-only; add() never replaces an existing rule.
        All versions of a rule coexist so history is fully preserved.
    """

    def __init__(self) -> None:
        super().__init__(Rule, key_field="domain")
        self._dags: dict[str, nx.DiGraph] = {}  # lazy cache, mirrors ConfigurableObjectFactory._objects
        self._excluded_rule_ids: set[str] = set()
        self._validation_report: RuleGraphValidationReport = RuleGraphValidationReport()
        self._domain_configs: dict[str, mdl.DomainConfig] = {}

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
    def load_from(cls, *file_paths: str, domain_config_path: str | None = None) -> RuleRegistry:
        """Create and populate a registry from one or more rule data files.

        Deserializes all rules from every file via RuleFactory, loads them into
        the registry in a single pass, then runs three validations over the active
        rule graph:

        1. Cycle detection: raises if any domain's active graph contains a cycle.
        2. Phantom consumer detection: rules that declare an input field no active
           rule produces are logged at ERROR level and stored in _excluded_rule_ids;
           they are filtered out of _get_effective() and never reach the DAG.
        3. Orphan producer detection: rules that produce a field no active rule
           consumes are logged at WARNING level.

        The full validation report is stored on the registry as _validation_report.

        Args:
            file_paths: One or more paths to JSON rule data files. All files are
                combined before loading so validation covers every domain.

        Returns:
            RuleRegistry populated with rules from all provided files.

        Raises:
            ValueError: If any domain's active rule graph contains a cycle.
        """
        factory = RuleFactory()
        all_rules = []
        for file_path in file_paths:
            storage = shd_data.JsonDataStorage(
                shd_data.JsonDataStorageConfig(model_class=Rule, key_field="id", file_path=file_path)
            )
            all_rules.extend(factory.from_dict(raw) for raw in storage.read_as_dicts())

        registry = cls()
        registry.load(all_rules)

        if domain_config_path:
            import json
            from pathlib import Path
            raw_configs = json.loads(Path(domain_config_path).read_text())
            registry._domain_configs = {
                dc["domain"]: mdl.DomainConfig.model_validate(dc) for dc in raw_configs
            }

        domains = {r.domain for r in registry.all()}
        # Cycle detection: a cyclic graph would cause topological sort to hang; 
        # raise before any execution.
        for domain in domains:
            if not nx.is_directed_acyclic_graph(registry.get_dag(domain)):
                raise ValueError(f"Cycle detected in rules for domain: {domain}")

        # Collect all active rules across domains for a single validation pass.
        all_effective = [r for d in domains for r in registry._get_effective(d)]

        # Suppresses orphan producer warnings for declared terminal fields.
        terminal_outputs_map = {
            domain: set(cfg.terminal_outputs)
            for domain, cfg in registry._domain_configs.items()
        }

        # Enables explicit-mode phantom detection; omit domains with no declaration
        # so the validator falls back to namespace-scoping for those domains.
        context_inputs_map = {
            domain: set(cfg.context_inputs)
            for domain, cfg in registry._domain_configs.items()
            if cfg.context_inputs
        }

        report = RuleGraphValidator().validate(
            all_effective,
            terminal_outputs=terminal_outputs_map,
            context_inputs=context_inputs_map or None,
        )

        for finding in report.findings:
            if finding.severity == RuleValidationSeverity.ERROR:
                logger.error(f"[{finding.kind}] {finding.domain}/{finding.rule_id} field='{finding.field}': {finding.message}")
            else:
                logger.warning(f"[{finding.kind}] {finding.domain}/{finding.rule_id} field='{finding.field}': {finding.message}")

        registry._excluded_rule_ids = {f.rule_id for f in report.errors}
        registry._validation_report = report
        registry._dags.clear()  # invalidate DAGs built during cycle detection; rebuild excludes phantom consumers

        return registry

    async def execute_async(self, domain: str, context: dict, 
                            trace_id: str = "", executed_by: str = "", 
                            group: str = "", entities: dict | None = None) -> RuleExecutionResult:
        """Execute all active rules for a domain against a shared context dict.

        Uses a wave-based ready-queue model (Kahn's variant): all rules whose upstream producers 
        have settled form a wave and are evaluated concurrently via asyncio.gather. Each coroutine 
        reads from context but does not write to it; outputs are merged to context after the full 
        wave completes. Consumers are then cascaded and execution stops as soon as a terminal 
        output appears in context after a wave.

        A rule appears in evaluations if and only if it entered the ready queue or was cascade-pruned. 
        Rules still waiting on a producer cut off by early exit never became ready and get no entry. 
        Of rules that do appear, each gets one of five outcomes:
          - triggered: preconditions met and condition matched; outputs written to context.
          - skipped_no_match: preconditions met but condition did not match.
          - pruned: a required internal input is absent and all its producers failed or
            were pruned; rule never evaluated.
          - skipped_precondition: a required external field is absent from context;
            rule never evaluated.
          - not_evaluated: rule was in the ready queue when early exit fired; never dequeued.

        Callers seed context with domain object fields using dot-notation keys
        (e.g. "claim.amount", "customer.tenure_years") before calling execute_async().
        Only outputs produced during execution are included in the result; all seeded caller fields are excluded.

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

        # Precomputed at build time; reading here avoids rebuilding field maps on every execute_async() call.
        producers: dict[str, list[str]] = G.graph["producers"]
        consumers: dict[str, list[str]] = G.graph["consumers"]
        terminal_outputs: set[str] = G.graph["terminal_outputs"]

        # Single record of every decided rule: rule_id -> outcome, in decision order.
        # Replaces separate triggered/failed/pruned collections; _is_prunable reads it for O(1) lookup.
        # skipped_precondition is treated as permanent failure because context is sealed at call
        # time; no external field appears mid-execution, so a missing input stays missing.
        decided: dict[str, str] = {}

        def _is_prunable(rule: Rule) -> bool:
            """A field is permanently unreachable when it is absent from context, has at least one
            producer rule in the graph (i.e. it is an internal field, not an external context input),
            and every one of those producers has already failed or been pruned. Pruning cascades:
            a rule pruned here may itself unblock further pruning of its own consumers via _cascade().

            Args:
                rule: The candidate rule to test.

            Returns:
                True if at least one required internal input field is permanently unreachable.
            """
            return any(
                field_name not in context
                and field_name in producers
                and all(decided.get(p) in _FAILED_OUTCOMES for p in producers[field_name])
                for field_name in rule.input
            )

        # Step 1: Initialise execution state.
        # pending_producers: a local copy of pending_producers attribute of provided 
        #   graph (upstream producer IDs for each rule); as producers settle, 
        #   their IDs are removed from the consumer's pending_producers set. 
        # queue: seeded with generation-0 rules - those whose pending_producers set is already
        #   empty at the start (root rules with no upstream producers), sorted by priority 
        #   descending with rule ID as tiebreaker for deterministic ordering.
        pending_producers: dict[str, set[str]] = {
            k: set(v) for k, v in G.graph["pending_producers"].items()
        }

        gen0_candidates = sorted(
            [rule_id for rule_id, s in pending_producers.items() if not s],
            key=lambda rule_id: (-G.nodes[rule_id]["rule"].priority, rule_id)
        )
        queue: deque[str] = deque(gen0_candidates)

        def _cascade(decided_id: str) -> None:
            """Uses an iterative stack (not recursion) to walk the consumer graph. For each
            consumer of decided_id, its entry in pending_producers is reduced by one. A consumer
            whose pending_producers entry is empty is immediately ready; it is either pruned
            (if _is_prunable returns True) or added to the ready queue. Pruned consumers are
            themselves cascaded so their own downstream consumers are settled without waiting
            for the next generation.

            Args:
                decided_id: Rule ID whose outcome has just been recorded (triggered, skipped,
                    or pruned); its consumers are now candidates for readiness.
            """
            stack = [decided_id]
            while stack:
                src = stack.pop()
                newly_ready = []
                for consumer_id in consumers.get(src, []):
                    pending_producers[consumer_id].discard(src)
                    if pending_producers[consumer_id]:
                        continue  # still waiting on other producers
                    rule = G.nodes[consumer_id]["rule"]
                    if rule.input and _is_prunable(rule):
                        decided[consumer_id] = "pruned"
                        stack.append(consumer_id)
                    else:
                        newly_ready.append(consumer_id)
                # Preserve priority ordering within each cascade generation.
                newly_ready.sort(key=lambda rule_id: (-G.nodes[rule_id]["rule"].priority, rule_id))
                queue.extend(newly_ready)

        async def _eval_rule(rule_id: str) -> tuple[str, str, dict | None]:
            """Evaluate one rule and return its outcome without writing to context.

            Context mutation and cascade happen in the outer loop after asyncio.gather
            completes the generation, so all rules in a generation read from the same pre-generation state.

            Args:
                rule_id: ID of the rule to evaluate.

            Returns:
                Tuple of (rule_id, outcome, outputs) where outputs is the rule's output
                dict on a match, or None for any non-triggered outcome.
            """
            rule = G.nodes[rule_id]["rule"]

            # Defensive prune check: _cascade() detects prunable consumers before they
            # are enqueued; this guard catches the edge case where a rule reaches the
            # queue in a prunable state despite that.
            if rule.input and _is_prunable(rule):
                return rule_id, "pruned", None

            if not rule.ready(context):
                return rule_id, "skipped_precondition", None

            matched, rule_outputs = rule.evaluate(context)
            if matched:
                return rule_id, "triggered", rule_outputs
            return rule_id, "skipped_no_match", None

        # Step 2: Execute generations until queue is empty or early exit fires.
        while queue:
            # Drain the entire queue into the current generation.
            gen_curr: list[str] = list(queue)
            queue.clear()

            # Evaluate all rules in the generation concurrently; none writes to context during evaluation.
            results = await asyncio.gather(*[_eval_rule(rule_id) for rule_id in gen_curr])

            # Apply all outputs before cascading so _is_prunable sees the complete post-generation context.
            # Sorted by priority descending so higher-priority outputs are written first; if two
            # rules in the same generation produce the same field the lower-priority rule wins (last write).
            # Same-field conflicts are a DAG design error caught by the graph linter in practice.
            results = sorted(results, key=lambda r: (-G.nodes[r[0]]["rule"].priority, r[0]))
            for rule_id, outcome, outputs in results:
                decided[rule_id] = outcome
                if outcome == "triggered" and outputs is not None:
                    context.update(outputs)

            # Cascade all decided rules now that context reflects the full generation.
            for rule_id, _, _ in results:
                _cascade(rule_id)

            # Early exit: terminal output produced; do not start the next generation.
            if terminal_outputs & context.keys():
                break

        # Step 3: Finalise results.
        #   Record not_evaluated for rules queued when early exit fired, derive triggered and
        #   evaluations from decided, strip caller-seeded keys, and return the execution result.
        for remaining_id in queue:
            decided[remaining_id] = "not_evaluated"

        triggered = [G.nodes[rid]["rule"] for rid, oc in decided.items() if oc == "triggered"]
        evaluations = [{"rule_id": rid, "outcome": oc} for rid, oc in decided.items()]

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

    def terminal_outputs(self, domain: str) -> set[str]:
        """Return the declared terminal output fields for a domain.

        Returns the set from mdl.DomainConfig if one was loaded for this domain,
        otherwise an empty set.

        Args:
            domain: Domain key, e.g. "claim_appeal".

        Returns:
            Set of terminal output field names, or empty set if domain has no config.
        """
        config = self._domain_configs.get(domain)
        return set(config.terminal_outputs) if config else set()

    def all(self) -> list[Rule]:
        """Return all rule versions across all domains, sorted by id then version descending.

        Sorted so all versions of the same rule are grouped together with the
        latest version first - consistent ordering for the output JSON file.

        Returns:
            Flat list of all Rule instances across all domains and versions.
        """
        rules = [r for items in self._registry.values() for r in items]
        return sorted(rules, key=lambda r: (r.id, -r.metadata.version))

    def _build_graph(self, rules: list[Rule]) -> nx.DiGraph:
        """Build a DiGraph from a rule list using input/output field overlap as edges.

        An edge A -> B is added for every field that appears in both A.output and
        B.input. External inputs (fields with no producer in the rule set) make
        their rule a root node with in-degree zero.

        Four structures are precomputed and stored on the graph for use by execute_async():
            G.graph["producers"]: field -> [rule_ids] that produce it.
            G.graph["consumers"]: rule_id -> [rule_ids] that consume any of its outputs.
            G.graph["terminal_outputs"]: fields produced but consumed by no rule;
                presence in context signals early exit after a wave.
            G.graph["pending_producers"]: rule_id -> frozenset of upstream producer IDs;
                execute_async() copies this per call to seed its mutable pending_producers tracking dict.

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

        # Store all structures on the graph so execute_async() reads them in O(1), avoiding per-call rebuilds.
        G.graph["producers"] = dict(producers)  # plain dict: prevents defaultdict __missing__ side effects in execute_async()
        G.graph["consumers"] = {k: list(v) for k, v in consumers.items()}
        G.graph["terminal_outputs"] = all_outputs - all_inputs  # fields with no consumer; triggers early exit after a wave
        # Per-rule upstream producer sets; execute_async() copies these per call instead of recomputing from the graph.
        G.graph["pending_producers"] = {
            rule.id: frozenset(p for f in rule.input for p in producers.get(f, []))
            for rule in rules
        }

        return G

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
        return [rule for rule in latest.values() if rule.is_effective and rule.id not in self._excluded_rule_ids]
