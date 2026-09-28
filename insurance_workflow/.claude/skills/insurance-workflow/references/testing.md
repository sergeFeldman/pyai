# Test Conventions

Read this before writing or changing tests.

## Layout and running

- Tests live in `tests/`, mirroring `src/` packages (`tests/rules/`, `tests/agents/`). Each test package has an empty `__init__.py`.
- Run from the project root with `python -m pytest tests/ -q`. The root `conftest.py` puts `src` and `../shared/src` on `sys.path`, so tests import packages directly (`from rules import RuleRegistry`, `import models as mdl`).
- Start each test module with a docstring naming what it covers.

## Structure

- Group tests in one `Test<Behavior>` class per behavior or method (`TestDecisionRuleMatches`, `TestExecute`). Name tests `test_<expected_behavior>` (`test_bool_false_string_coerced_correctly`).
- Build inputs with module-level helpers that take overrides: `_ts(delta_days)` for ISO timestamps relative to now, `_rule(...)` or `_drule(...)` for rules, `_claim(**kwargs)` and `_customer(**kwargs)` that merge `defaults | kwargs`.
- Generate rule effective windows relative to now (`effective_from=_ts(-1), effective_to=_ts(365)`), never as fixed dates, so tests do not expire.
- When a test compares two objects for equality (for example `is_changed`), compute each timestamp once and pass the same value to both. Separate `_ts()` calls can straddle a clock tick and make the test flaky; helper defaults should come from module-level constants (`_FROM`, `_TO` in `tests/rules/test_rule.py`).
- Use synthetic ids (`claim_1`, `cust_1`, `shop_1`); never copy real records into tests.

## Singletons

Singleton instances persist for the whole process. Any test that creates or uses a singleton must reset it afterwards with an autouse fixture:

```python
@pytest.fixture(autouse=True)
def reset_singletons():
    yield
    Singleton._instances.pop(RuleRegistry, None)
    Singleton._instances.pop(RuleFactory, None)
```

Pop every singleton the module touches (`RuleRegistry`, `RuleFactory`, `RuleExecutionAuditService`, `AuditService`, `WorkflowOrchestrator`, factories).

## Audit isolation

`tests/conftest.py` has an autouse fixture that constructs `RuleExecutionAuditService` with `data/test/audit/rule_executions.jsonl` before each test. Because a singleton keeps its first constructor arguments, tests must never construct it with the default path. Tests that assert on audit records clear that file in an autouse fixture first and read records back with `RuleExecutionAuditService().list(domain)`. `data/test/` is git-ignored.

## What to cover

For each behavior change, add at least one normal case and the edge cases that matter: missing inputs, not-found entities, inactive or expired rules, bool and Enum coercion, and ordering ties.
