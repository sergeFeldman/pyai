#!/usr/bin/env python3
"""Audit Claude Code project skills and their routing from CLAUDE.md.

Checks skill layout, frontmatter, route targets, reachability, cycles, and
relative links. Exits 0 when no errors are found and 1 otherwise; warnings
never fail the run. Uses only the Python standard library (PyYAML is used for
frontmatter when available, with a simple fallback parser otherwise).

Default project root: four levels above this file, i.e. the folder that
contains .claude/skills/skill-authoring/scripts/audit_skills.py.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

KNOWN_FIELDS = {
    "name", "description", "when_to_use", "argument-hint", "arguments",
    "disable-model-invocation", "user-invocable", "allowed-tools",
    "disallowed-tools", "model", "effort", "context", "agent", "background",
    "hooks", "paths", "shell", "metadata", "license", "compatibility",
}
LISTING_LIMIT = 1536
MAX_LINES = 500
MIN_ACTION_WORDS = 3

ROUTE_RE = re.compile(r"\bUse \[([^\]]+)\]\(([^)\s]+)\)\s+to\s+(\S.*)")
LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)\)")
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
INLINE_CODE_RE = re.compile(r"`[^`]*`")
NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
ROOT = "CLAUDE.md"


def norm(path: Path) -> Path:
    """Absolute, normalized path that keeps symlinks (Claude Code allows symlinked skill folders)."""
    return Path(os.path.normpath(os.path.abspath(path)))


class Report:
    def __init__(self, root: Path):
        self.root = root
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def _loc(self, path: Path, line: int | None) -> str:
        try:
            rel = norm(path).relative_to(self.root)
        except ValueError:
            rel = path
        return f"{rel}:{line}" if line else str(rel)

    def error(self, path: Path, line: int | None, msg: str) -> None:
        self.errors.append(f"ERROR    {self._loc(path, line)}  {msg}")

    def warn(self, path: Path, line: int | None, msg: str) -> None:
        self.warnings.append(f"WARNING  {self._loc(path, line)}  {msg}")


def prose_lines(text: str):
    """Yield (line_number, text) for body lines outside frontmatter and fenced code,
    with inline code removed."""
    _, body_start = split_frontmatter(text)
    fence = None
    for number, line in enumerate(text.splitlines(), start=1):
        if number <= body_start:
            continue
        match = FENCE_RE.match(line)
        if match:
            marker = match.group(1)[0]
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None
            continue
        if fence is None:
            yield number, INLINE_CODE_RE.sub("", line)


def split_frontmatter(text: str):
    """Return (frontmatter_text, body_start_line) or (None, 0) if absent."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None, 0
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            return "\n".join(lines[1:index]), index + 1
    return None, 0


def parse_frontmatter(raw: str) -> dict:
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(raw) or {}
        if not isinstance(data, dict):
            raise ValueError("frontmatter is not a mapping")
        return data
    except ImportError:
        pass
    data: dict = {}
    key = None
    buffer: list[str] = []

    def flush():
        if key is not None and buffer:
            data[key] = " ".join(part.strip() for part in buffer).strip()

    for line in raw.splitlines():
        top = re.match(r"^([A-Za-z_][\w-]*):\s*(.*)$", line)
        if top:
            flush()
            key, value = top.group(1), top.group(2).strip()
            buffer = []
            if value in {"", ">", "|", ">-", "|-", ">+", "|+"}:
                data[key] = ""
            else:
                data[key] = value.strip("\"'")
        elif key is not None and (line.startswith((" ", "\t")) or not line.strip()):
            buffer.append(line)
        elif line.strip():
            raise ValueError(f"cannot parse line: {line!r}")
    flush()
    return data


def is_local_link(target: str) -> bool:
    return not re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I) and not target.startswith("#")


def main() -> int:
    default_root = norm(Path(__file__)).parents[4]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project-root", type=Path, default=default_root,
                        help="folder containing CLAUDE.md and .claude/skills (default: %(default)s)")
    parser.add_argument("--tree", action="store_true", help="print the routing tree")
    args = parser.parse_args()

    root = norm(args.project_root)
    skills_dir = root / ".claude" / "skills"
    report = Report(root)

    def is_workspace(folder: Path) -> bool:
        return folder.name.endswith("-workspace")

    skill_files = {p.parent.name: p for p in sorted(skills_dir.glob("*/SKILL.md"))
                   if not is_workspace(p.parent)}
    for nested in sorted(skills_dir.glob("*/**/SKILL.md")):
        top = nested.relative_to(skills_dir).parts[0]
        if nested.parent.parent != skills_dir and not is_workspace(skills_dir / top):
            report.error(nested, None, "skill is nested inside another folder; move it directly under .claude/skills/")
    for folder in sorted(p for p in skills_dir.glob("*") if p.is_dir()):
        if folder.name.startswith("."):
            continue
        if is_workspace(folder):
            report.warn(folder, None, "evaluation workspace inside .claude/skills; move it outside and keep it out of version control")
        elif folder.name not in skill_files:
            report.warn(folder, None, "folder under .claude/skills has no SKILL.md")

    # Claude Code accepts a project CLAUDE.md at ./CLAUDE.md or ./.claude/CLAUDE.md.
    root_files = [f for f in (root / ROOT, root / ".claude" / ROOT) if f.exists()]
    claude_md = root_files[0] if root_files else root / ROOT
    if root_files:
        combined = "\n".join(f.read_text(encoding="utf-8") for f in root_files)
        if not re.search(r"^#+\s*Following skill routes", combined, re.M | re.I):
            report.warn(claude_md, None, "no 'Following skill routes' section; copy it from skill-authoring/references/routing-contract.md")
    else:
        report.error(claude_md, None, "CLAUDE.md not found at ./CLAUDE.md or ./.claude/CLAUDE.md; skills cannot be reached through routes")
    sources: list[tuple[str, Path]] = [(ROOT, f) for f in root_files] + list(skill_files.items())

    # Per-skill checks: name, frontmatter, size.
    for name, path in skill_files.items():
        text = path.read_text(encoding="utf-8")
        if not NAME_RE.match(name) or len(name) > 64:
            report.error(path, None, f"folder name '{name}' should be lowercase words joined by hyphens, 64 characters max")
        raw, _ = split_frontmatter(text)
        if raw is None:
            report.error(path, 1, "frontmatter missing: the first line must be '---' and the block must be closed with '---'")
            continue
        try:
            meta = parse_frontmatter(raw)
        except Exception as exc:  # noqa: BLE001
            report.error(path, 1, f"frontmatter does not parse: {exc}")
            continue
        description = str(meta.get("description") or "").strip()
        if not description:
            report.error(path, 1, "frontmatter has no description; Claude uses it to decide when to load the skill")
        listing = len(description) + len(str(meta.get("when_to_use") or ""))
        if listing > LISTING_LIMIT:
            report.error(path, 1, f"description plus when_to_use is {listing} characters; the listing cuts off at {LISTING_LIMIT}")
        if "name" in meta and str(meta["name"]).strip() != name:
            report.warn(path, 1, f"frontmatter name '{meta['name']}' differs from folder name '{name}'")
        for unknown in sorted(set(meta) - KNOWN_FIELDS):
            report.warn(path, 1, f"unknown frontmatter field '{unknown}' is ignored by Claude Code")
        line_count = len(text.splitlines())
        if line_count > MAX_LINES:
            report.warn(path, None, f"SKILL.md has {line_count} lines; move detail into references/ (target under {MAX_LINES})")

    # Routes and links.
    edges: dict[str, dict[str, tuple[Path, int]]] = defaultdict(dict)
    supporting: dict[str, set[str]] = defaultdict(set)
    route_count = 0
    for source, path in sources:
        text = path.read_text(encoding="utf-8")
        for number, line in prose_lines(text):
            route_targets = set()
            for match in ROUTE_RE.finditer(line):
                label, target, action = match.groups()
                if not target.endswith("SKILL.md") or not is_local_link(target):
                    continue
                route_targets.add(target)
                route_count += 1
                resolved = norm(path.parent / target.split("#")[0])
                if not resolved.exists():
                    report.error(path, number, f"route target '{target}' does not exist")
                    continue
                if resolved.parent.parent != skills_dir:
                    report.error(path, number, f"route target '{target}' is not a skill directly under .claude/skills/")
                    continue
                target_name = resolved.parent.name
                if label != target_name:
                    report.error(path, number, f"route label '{label}' should match the target folder name '{target_name}'")
                if target_name == source:
                    report.error(path, number, "skill routes to itself")
                    continue
                words = len(re.findall(r"\w+", action))
                if words < MIN_ACTION_WORDS:
                    report.warn(path, number, f"route action '{action.strip()}' is too short to choose the skill reliably")
                edges[source][target_name] = (path, number)
            for target in LINK_RE.findall(line):
                if target in route_targets or not is_local_link(target):
                    continue
                resolved = norm(path.parent / target.split("#")[0])
                if not resolved.exists():
                    report.error(path, number, f"link target '{target}' does not exist")
                elif resolved.name == "SKILL.md" and resolved.parent.parent == skills_dir:
                    supporting[source].add(resolved.parent.name)

    # Links inside skill reference files (routes there are not part of the hierarchy).
    for name, path in skill_files.items():
        for ref in sorted(path.parent.glob("references/**/*.md")):
            for number, line in prose_lines(ref.read_text(encoding="utf-8")):
                for target in LINK_RE.findall(line):
                    if is_local_link(target) and not (ref.parent / target.split("#")[0]).exists():
                        report.error(ref, number, f"link target '{target}' does not exist")

    # Multiple routers and backlinks.
    routers: dict[str, list[str]] = defaultdict(list)
    for source, targets in edges.items():
        for target in targets:
            routers[target].append(source)
    for target, sources_list in sorted(routers.items()):
        if len(sources_list) > 1:
            report.warn(skill_files[target], None, f"routed from several files ({', '.join(sorted(sources_list))}); check ownership")
        for router in sources_list:
            if router != ROOT and router in supporting.get(target, set()):
                report.warn(skill_files[target], None, f"links back to its router '{router}'; forward routes are the only hierarchy")

    # Reachability from CLAUDE.md.
    reached: set[str] = set()
    stack = [ROOT]
    while stack:
        node = stack.pop()
        for child in edges.get(node, {}):
            if child not in reached:
                reached.add(child)
                stack.append(child)
    if root_files:
        for name in sorted(set(skill_files) - reached):
            report.error(skill_files[name], None, "skill is not reachable from CLAUDE.md through Use [...](...) to ... routes")

    # Cycles.
    state: dict[str, int] = {}
    seen_cycles: set[tuple[str, ...]] = set()

    def visit(node: str, trail: list[str]) -> None:
        state[node] = 1
        for child in edges.get(node, {}):
            if state.get(child) == 1:
                cycle = trail[trail.index(child):] + [child]
                key = tuple(sorted(set(cycle)))
                if key not in seen_cycles:
                    seen_cycles.add(key)
                    path, number = edges[node][child]
                    report.error(path, number, "routing cycle: " + " -> ".join(cycle))
            elif state.get(child) is None:
                visit(child, trail + [child])
        state[node] = 2

    for start in [ROOT, *skill_files]:
        if state.get(start) is None:
            visit(start, [start])

    if args.tree:
        print("Routing tree")
        printed: set[str] = set()

        def show(node: str, depth: int) -> None:
            print("  " * depth + ("" if depth == 0 else "- ") + node)
            if node in printed:
                return
            printed.add(node)
            for child in sorted(edges.get(node, {})):
                show(child, depth + 1)

        show(ROOT, 0)
        print()

    for line in report.errors + report.warnings:
        print(line)
    print(f"\n{len(skill_files)} skills, {route_count} routes, "
          f"{len(report.errors)} errors, {len(report.warnings)} warnings")
    return 1 if report.errors else 0


if __name__ == "__main__":
    sys.exit(main())
