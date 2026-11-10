"""Which files the library may read, and which it only knows by name (design/PRIVACY.md).

A file is read in one of two ways:

* **full** -- its text is parsed, embedded by the model service and quoted in answers;
* **name** ("只按文件名收录") -- it is findable by name, folder and date on this machine, but
  its content is never parsed and nothing about it is sent anywhere.

and a file can also be **excluded**: the library does not register it at all.

The per-file choice is ``files.content_mode`` ("name" or NULL). Rules (``library_rules``)
decide it for files that appear later, and decide what is excluded:

* a *path* rule names a folder or file inside a watched folder by its path relative to that
  folder ("" is the watched folder itself): ``name``, ``full`` (read again below a ``name``
  folder) or ``exclude``;
* a *pattern* rule is a file or folder name, with ``*`` and ``?`` (``*.key``, ``证件``), for
  one watched folder or for every one (``root_id`` NULL): ``name`` or ``exclude``. A pattern
  is matched against every folder name along the path and the file name.

Exclusion wins over everything. Otherwise the most specific rule decides (a deeper match beats
a shallower one; at the same depth a path rule beats a pattern, then ``name`` beats ``full``),
and without a rule the file is read in full.
"""

from __future__ import annotations

from dataclasses import dataclass
import fnmatch
import ntpath
import os
from typing import Any, Iterable, Sequence

ACTIONS = ("name", "full", "exclude")
KINDS = ("path", "pattern")
# Pattern rules can only exclude or keep names; "full" is the absence of a rule.
PATTERN_ACTIONS = ("name", "exclude")
MAX_PATTERN = 200


def split_parts(path: str) -> list[str]:
    """Components of a path below its drive / anchor, Windows style on Windows."""
    if os.name == "nt":
        _, rest = ntpath.splitdrive(path)
        parts = [part for part in rest.replace("/", "\\").split("\\") if part]
    else:
        parts = [part for part in path.split("/") if part]
    return parts


def relative_parts(path: str, root_path: str) -> list[str] | None:
    """``path`` as components below ``root_path``, or None when it is not inside it."""
    try:
        relative = os.path.relpath(path, root_path)
    except ValueError:  # another drive
        return None
    if relative == ".":
        return []
    parts = split_parts(relative)
    if not parts or parts[0] == "..":
        return None
    return parts


def _key(part: str) -> str:
    return os.path.normcase(part)


def clean_pattern(raw: str) -> str:
    """A name pattern as the user typed it, checked: one name, no folders."""
    pattern = (raw or "").strip().strip('"').strip("'").strip()
    if not pattern:
        raise ValueError("请输入文件名或文件夹名，例如 *.key 或 证件。")
    if len(pattern) > MAX_PATTERN or "\\" in pattern or "/" in pattern or ":" in pattern:
        raise ValueError("只填一个名字，不要带路径。要排除某个子文件夹，请在知识库里对它操作。")
    if pattern.strip("*?. ") == "":
        raise ValueError("这个规则会把所有文件都排除，请写具体一些。")
    return pattern


def pattern_matches(pattern: str, name: str) -> bool:
    return fnmatch.fnmatchcase(_key(name), _key(pattern))


@dataclass(frozen=True)
class Decision:
    """What the rules say about one path: excluded (and by which rule), else its mode."""

    excluded: bool
    mode: str  # "name" | "full"
    rule: dict[str, Any] | None = None

    @property
    def name_only(self) -> bool:
        return not self.excluded and self.mode == "name"


FULL = Decision(False, "full")


def decide(
    parts_in_root: Sequence[str] | None,
    path_rules: Iterable[dict[str, Any]],
    pattern_rules: Iterable[dict[str, Any]],
    outer_parts: Sequence[str] = (),
) -> Decision:
    """The decision for a path.

    ``parts_in_root``: its components below the watched folder (None for a single file);
    ``outer_parts``: the components above them (the watched folder's own path, or a single
    file's folders), which patterns also look at. ``path_rules`` are the folder's path rules,
    ``pattern_rules`` its and the global pattern rules (live ones only).
    """
    inner = list(parts_in_root or [])
    inner_keys = [_key(part) for part in inner]
    names = [*outer_parts, *inner]
    # Depth of a match: 0 for the watched folder or anything above it, k for inner[k-1].
    offset = len(outer_parts)
    # (depth, a path rule beats a pattern at the same depth, then "name" wins, rule)
    best: tuple[int, int, int, dict[str, Any]] | None = None

    def consider(depth: int, rule: dict[str, Any]) -> None:
        nonlocal best
        rank = (depth, 1 if rule["kind"] == "path" else 0, 1 if rule["action"] == "name" else 0)
        if best is None or rank > best[:3]:
            best = (*rank, rule)

    if parts_in_root is not None:
        for rule in path_rules:
            rule_parts = [_key(part) for part in split_parts(rule["value"])]
            if inner_keys[: len(rule_parts)] != rule_parts or len(rule_parts) > len(inner_keys):
                continue
            if rule["action"] == "exclude":
                if rule_parts:  # the watched folder itself is never "excluded": it is removed instead
                    return Decision(True, "full", rule)
                continue
            consider(len(rule_parts), rule)
    for rule in pattern_rules:
        for index, name in enumerate(names):
            if not pattern_matches(rule["value"], name):
                continue
            if rule["action"] == "exclude":
                return Decision(True, "full", rule)
            consider(max(0, index - offset + 1), rule)
            break
    if best is None:
        return FULL
    rule = best[3]
    return Decision(False, "name" if rule["action"] == "name" else "full", rule)


def describe_rule(rule: dict[str, Any], root_path: str | None) -> dict[str, Any]:
    """A rule as the UI shows it: with the folder's absolute path for path rules."""
    view = {key: rule.get(key) for key in ("rule_id", "root_id", "kind", "value", "action", "created_at")}
    view["root_path"] = root_path
    if rule.get("kind") == "path" and root_path is not None:
        view["path"] = os.path.join(root_path, rule["value"]) if rule["value"] else root_path
    return view
