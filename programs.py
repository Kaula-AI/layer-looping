"""
Edit grammar for per-token layer paths.

A path is a list of layer indices, e.g. the base path 0..L-1. Edits act on path POSITIONS:
  repeat(i, k)   insert k-1 extra copies of path[i] after it         cost +(k-1)
  skip(i)        delete path[i]                                     cost -1
  swap(i)        exchange path[i] and path[i+1]                     cost 0
  reverse(i, w)  reverse the block path[i:i+w] (w >= 3)             cost 0
  loop(i, w)     run the block path[i:i+w] twice                    cost +w
The first and last `protect` base layers are never touched (they adapt the embedding and the
unembedding; deleting them breaks models, Lad et al. 2406.19384). Paths keep
L - max_skip <= len(path) <= L + max_extra.
"""

KINDS = ("repeat", "skip", "swap", "reverse", "loop")


def path_cost(path, L):
    return len(path) - L


def path_key(path):
    return ",".join(map(str, path))


def parse_key(key):
    return [int(v) for v in key.split(",")] if key else []


def apply_edit(path, edit):
    kind, i, w = edit
    if kind == "repeat":
        return path[: i + 1] + [path[i]] * (w - 1) + path[i + 1 :]
    if kind == "skip":
        return path[:i] + path[i + 1 :]
    if kind == "swap":
        return path[:i] + [path[i + 1], path[i]] + path[i + 2 :]
    if kind == "reverse":
        return path[:i] + path[i : i + w][::-1] + path[i + w :]
    if kind == "loop":
        return path[: i + w] + path[i : i + w] + path[i + w :]
    raise ValueError(kind)


def candidate_edits(path, L, kinds=KINDS, repeats=(2, 3), rev_widths=(3, 4), loop_widths=(2, 3, 4),
                    protect=2, max_extra=6, max_skip=4, focus=None):
    """All valid single edits of `path`, deduplicated by resulting path.
    focus: optional set of layer indices; only edits whose first touched entry is one of them.
    Returns a list of (edit, new_path)."""
    ok_layer = lambda l: protect <= l < L - protect
    n = len(path)
    out, seen = [], {path_key(path)}

    def add(edit, span):
        i = edit[1]
        if i + span > n or not all(ok_layer(path[q]) for q in range(i, i + span)):
            return
        if focus is not None and path[i] not in focus:
            return
        new = apply_edit(path, edit)
        if not (L - max_skip <= len(new) <= L + max_extra):
            return
        k = path_key(new)
        if k not in seen:
            seen.add(k)
            out.append((edit, new))

    for i in range(n):
        if "repeat" in kinds:
            for r in repeats:
                add(("repeat", i, r), 1)
        if "skip" in kinds:
            add(("skip", i, 0), 1)
        if "swap" in kinds:
            add(("swap", i, 2), 2)
        if "reverse" in kinds:
            for w in rev_widths:
                add(("reverse", i, w), w)
        if "loop" in kinds:
            for w in loop_widths:
                add(("loop", i, w), w)
    return out


def single_edit_library(L, **kw):
    """Every single edit of the base path, as (kind, anchor_layer, width_or_k, path)."""
    base = list(range(L))
    return [(e[0], base[e[1]], e[2], p) for e, p in candidate_edits(base, L, **kw)]
