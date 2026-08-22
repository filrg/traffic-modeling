from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

from traffic_modeling.core import ArtifactRecord, sha256_file

from .context import PipelineContext, PortableStateStore
from .manifest import MANIFEST_NAME

ArtifactKey: TypeAlias = tuple[str, str]
FileStamp: TypeAlias = tuple[int, int, int]
ArtifactSnapshot: TypeAlias = dict[ArtifactKey, FileStamp]


def artifact_roots(context: PipelineContext) -> tuple[tuple[str, Path], ...]:
    results = context.results.resolve()
    roots = [("results", results)]
    if isinstance(context.state_store, PortableStateStore):
        state = context.state_store.directory.resolve()
        if state != results and results not in state.parents:
            roots.append(("state", state))
    return tuple(roots)


def _stamp(path: Path) -> FileStamp:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ino


def _scan(context: PipelineContext) -> dict[ArtifactKey, tuple[Path, FileStamp]]:
    files: dict[ArtifactKey, tuple[Path, FileStamp]] = {}
    for scope, root in artifact_roots(context):
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"artifact trees cannot contain symbolic links: {path}")
            if not path.is_file():
                continue
            relative_path = path.relative_to(root).as_posix()
            if scope == "results" and relative_path == MANIFEST_NAME:
                continue
            files[(scope, relative_path)] = (path, _stamp(path))
    return files


def snapshot_artifacts(context: PipelineContext) -> ArtifactSnapshot:
    """Capture cheap file metadata before a stage runs."""
    return {key: stamp for key, (_, stamp) in _scan(context).items()}


def current_artifact(
    context: PipelineContext,
    address: ArtifactRecord,
) -> ArtifactRecord | None:
    roots = dict(artifact_roots(context))
    root = roots.get(address.scope)
    if root is None:
        return None
    path = root / address.relative_path
    if path.is_symlink():
        raise ValueError(f"artifacts cannot be symbolic links: {path}")
    if not path.is_file():
        return None
    stamp = _stamp(path)
    digest = sha256_file(path)
    if _stamp(path) != stamp:
        raise RuntimeError(f"artifact changed while it was being hashed: {path}")
    return ArtifactRecord(address.scope, address.relative_path, digest, stamp[0])


def changed_artifacts(
    context: PipelineContext,
    before: ArtifactSnapshot,
) -> tuple[ArtifactRecord, ...]:
    """Hash files created or rewritten since ``before``."""
    records = []
    for (scope, relative_path), (path, stamp) in _scan(context).items():
        if before.get((scope, relative_path)) == stamp:
            continue
        digest = sha256_file(path)
        if _stamp(path) != stamp:
            raise RuntimeError(f"artifact changed while it was being hashed: {path}")
        records.append(
            ArtifactRecord(
                scope=scope,
                relative_path=relative_path,
                sha256=digest,
                size_bytes=stamp[0],
            )
        )
    return tuple(sorted(records, key=lambda item: (item.scope, item.relative_path)))
