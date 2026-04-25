from collections.abc import Iterable
from datetime import date
from pathlib import Path
from urllib.parse import quote
import re


def project_folder_name(project_title: str | None, project_id: str) -> str:
    raw_name = (project_title or "").strip() or project_id
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1F]', " ", raw_name)
    sanitized = re.sub(r"\s+", " ", sanitized).strip().rstrip(".")
    return sanitized or project_id


def output_bucket_name(output_bucket: str | None) -> str:
    raw_name = (output_bucket or "").strip() or "misc"
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1F]', " ", raw_name)
    sanitized = re.sub(r"\s+", "_", sanitized).strip(" ._").lower()
    return sanitized or "misc"


def project_output_root(*, output_dir: str | Path, project_title: str | None, project_id: str) -> Path:
    return Path(output_dir) / project_folder_name(project_title, project_id)


def bucket_output_root(*, output_dir: str | Path, output_bucket: str | None) -> Path:
    return Path(output_dir) / output_bucket_name(output_bucket)


def dated_project_output_root(
    *,
    output_dir: str | Path,
    output_bucket: str | None,
    project_title: str | None,
    project_id: str,
    rendered_on: date | None = None,
) -> Path:
    return (
        bucket_output_root(output_dir=output_dir, output_bucket=output_bucket)
        / (rendered_on or date.today()).isoformat()
        / project_folder_name(project_title, project_id)
    )


def stage_output_dir(
    *,
    output_dir: str | Path,
    output_bucket: str | None,
    project_title: str | None,
    project_id: str,
    stage_name: str,
    rendered_on: date | None = None,
) -> Path:
    return dated_project_output_root(
        output_dir=output_dir,
        output_bucket=output_bucket,
        project_title=project_title,
        project_id=project_id,
        rendered_on=rendered_on,
    ) / stage_name


def dated_stage_output_dir(
    *,
    output_dir: str | Path,
    output_bucket: str | None,
    project_title: str | None,
    project_id: str,
    stage_name: str,
    rendered_on: date | None = None,
) -> Path:
    return stage_output_dir(
        output_dir=output_dir,
        output_bucket=output_bucket,
        project_title=project_title,
        project_id=project_id,
        stage_name=stage_name,
        rendered_on=rendered_on,
    )


def iter_project_output_roots(
    *,
    output_dir: str | Path,
    project_title: str | None,
    project_id: str,
    output_bucket: str | None = None,
    known_buckets: Iterable[str] = ("clipping", "faceless_story", "reddit"),
) -> list[Path]:
    root = Path(output_dir)
    project_folder = project_folder_name(project_title, project_id)
    candidate_paths: list[Path] = []

    if output_bucket:
        buckets = [output_bucket_name(output_bucket)]
    else:
        buckets = [output_bucket_name(bucket) for bucket in known_buckets]

    for bucket in buckets:
        bucket_root = root / bucket
        if not bucket_root.exists():
            continue
        for dated_dir in bucket_root.iterdir():
            if not dated_dir.is_dir():
                continue
            project_root = dated_dir / project_folder
            if project_root.exists():
                candidate_paths.append(project_root)

    legacy_project_root = root / project_folder
    if legacy_project_root.exists():
        candidate_paths.append(legacy_project_root)

    return candidate_paths


def output_url(*, output_dir: str | Path, file_path: str | Path) -> str:
    relative_path = Path(file_path).resolve().relative_to(Path(output_dir).resolve())
    return f"/outputs/{quote(relative_path.as_posix(), safe='/')}"
