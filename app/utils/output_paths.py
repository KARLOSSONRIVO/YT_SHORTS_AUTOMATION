from pathlib import Path
from urllib.parse import quote
import re


def project_folder_name(project_title: str | None, project_id: str) -> str:
    raw_name = (project_title or "").strip() or project_id
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1F]', " ", raw_name)
    sanitized = re.sub(r"\s+", " ", sanitized).strip().rstrip(".")
    return sanitized or project_id


def project_output_root(*, output_dir: str | Path, project_title: str | None, project_id: str) -> Path:
    return Path(output_dir) / project_folder_name(project_title, project_id)


def stage_output_dir(
    *,
    output_dir: str | Path,
    project_title: str | None,
    project_id: str,
    stage_name: str,
) -> Path:
    return project_output_root(
        output_dir=output_dir,
        project_title=project_title,
        project_id=project_id,
    ) / stage_name


def output_url(*, output_dir: str | Path, file_path: str | Path) -> str:
    relative_path = Path(file_path).resolve().relative_to(Path(output_dir).resolve())
    return f"/outputs/{quote(relative_path.as_posix(), safe='/')}"
