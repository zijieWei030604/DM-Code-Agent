from __future__ import annotations

import subprocess

from dm_agent.core.workspace_version import workspace_version


def test_workspace_version_ignores_gitignored_test_artifacts_but_tracks_new_source(
    tmp_path,
) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("generated.json\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("value = 1\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", ".gitignore", "app.py"],
        check=True,
    )
    initial = workspace_version(tmp_path)

    (tmp_path / "generated.json").write_text('{"cache": 1}\n', encoding="utf-8")
    assert workspace_version(tmp_path) == initial

    (tmp_path / "new_module.py").write_text("value = 2\n", encoding="utf-8")
    assert workspace_version(tmp_path) != initial
