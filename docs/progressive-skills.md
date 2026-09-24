# Progressive skills

At task start the runtime publishes a short skill catalog. Keyword/regex selection
marks recommendations but does not inject instructions or activate specialist tools.
The model calls `load_skill(name)` to read instructions and register any trusted
Python skill tools, then `load_skill_resource(name, resource)` for supporting text.
Both calls follow the ordinary tool validation, observation bounding and trace path.

Discovery precedence, from lowest to highest:

1. Registered Python skills and legacy bundled JSON skills.
2. Bundled `dm_agent/skills/packages/*/SKILL.md`.
3. User `~/.dm_agent/skills/*/SKILL.md`.
4. Project `<working-directory>/.dm_agent/skills/*/SKILL.md`.

Each package has YAML frontmatter containing `name` and `description`; optional
`keywords`, `patterns` and integer `priority` influence recommendations. Only metadata
is parsed during discovery. Markdown packages cannot register executable Python tools.

```markdown
---
name: testing-guide
description: Python test diagnosis and regression testing
keywords: [pytest, testing]
---
Read the relevant implementation before changing tests.
For fixture diagnosis, load resource fixtures.md.
```

Put supporting files under `resources/`, for example `resources/fixtures.md`.
Pass `fixtures.md` to `load_skill_resource`. Absolute paths, traversal and symlinks
escaping the resource directory are rejected. Existing tool names cannot be replaced
by skill activation. Skill tools become available in subsequent model requests.

Loading records source and SHA-256 content hash in tool-result metadata and optional
`skill_entry_loaded` / `skill_resource_loaded` trace events. `skill_recommendations`
is separate from `activated_skills`, which tracks actual entry loads in the task.
New tasks reset active skill tools to the base tool set. Repeated reads return the
full current content through normal output limits; no permanent loaded flag blocks
reloading after LCM compaction. Compression may summarize these tool results.
