# Task: Dead Code Cleanup

Read CLAUDE.md. Audit the entire codebase and remove all dead code.
This is a cleanup task — no new features, only removal and fixes.

## Known Dead Code

### 1. src/nova/tools/reminders.py (old asyncio timer implementation)
This file was replaced by heartbeat_reminders.py in Task 39.
It's disconnected from registry.py but still exists.
- Delete this file entirely
- Remove any imports referencing it anywhere in the project
- Search: `grep -r "from nova.tools.reminders" src/`
- Search: `grep -r "import reminders" src/`

### 2. tests/test_tools.py — broken TestUserMemory tests
These tests reference `nova.memory.persistent._MEMORY_FILE` and
`nova.memory.persistent._MEMORY_DIR` which no longer exist since
Phase 3 SQLite migration. They error on every test run:
```
AttributeError: <module 'nova.memory.persistent'> does not have the attribute '_MEMORY_FILE'
```
- If nova.memory.persistent still exists but is unused, delete the module
- If it's still used somewhere, fix the tests
- If the test class TestUserMemory tests functionality that's now covered
  by other tests (test_memory_store.py), delete the test class
- Search: `grep -r "nova.memory.persistent" src/ tests/`

### 3. src/nova/memory/persistent.py (old JSON-based memory)
Phase 3 migrated to SQLite (memory_store.py). Check if persistent.py
is still imported or used anywhere:
- Search: `grep -r "from nova.memory.persistent\|import persistent" src/`
- If no imports found, delete the file
- If still imported somewhere, trace and update those references

## Full Audit

Run these searches and act on results:

### Unused imports
```bash
# Check with ruff for unused imports
ruff check src/ --select F401
```
Fix all F401 (unused import) errors.

### Unused functions/methods
```bash
# Find all function definitions
grep -rn "def " src/nova/ --include="*.py" | grep -v "__" | grep -v "test_"
```
For each function, verify it's actually called somewhere:
```bash
grep -rn "function_name" src/ tests/
```
If a function is defined but never called (and not part of a public API
or tool registry), remove it.

### Unused files
Check every .py file in src/nova/ — is it imported by anything?
```bash
# List all Python files
find src/nova -name "*.py" ! -name "__init__.py"
```
For each file, check if it's imported:
```bash
grep -r "from nova.module_name\|import module_name" src/
```

### Orphaned test files
Check if test files test modules that still exist:
```bash
ls tests/test_*.py
```
Each test file should correspond to existing source modules.

### Empty __init__.py files
These are fine to keep, but check if any __init__.py has actual code
that's no longer needed.

### Commented-out code
```bash
grep -rn "^#.*def \|^#.*class \|^#.*import " src/nova/ --include="*.py"
```
Remove large blocks of commented-out code. Comments explaining WHY
are fine — commented-out CODE is not.

### Unused dependencies
Check pyproject.toml or requirements — are all listed packages actually
imported somewhere in src/?

## Rules
- Do NOT delete any file without first verifying it's not imported/used
- Do NOT change any functionality — only remove dead code
- Run `python -m pytest tests/ -x` after each deletion to verify nothing breaks
- Run `ruff check src/ tests/` to verify no new errors
- If unsure whether something is dead, leave it and add a comment:
  `# TODO: verify if this is still used`

## Verification Checklist
- [ ] Old reminders.py deleted
- [ ] persistent.py deleted (if unused)
- [ ] Broken tests in test_tools.py fixed or removed
- [ ] `ruff check src/ --select F401` — no unused imports
- [ ] No large blocks of commented-out code remain
- [ ] `python -m pytest tests/ -x` — all pass, 0 errors
- [ ] `ruff check src/ tests/` — clean
- [ ] CLAUDE.md updated with Task 48
- [ ] git diff --stat shows only deletions/modifications, no new files
