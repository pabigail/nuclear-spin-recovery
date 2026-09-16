#!/usr/bin/env bash
# Run this from INSIDE the empty refactor_nuclear_spin_recovery/ folder.
#   bash bootstrap.sh
set -euo pipefail

PKG=nuclear_spin_recovery                 # importable name
DIST=nuclear-spin-recovery                # PyPI-style name
REPO="$(basename "$PWD")"                 # refactor_nuclear_spin_recovery
PY=python3

# ---------------------------------------------------------------- sanity
if [ -e .git ]; then
  echo "Already a git repo. Refusing to overwrite." >&2; exit 1
fi

# ---------------------------------------------------------------- layout
mkdir -p "src/$PKG" tests/{unit,theory,oracle} tests/data/reference \
         docs notebooks legacy scripts .claude/hooks

touch "src/$PKG/__init__.py" tests/__init__.py
echo "Reference implementation. Frozen. Do not edit." > legacy/README.md

# ---------------------------------------------------------------- pyproject
cat > pyproject.toml <<EOF
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "$DIST"
version = "0.0.0"
description = "Bayesian recovery of nuclear spin configurations"
readme = "README.md"
requires-python = ">=3.11"
dependencies = ["numpy", "scipy"]

[project.optional-dependencies]
dev = [
  "pytest", "pytest-cov", "ruff",
  "jupyterlab", "ipykernel", "jupytext", "matplotlib",
]

[tool.hatch.build.targets.wheel]
packages = ["src/$PKG"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
markers = ["slow: long-running statistical tests"]

[tool.ruff]
line-length = 100
src = ["src", "tests"]
EOF

# ------------------------------------------------- notebooks as paired .py
cat > notebooks/.jupytext.toml <<'EOF'
# Every notebook here is paired with a .py:percent file.
# The .py is the version control artifact; the .ipynb is derived.
formats = "ipynb,py:percent"
EOF

cat > .gitignore <<'EOF'
__pycache__/
*.py[cod]
.venv/
*.egg-info/
build/
dist/
.pytest_cache/
.ruff_cache/
.ipynb_checkpoints/
.DS_Store

# Notebooks are tracked as paired .py:percent files, not .ipynb
notebooks/*.ipynb

# Large or generated data
data/
*.h5
*.hdf5
*.npy
!tests/data/reference/*.npz
EOF

cat > README.md <<EOF
# $DIST

Bayesian recovery of nuclear spin configurations.

## Development

\`\`\`bash
source .venv/bin/activate
pytest
jupyter lab
\`\`\`

Notebooks in \`notebooks/\` are paired with \`.py:percent\` files via jupytext.
Edit either; commit only the \`.py\`.
EOF

# ---------------------------------------------------------------- env
if command -v uv >/dev/null 2>&1; then
  uv venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  uv pip install -e ".[dev]"
else
  "$PY" -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -e ".[dev]"
fi

# Jupyter kernel bound to THIS venv, so notebooks import the editable package
python -m ipykernel install --user --name "$PKG" --display-name "$DIST"

# ---------------------------------------------------------------- git
git init -b main
git add -A
git commit -m "Initial package skeleton"

if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  gh repo create "$REPO" --private --source=. --remote=origin --push
  echo "Private repo created and pushed."
else
  cat <<'EOF'

gh CLI not found or not authenticated. To finish manually:
  gh auth login
  gh repo create <name> --private --source=. --remote=origin --push
Or create the repo on github.com and then:
  git remote add origin git@github.com:<you>/<name>.git
  git push -u origin main
EOF
fi

echo
echo "Done. Next: bash scripts/dev-tmux.sh"
