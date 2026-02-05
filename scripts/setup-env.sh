#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
cd "$ROOT_DIR"

MANAGER="auto"
INSTALL_PYTHON=0
WITH_DEV=1
RUN_SETUP=0
RUN_DOCTOR=0
PYTHON_BIN="${PYTHON_BIN:-}"

usage() {
  cat <<'EOF'
Usage: ./scripts/setup-env.sh [options]

Options:
  --manager <auto|venv|poetry>  Environment manager (default: auto)
  --install-python              Install .python-version via pyenv if needed
  --python <path>               Explicit Python executable to use
  --no-dev                      Install only runtime dependencies
  --run-setup                   Run 'transcribe-local setup --non-interactive'
  --run-doctor                  Run 'transcribe-local doctor' after install
  -h, --help                    Show this help

Examples:
  ./scripts/setup-env.sh
  ./scripts/setup-env.sh --install-python
  ./scripts/setup-env.sh --manager poetry --install-python --run-doctor
EOF
}

fail() {
  echo "error: $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manager)
      [[ $# -ge 2 ]] || fail "--manager requires a value"
      MANAGER="$2"
      shift 2
      ;;
    --install-python)
      INSTALL_PYTHON=1
      shift
      ;;
    --python)
      [[ $# -ge 2 ]] || fail "--python requires a path"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --no-dev)
      WITH_DEV=0
      shift
      ;;
    --run-setup)
      RUN_SETUP=1
      shift
      ;;
    --run-doctor)
      RUN_DOCTOR=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "unknown option: $1"
      ;;
  esac
done

case "$MANAGER" in
  auto|venv|poetry) ;;
  *) fail "--manager must be one of: auto, venv, poetry" ;;
esac

PYTHON_VERSION=""
if [[ -f "$ROOT_DIR/.python-version" ]]; then
  PYTHON_VERSION="$(tr -d '[:space:]' < "$ROOT_DIR/.python-version")"
fi

resolve_python() {
  if [[ -n "$PYTHON_BIN" ]]; then
    [[ -x "$PYTHON_BIN" ]] || fail "python not executable: $PYTHON_BIN"
    echo "$PYTHON_BIN"
    return
  fi

  if command -v pyenv >/dev/null 2>&1 && [[ -n "$PYTHON_VERSION" ]]; then
    if pyenv versions --bare | grep -Fxq "$PYTHON_VERSION"; then
      echo "$(pyenv prefix "$PYTHON_VERSION")/bin/python"
      return
    fi

    if [[ "$INSTALL_PYTHON" -eq 1 ]]; then
      echo "Installing Python $PYTHON_VERSION via pyenv..."
      pyenv install -s "$PYTHON_VERSION"
      echo "$(pyenv prefix "$PYTHON_VERSION")/bin/python"
      return
    fi

    echo "pyenv detected but Python $PYTHON_VERSION is not installed."
    echo "Re-run with --install-python to install it automatically."
  fi

  if command -v python3 >/dev/null 2>&1; then
    echo "$(command -v python3)"
    return
  fi

  fail "python3 not found"
}

PYTHON_BIN="$(resolve_python)"
PYTHON_VERSION_ACTUAL="$("$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(f"Python >= 3.10 required, found {sys.version.split()[0]}")
print(sys.version.split()[0])
PY
)"

if [[ "$MANAGER" == "auto" ]]; then
  if command -v poetry >/dev/null 2>&1; then
    MANAGER="poetry"
  else
    MANAGER="venv"
  fi
fi

echo "Repository: $ROOT_DIR"
echo "Python: $PYTHON_BIN ($PYTHON_VERSION_ACTUAL)"
echo "Manager: $MANAGER"

run_cli() {
  if [[ "$MANAGER" == "poetry" ]]; then
    poetry run transcribe-local "$@"
  else
    "$ROOT_DIR/.venv/bin/transcribe-local" "$@"
  fi
}

if [[ "$MANAGER" == "poetry" ]]; then
  command -v poetry >/dev/null 2>&1 || fail "poetry not found in PATH"
  poetry config virtualenvs.in-project true --local >/dev/null
  poetry env use "$PYTHON_BIN"
  if [[ "$WITH_DEV" -eq 1 ]]; then
    poetry install
  else
    poetry install --without dev
  fi
else
  "$PYTHON_BIN" -m venv "$ROOT_DIR/.venv"
  "$ROOT_DIR/.venv/bin/python" -m pip install --upgrade pip setuptools wheel
  if [[ "$WITH_DEV" -eq 1 ]]; then
    "$ROOT_DIR/.venv/bin/pip" install -e ".[dev]"
  else
    "$ROOT_DIR/.venv/bin/pip" install -e .
  fi
fi

if [[ "$RUN_SETUP" -eq 1 ]]; then
  run_cli setup --non-interactive
fi

if [[ "$RUN_DOCTOR" -eq 1 ]]; then
  run_cli doctor
fi

echo ""
echo "Environment is ready."
if [[ "$MANAGER" == "poetry" ]]; then
  echo "Run commands with: poetry run transcribe-local ..."
else
  echo "Activate env with: source .venv/bin/activate"
  echo "Then run: transcribe-local ..."
fi
