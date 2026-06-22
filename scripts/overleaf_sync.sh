#!/usr/bin/env bash
# overleaf_sync.sh -- bidirectional sync between report/ and Overleaf (Premium Git)
#
# USAGE
#   ./scripts/overleaf_sync.sh push   # local report/ → Overleaf
#   ./scripts/overleaf_sync.sh pull   # Overleaf → local report/
#   ./scripts/overleaf_sync.sh setup  # one-time: add overleaf remote
#
# AUTHENTICATION
#   Overleaf uses a personal auth token (not your Overleaf password).
#   Generate one at: overleaf.com → Account → Settings → Git Authentication Tokens
#   The first push/pull will prompt for it; git then caches it.
#
# HOW IT WORKS
#   git subtree maps the report/ subdirectory in this repo to the root of the
#   Overleaf project.  No separate clone is needed.

set -euo pipefail

REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
OVERLEAF_REMOTE="overleaf"
OVERLEAF_URL="https://git@git.overleaf.com/6a393f237e2c170adf310484"
SUBTREE_PREFIX="report"
OVERLEAF_BRANCH="master"   # Overleaf always uses master on its side

cd "$REPO_ROOT"

# ── helpers ────────────────────────────────────────────────────────────────────

check_remote() {
    if ! git remote get-url "$OVERLEAF_REMOTE" &>/dev/null; then
        echo "ERROR: remote '$OVERLEAF_REMOTE' not found. Run:"
        echo "  ./scripts/overleaf_sync.sh setup"
        exit 1
    fi
}

# ── commands ───────────────────────────────────────────────────────────────────

case "${1:-help}" in

setup)
    if git remote get-url "$OVERLEAF_REMOTE" &>/dev/null; then
        echo "Remote '$OVERLEAF_REMOTE' already exists: $(git remote get-url $OVERLEAF_REMOTE)"
    else
        git remote add "$OVERLEAF_REMOTE" "$OVERLEAF_URL"
        echo "Added remote '$OVERLEAF_REMOTE' → $OVERLEAF_URL"
    fi
    echo ""
    echo "Next steps:"
    echo "  1. Generate an auth token at overleaf.com → Account → Settings → Git Authentication Tokens"
    echo "  2. Run:  ./scripts/overleaf_sync.sh push"
    echo "     When prompted for password, paste the token (it will be cached)."
    ;;

push)
    check_remote
    echo "── Pushing report/ → Overleaf ──────────────────────────────────────────"
    echo "  This rewrites the $SUBTREE_PREFIX/ history into Overleaf's root."
    echo "  Current HEAD commit:"
    git log --oneline -1
    echo ""
    git subtree push --prefix="$SUBTREE_PREFIX" "$OVERLEAF_REMOTE" "$OVERLEAF_BRANCH"
    echo ""
    echo "Done. Open overleaf.com and click Recompile to see your changes."
    ;;

pull)
    check_remote
    echo "── Pulling Overleaf → report/ ──────────────────────────────────────────"
    echo "  Fetching Overleaf changes and merging into $SUBTREE_PREFIX/ ..."
    echo ""
    git subtree pull \
        --prefix="$SUBTREE_PREFIX" \
        "$OVERLEAF_REMOTE" "$OVERLEAF_BRANCH" \
        --squash \
        -m "sync: pull from Overleaf $(date +%Y-%m-%d)"
    echo ""
    echo "Done. Changed files are now in report/. Review with:  git diff HEAD~1"
    ;;

status)
    check_remote
    echo "── Overleaf remote ─────────────────────────────────────────────────────"
    git remote get-url "$OVERLEAF_REMOTE"
    echo ""
    echo "── Local report/ changes not yet pushed ────────────────────────────────"
    git status report/ --short
    ;;

help|*)
    echo "Usage: ./scripts/overleaf_sync.sh <command>"
    echo ""
    echo "Commands:"
    echo "  setup   Add the Overleaf git remote (run once)"
    echo "  push    Send local report/ changes → Overleaf"
    echo "  pull    Fetch Overleaf changes → local report/"
    echo "  status  Show remote URL and uncommitted changes in report/"
    ;;
esac
