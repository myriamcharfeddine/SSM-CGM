# Overleaf Bidirectional Sync (Premium Git)

You have Overleaf Premium and the Overleaf project already exists at:
`https://overleaf.com/project/6a393f237e2c170adf310484`

The sync uses `git subtree`, which maps the `report/` subfolder in this repo
to the root of the Overleaf project. No separate clone is needed.

---

## One-time setup

### 1. Generate an Overleaf auth token

In Overleaf: **Account (top-right) → Account Settings → Git Authentication Tokens → Generate**.
Copy the token. You will paste it as the password on the first push/pull.

### 2. Add the Overleaf remote

```bash
cd /home/myriamcharfeddine/CGM/SSM-CGM
./scripts/overleaf_sync.sh setup
```

This runs:
```bash
git remote add overleaf https://git@git.overleaf.com/6a393f237e2c170adf310484
```

### 3. Cache your token so you are not prompted every time

```bash
git config --global credential.helper store
```

Then do your first push (step below). Git will ask for the password once and
store the token permanently in `~/.git-credentials`.

---

## Day-to-day workflow

### Edit in VS Code → push to Overleaf

```bash
# 1. Make changes in report/ as normal
# 2. Commit them to the local repo
git add report/
git commit -m "your message"

# 3. Push to Overleaf
./scripts/overleaf_sync.sh push
```

Then open Overleaf and click **Recompile** to see the result.

### Edit in Overleaf → pull to VS Code

```bash
./scripts/overleaf_sync.sh pull
```

This fetches the Overleaf project state and merges it into `report/`.
Changed `.tex` / `.bib` files appear immediately in VS Code.

---

## How it works internally

```
Local repo                           Overleaf project (git root)
──────────────────                   ──────────────────────────
report/                 push/pull    main.tex
  main.tex          ←──────────→    macros.tex
  sections/                         sections/
  figures/                          figures/
  tables/                           tables/
  ...                               references.bib
```

`git subtree push --prefix=report overleaf master` rewrites commits that touch
`report/` and pushes them as if `report/` were the repo root. Overleaf sees
`main.tex` at its root, as expected.

`git subtree pull --prefix=report overleaf master --squash` fetches Overleaf's
current state and merges it back into `report/` with a single squash commit.

---

## Conflict resolution

If you edited locally AND in Overleaf at the same time:

```bash
# 1. Pull Overleaf first (may create a merge commit)
./scripts/overleaf_sync.sh pull

# 2. Resolve any merge conflicts in the affected .tex files, then:
git add report/
git commit -m "merge: resolve conflict with Overleaf edits"

# 3. Push the merged result back to Overleaf
./scripts/overleaf_sync.sh push
```

---

## Compile locally before pushing

```bash
cd /home/myriamcharfeddine/CGM/SSM-CGM/report
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex   # second pass for cross-refs
```

Or with latexmk (preferred):
```bash
latexmk -pdf -interaction=nonstopmode main.tex
```

---

## What NOT to push to Overleaf

Never let these reach Overleaf via push (they are already gitignored):

| Type | Reason |
|---|---|
| `*.parquet`, `*.feather`, `*.pkl` | Participant-level data |
| `*.pth`, `*.ckpt` | Model weights |
| `outputs/` | Too large, participant data |
| `report/*.pdf` | Compiled output, not source |
| `report/*.aux`, `*.log`, `*.bbl` | Build artifacts |

The `.gitignore` already excludes these, so `git add report/` will not
accidentally stage them.

---

## Quick reference

| Goal | Command |
|---|---|
| First-time setup | `./scripts/overleaf_sync.sh setup` |
| Local → Overleaf | `./scripts/overleaf_sync.sh push` |
| Overleaf → Local | `./scripts/overleaf_sync.sh pull` |
| Check status | `./scripts/overleaf_sync.sh status` |
| Compile locally | `cd report && pdflatex -interaction=nonstopmode main.tex` |
