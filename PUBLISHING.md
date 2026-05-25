# Publishing `invisensing` to PyPI

This document covers the **one-time PyPI setup** and the **per-release
workflow**. The release pipeline is fully automated via GitHub
Actions; you only need to push a git tag.

---

## One-time setup (do this once, ever)

### 1. Create the PyPI account + project

If `invisensing` doesn't exist on PyPI yet, create a placeholder
first so we can wire up Trusted Publishing before the first real
upload:

1. Sign up at <https://pypi.org/account/register/>. Use an
   organisation email (e.g. `contact@invisensing.io`), enable 2FA.
2. (Optional but recommended) Also sign up at
   <https://test.pypi.org/account/register/> with the same email so
   you have a safe sandbox for trial uploads.

### 2. Reserve the project name

Until the first version is uploaded, PyPI doesn't track the project
at all. You can either:

- **Upload a tiny v0.1.0 from your laptop once**, then continue with
  CI from v1.0.0 onwards. Simplest. See the manual fallback below.
- **Or skip the reservation** — when CI uploads v1.0.0 it will create
  the project on the fly. The risk is the name being squatted in the
  meantime, but `invisensing` is unique enough that this is unlikely.

### 3. Wire up Trusted Publishing (OIDC) — *the key step*

Trusted Publishing lets the GitHub Actions workflow upload to PyPI
without any long-lived API token. PyPI verifies the OIDC token GitHub
mints for the job at runtime.

On PyPI:

1. Sign in, go to <https://pypi.org/manage/account/publishing/>.
2. Under **Add a new pending publisher**, fill in:
   - **PyPI Project Name**: `invisensing`
   - **Owner**: `invisensing-io`
   - **Repository name**: `python-lib`
   - **Workflow name**: `release.yml`
   - **Environment name**: `pypi`  *(must match `environment.name`
     in `.github/workflows/release.yml`)*
3. Click **Add**.

If you also want a TestPyPI staging path, repeat at
<https://test.pypi.org/manage/account/publishing/> with the same
fields. Add a separate `testpypi` GitHub environment if you want a
"deploy to TestPyPI first, then prod" gate.

On GitHub:

1. Go to **Settings → Environments → New environment** in the
   `invisensing-io/python-lib` repo.
2. Name it `pypi`. (Optional: add required reviewers so a tag push
   requires manual approval before publishing — recommended for
   public-package paranoia.)

That's it. No `PYPI_API_TOKEN` secret to store.

---

## Per-release workflow

Everything below is the routine you'll run for every new version.

### Step 1 — Bump the version

Open [`pyproject.toml`](pyproject.toml) and bump
`[project] version = "X.Y.Z"`. Follow semver:

- `0.X.Z`: patch (bug fixes, perf, doc-only changes).
- `0.X.0`: minor (additive API only — no breaking changes).
- `X.0.0`: major (breaking changes — write a migration note).

Also bump `[package] version` in [`Cargo.toml`](Cargo.toml) to the
same value so the wheel and the inner Rust crate report the same
number.

### Step 2 — Sanity-check locally

```bash
# Build the sdist + a local wheel, then install the wheel into a
# disposable venv and run the full test suite against it.
rm -rf target/wheels dist
maturin build --release --sdist
python -m venv /tmp/inv-test && source /tmp/inv-test/bin/activate
pip install target/wheels/invisensing-*.whl
pip install pytest
pytest tests/
deactivate && rm -rf /tmp/inv-test
```

If the test suite passes against the freshly-built wheel, you're
good. Skip ahead to step 3.

### Step 3 — Commit + tag + push

```bash
git add pyproject.toml Cargo.toml
git commit -m "release: v1.0.0"
git tag v1.0.0
git push origin master
git push origin v1.0.0       # ← this is what triggers the release
```

### Step 4 — Watch the CI

Go to <https://github.com/invisensing-io/python-lib/actions>. The
`Release to PyPI` workflow takes ~10–15 min and runs five jobs in
parallel:

| Job | Output |
|---|---|
| `wheels · linux · x86_64` | 5 wheels (cp39…cp313) |
| `wheels · linux · aarch64` | 5 wheels (cross-built, no smoke test) |
| `wheels · macos · x86_64` | 5 wheels |
| `wheels · macos · aarch64` | 5 wheels (Apple Silicon native) |
| `wheels · windows · x86_64` | 5 wheels |
| `sdist` | 1 `invisensing-X.Y.Z.tar.gz` |

If all six finish green, the `publish to PyPI` job runs, downloads
all 26 artefacts, and uploads them in one atomic POST. New version
is live on PyPI within seconds.

### Step 5 — Verify it's live

```bash
pip install --upgrade invisensing
python -c "import invisensing; print(invisensing.__version__)"
```

Should print the version you just released.

---

## TestPyPI dry-run (optional)

If you want to test the publish pipeline without touching the real
PyPI (e.g. before the first ever upload):

1. Add a `testpypi` environment in GitHub Settings → Environments.
2. Add a TestPyPI Trusted Publisher with the same fields as above but
   `Environment name = testpypi`.
3. Duplicate `release.yml` as `release-testpypi.yml`, change
   `environment.name` to `testpypi`, add
   `repository-url: https://test.pypi.org/legacy/` to the
   `pypa/gh-action-pypi-publish` step.
4. Push a tag like `v1.0.0-rc1` (the regex matches `v*` so any tag
   triggers — either change the trigger or just be careful which
   workflow you trigger).

Install from TestPyPI with:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple invisensing
```

---

## Manual fallback (in case CI is broken)

You generally shouldn't need this, but for emergencies:

```bash
pip install maturin twine

# Build for the current platform only:
rm -rf target/wheels
maturin build --release --sdist

# Upload using a PyPI API token (one-time create at pypi.org → Account → API tokens):
twine upload target/wheels/*
```

You'll be prompted for the username (`__token__`) and the token. The
manual path only uploads wheels for your local platform — other
users on different platforms / Python versions will have to compile
from the sdist (which requires the Rust toolchain). Reserve this
path for emergencies only; the CI flow is the supported one.

---

## Yanking a bad release

If a release has a serious bug:

1. On PyPI, go to the project page → Manage → Releases → the bad
   version → **Yank release**. Yanked versions stay installable for
   pinned `==X.Y.Z` constraints but are excluded from "give me the
   latest" resolutions.
2. Fix the bug, bump to the next patch version, push the new tag.

Don't delete — yanking preserves the audit trail; deletion is
permanent and the version number can never be reused.

---

## Pre-flight checklist

Before pushing a release tag, confirm:

- [ ] `pyproject.toml` and `Cargo.toml` versions match.
- [ ] `CHANGELOG.md` (or the GitHub Release notes) describes the
      user-visible changes.
- [ ] Local `pytest` is green.
- [ ] `maturin build --release --sdist` succeeds without warnings.
- [ ] The git working tree is clean (`git status` shows nothing).
- [ ] The tag is on `master` and corresponds to a commit that has
      already passed CI (`CI` workflow green for that SHA).
