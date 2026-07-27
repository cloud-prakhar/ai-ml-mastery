# Answers — 00 Getting Started

Explanations, not one-liners. If you got an answer right for the wrong reason, that still counts
as getting it wrong.

[← Back to the questions](../00-getting-started.md)

---

## Beginner

**1. What does `pwd` tell you?**
"Print Working Directory" — the full path of the folder your terminal is currently in. Every
relative command (`ls`, `cd foldername`, `python script.py`) is interpreted from there, which is
why "file not found" is so often really "wrong directory".

**2. What does `(.venv)` mean?**
A virtual environment is active. Your `python` and `pip` now resolve to the copies inside that
folder rather than the system ones. Its absence is the first thing to check when packages go
missing.

**3. Which command lists installed packages?**
`pip list` for a readable list; `pip freeze` for the pinned format you would redirect into
`requirements.txt`. `pip show <package>` adds the install location, which is useful when you
suspect the wrong environment.

**4. Git vs GitHub?**
Git is version-control software running on your machine — it works with no internet at all.
GitHub is a commercial website that hosts Git repositories and adds collaboration features
(pull requests, issues, actions). Git is the tool; GitHub is one place to put its output.

**5. Why never commit `.venv/`?**
Three reasons. It is large (hundreds of megabytes). It is platform-specific — a Linux `.venv`
is useless on Windows. And it is entirely reconstructible from `requirements.txt`, which is
kilobytes. Commit the recipe, not the meal.

## Conceptual

**6. Explain a virtual environment with an analogy.**
One good answer: imagine every restaurant in a city sharing a single kitchen shelf. One needs the
2019 recipe book, another the 2024 edition, and there is only room for one. Installing what the
second needs silently breaks the first. A virtual environment gives every project its own private
shelf, so nothing you install for one can break another.

The technical reality underneath: it is a folder containing a Python interpreter and its own
`site-packages`, and activation simply puts that folder first on your PATH.

**7. Why `python3 -m pip install`?**
`pip` as a bare command might be a different pip than the one belonging to the Python you are
running — a real risk when several Pythons are installed. `python3 -m pip` guarantees you are
using the pip that belongs to *that specific interpreter*. Inside an activated virtual environment
the distinction disappears, which is another argument for always using one.

**8. Virtual environment vs container?**
A virtual environment isolates Python packages only. Everything else — the operating system, system
libraries, the C compiler, CUDA — is shared with your machine. A container isolates the whole
userspace environment, so it behaves identically on any host.

Practically: virtual environments for local development, containers for sharing and deployment.
They compose — a container usually contains a virtual environment or its equivalent.

**9. Why restart-and-run-all before sharing a notebook?**
Notebooks execute cells in whatever order you clicked them, and the kernel remembers everything.
You can delete the cell that defined a variable and keep using it. The notebook works for you and
fails for everyone else. Restarting clears all state, so a clean run proves the document is
self-contained and correctly ordered.

**10. Why pin versions?**
`numpy` means "whatever the latest version is on the day someone installs this" — which changes,
and which introduces behaviour differences. `numpy==2.1.3` means the same code runs the same way in
a year. In machine learning this matters more than in most software, because a library change can
alter numerical results without raising a single error, and you will not notice.

## Practical

**11. Pandas installed but missing — what to check?**
In order of likelihood:
1. Is the virtual environment activated? Look for `(.venv)` in the prompt.
2. Which Python is running? `which python` / `where python` — is it the one inside `.venv`?
3. Which environment did the install go into? `pip show pandas` reports the location.

If you are in VS Code or Jupyter, add: which interpreter/kernel is the editor using? It maintains
its own selection independently of your terminal.

**12. Make Jupyter use a specific environment.**
Register that environment as a named kernel:
```bash
source .venv/bin/activate
pip install ipykernel
python -m ipykernel install --user --name ai-ml-mastery --display-name "AI/ML Mastery"
```
Then select it via Kernel → Change Kernel. Simply launching `jupyter lab` from inside an activated
environment usually works too, but explicit registration is more reliable.

**13. Recreate a colleague's environment exactly.**
```bash
git clone <their-repo> && cd <repo>
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
This only works if they pinned versions. If their `requirements.txt` has bare names, you get
today's versions, not theirs — which is why pinning matters. For genuinely identical environments
including system libraries, you need a container.

**14. Which Python is active?**
```bash
which python        # macOS / Linux / WSL
where python        # Windows
python -c "import sys; print(sys.executable)"
```
The last one is the most trustworthy: it asks the running interpreter to identify itself, rather
than asking the shell what it *would* run.

**15. Committed an API key — first action?**
**Rotate the key.** Revoke it at the provider and issue a new one. Do this before anything else.

Removing the file, amending the commit or rewriting history does not help on its own: the key was
public, bots scan public repositories within seconds of a push, and you must assume it is
compromised. History cleanup is the second step, not the first.

Then prevent recurrence: `.env` for secrets, `.env` in `.gitignore`, and a secret-scanning hook.

## Scenario

**16. "It works on my machine."**
Ask for: their exact Python version, their `pip freeze` output, their operating system, and the
precise command they ran. Nine times out of ten the diff between their `pip freeze` and yours
contains the answer.

The deeper response is to remove the question entirely — pin dependencies, or containerise, so
"my machine" and "your machine" are the same machine.

**17. No GPU, module needs one.**
Use Google Colab with a GPU runtime. It is free, needs no installation, and every GPU-marked
example in this repository is sized to run there. Alternatives: a rented cloud GPU instance
(costs money, needs care to shut down), or reducing the problem — a smaller model, fewer samples,
or a quantised checkpoint. For learning purposes the smaller version usually teaches the same
lesson.

**18. Compiler error on Windows during pip install.**
Two independent fixes:
1. **Use Conda instead** — it ships pre-compiled binaries for scientific packages, sidestepping
   compilation entirely. This is Conda's strongest use case.
2. **Use WSL2** — build on Linux, where these packages have wheels available and the toolchain is
   what the maintainers actually test against.

A third: install the Microsoft C++ Build Tools so the compilation can succeed. It works, but it is
a large download to fix a problem the other two approaches avoid.

**19. Two PyTorch versions on one machine.**
Two virtual environments. That is the entire answer, and it is why the concept matters:
```bash
cd project-a && python3 -m venv .venv && source .venv/bin/activate && pip install torch==2.0.0
cd project-b && python3 -m venv .venv && source .venv/bin/activate && pip install torch==2.5.0
```
They coexist without knowing about each other.

**20. Notebook gives different results each run.**
Most likely:
1. **No random seed.** Anything involving sampling, shuffling, initialisation or dropout is random
   by default. Set the seed for every library you use.
2. **Out-of-order execution or leftover state.** A variable from a previous run is still in memory
   and is quietly changing your result. Restart and run all to confirm.

Less common but real: multi-threaded floating-point non-determinism, GPU non-determinism, and data
sources that genuinely change between runs.

## Interview

**21. "How do you ensure reproducibility?"**
Cover four layers, in this order:
- **Code**: version control, with the commit hash recorded alongside results
- **Environment**: pinned dependencies, and a container for anything shared or deployed
- **Data**: versioned datasets and recorded checksums — the layer people forget
- **Randomness**: fixed seeds, recorded in the config rather than hardcoded mid-script

Then the honest caveat: bit-for-bit reproducibility across different hardware is often not
achievable, particularly on GPU. The realistic goal is results reproducible within a stated
tolerance, and you should say so rather than claim more.

**22. "Onboard a new team member."**
The target is: clone, one setup command, tests pass — inside an hour. That requires a README with
exact prerequisites, pinned dependencies, a `make setup` or equivalent, a `.env.example`, a
verification script that fails loudly with a fix, and a small first task that exercises the whole
pipeline.

The strongest signal here is treating onboarding friction as a bug in the repository rather than a
deficiency in the new person.

**23. "`pip freeze` vs a hand-written requirements file?"**
`pip freeze` captures everything, including transitive dependencies, at exact versions. It is
maximally reproducible and unreadable — you cannot tell what you actually chose versus what got
pulled in.

A hand-written file lists your direct dependencies, so intent is visible, but leaves transitive
versions floating.

The usual resolution is both: declare direct dependencies with intent (in `pyproject.toml` or a
`.in` file), and generate a fully pinned lock file from it. Tools like `pip-compile` and `uv` exist
precisely to bridge this.

**24. "How do you handle secrets?"**
Never in source, never in notebooks, never in a container image layer. Local development uses a
git-ignored `.env` with an `.env.example` documenting the required keys. Production uses the
platform's secret manager or injected environment variables. Add automated secret scanning in CI
so a mistake is caught before it merges.

And the part people leave out: have a rotation procedure and know that a leaked secret is
compromised the moment it is pushed, not when someone notices.

**25. "Why containerise machine-learning work?"**
Machine learning has an unusually deep and fragile dependency stack: Python packages on top of
system libraries on top of CUDA drivers on top of specific hardware. A version mismatch anywhere
can change numerical results *without producing an error*, which is far worse than crashing.

Containers pin the whole stack, so training and serving run in identical environments, deployment
is a known artefact rather than a hopeful script, and rollback is switching an image tag.

The trade-offs you should mention unprompted: image size, GPU passthrough complexity, and slower
local iteration.

---

[🏠 Module](../../00-getting-started/README.md) · [← Questions](../00-getting-started.md)
