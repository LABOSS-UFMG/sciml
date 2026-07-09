# Documentation

This folder contains the documentation for the library — a set of tools for building and training Physics-Informed Neural Networks (PINNs) on top of PyTorch.

It's written primarily for undergraduate research students who are learning the library, PINNs, and (in some cases) Python itself, at the same time. Because of that, each document tries not only to describe *what* a class does, but *why* it's designed the way it is, and *where* it fits in the bigger picture — so that, once you're comfortable with the basics, you're also equipped to read, modify, and extend the code itself.

## Where to start

If this is your first contact with the library, read the documents in this order:

1. **[`1-core.md`](1-core.md)** — start here. Explains the overall architecture: the abstract contracts every component follows (`LossBase`, `SamplerBase`, `MetricBase`, `CallbackBase`, `TrainerBase`), how they connect to each other through `Batch`, and — most importantly — **which parts you'll actually need to write yourself**, versus which parts already come implemented and read to use.

2. **[`2.1-metrics.md`](2.1-metrics.md)** — how to monitor how well the network is doing, without affecting training. Covers `MeanAbsoluteError`, `MeanSquaredError`, `RootMeanSquaredError`, `MaxError`, `LpNorm`, and `RelativeLpNorm`.

3. **[`2.2-losses.md`](2.2-losses.md)** — the one component you're most likely to write from scratch, since the PDE residual is specific to your problem. Covers `Supervised` (known values: initial/boundary conditions, observed data) and `Residual` (PDE residuals).

3. **[`2.3-samplers.md`](2.3-samplers.md)** — how training points are generated. Covers `LatinHypercube`, the general-purpose sampler used for collocation, boundary, and initial-condition points alike.

4. **[`2.4-callbacks.md`](2.4-callbacks.md)** — how to save progress (`Checkpoint`) and record training history (`Logger`) as training runs.

5. **[`2.5-trainers.md`](2.5-trainers.md)** — the orchestrator that ties everything above together. Covers `Standard`, including how it supports both first-order optimizers (Adam, SGD) and L-BFGS with the same code.

## How the documentation is organized

Each document follows the same structure as the codebase itself: one document per module inside the library, generally split into:

- **`core`** — the abstract contracts (base classes) every component must follow. Read once, rarely revisited.

- **One document per concrete module** (`losses`, `samplers`, `metrics`, `callbacks`, `trainers`) — the actual implementations you use (and, sometimes, extend) to solve a problem.

This mirrors the codebase layout:

```
sciml/
├── core/           →  core.md
├── losses/         →  losses.md
├── samplers/       →  samplers.md
├── metrics/        →  metrics.md
├── callbacks/      →  callbacks.md
└── trainers/       →  trainers.md
```

---

## A note on code vs. documentation

Docstrings in the code (in English, like this documentation) describe each class and method precisely, for quick reference while coding — that's what your editor's autocomplete/hover will show you. These `docs/*.md` files complement that with the bigger picture: how pieces relate to each other, which design trade-offs were made and why, and worked examples that are easier to follow as prose than as a docstring.

If you ever find the documentation describing something that doesn't match what the code actually does, trust the code — and please flag the discrepancy, so the documentation can be fixed.
