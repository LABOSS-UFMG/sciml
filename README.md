# sciml

**sciml** is a modular and extensible framework for building Physics-Informed Neural Networks (PINNs) in Python.

## 1. Instalation

Install directly from github:

```bash
pip install git+https://github.com/LABOSS-UFMG/sciml
```

If you want to contribute with the code, follow the steps below.

## 2. Contributing to sciml

Follow these steps to set up your development environment and contribute to the project.

### 2.1. Clone the repository

```bash
git clone git@github.com:LABOSS-UFMG/sciml.git
cd sciml
```

### 2.2. Create a virtual environment

```bash
python3 -m venv .venv
```

### 2.3. Activate the virtual environment

```bash
source .venv/bin/activate
```

You should see `(.venv)` appear in the terminal prompt.

### 2.4. Install sciml in development mode

```bash
pip install -e .[dev]
```

* `-e` → editable mode
* `[dev]` → installs dev dependencies (pytest, linters, etc.)

### 2.5. Develop, commit, push

```bash
git add .
git commit -m "Implement new adaptive sampler"
git push
```
