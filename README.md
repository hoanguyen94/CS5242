# Project Environment Setup

This project uses **Conda** to manage the Python environment and dependencies.  
Follow the steps below to recreate the development environment.

---

## 1. Install Conda

If you do not have Conda installed, install **Miniconda** (recommended) or **Anaconda**.

Recommended: Miniconda  
https://docs.conda.io/en/latest/miniconda.html

Verify installation:

```bash
conda --version
```

---

## 2. Clone the Repository

```bash
git clone <repo_url>
cd <repo_name>
```

---

## 3. Create the Conda Environment

The project dependencies are specified in the file:

```
environment.yml
```

Create the environment:

```bash
conda env create -f environment.yml
```

---

## 4. Activate the Environment

```bash
conda activate cs5252_project
```

Verify Python version:

```bash
python --version
```

---

## 5. Update the Environment

If the `environment.yml` file changes, update the environment:

```bash
conda env update -f environment.yml --prune
```

The `--prune` option removes packages that are no longer required.

---

## 6. Running Jupyter Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

or

```bash
jupyter lab
```

---

## 7. Export Environment Updates (For Contributors)

If you install new packages and want to update the environment file:

```bash
conda env export --from-history > environment.yml
```

---

## 8. Removing the Environment

```bash
conda remove --name cs5252_project --all
```

---

## 9. Troubleshooting

If dependency conflicts occur, recreate the environment:

```bash
conda remove --name cs5252_project --all
conda env create -f environment.yml
```

---

## 10. Project Structure

Example repository layout:

```
project/
│
├── data/
├── notebooks/
├── src/
├── environment.yml
└── README.md
```

---

## Notes

- Always activate the environment before running the project.
- If you add new dependencies, remember to update `environment.yml`.