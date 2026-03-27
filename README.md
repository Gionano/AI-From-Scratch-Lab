# Own AI Model From Scratch

A pure Python local AI/ML playground built without external ML libraries.

Suggested GitHub description:

> Local AI/ML playground in pure Python with neural networks from scratch, classical ML baselines, chatbot, and AI foundation demos.

This repository includes:

- A neural network written from scratch
- Classical machine-learning baselines
- A local intent-based chatbot in Indonesian
- An AI stack explorer from AI -> ML -> Neural Networks -> Deep Learning -> Generative AI -> SLM
- Artificial Intelligence foundation demos such as reinforcement learning, algorithm planning, code assistance, and AI ethics checks
- Julia, MATLAB, and C++ reference ports for the numeric model demo

## Features

- `Neural network from scratch`
  Forward pass, backpropagation, configurable activation, configurable loss, gradient descent, momentum, clipping, decay, and metrics.
- `Classical ML`
  Logistic Regression, K-Nearest Neighbours, Decision Tree, and simple feature engineering baselines.
- `Chatbot`
  Local chatbot with intent classification, memory, and AI concept explanations.
- `AI tooling`
  Reinforcement learning demo, algorithm planning helper, code analysis helper, and ethics assessment helper.
- `Project map`
  Built-in AI hierarchy explorer for learning and navigation.

## Project Structure

- `own_ai_model/`
  Core Python modules for ML, chatbot, AI stack, and Artificial Intelligence demos.
- `config/`
  JSON configuration files.
- `chatbot_data/`
  Chatbot training dataset.
- `artifacts/`
  Saved model outputs and generated reports.
- `ports/`
  Julia, MATLAB, and C++ reference implementations for the numeric demo.
- `train.py`
  Train the neural network.
- `predict.py`
  Predict one point or a CSV batch with the trained neural network.
- `train_chatbot.py`
  Train the chatbot model.
- `chat.py`
  Run the chatbot interactively.
- `compare_classical_models.py`
  Compare classical ML models on the synthetic dataset.
- `show_ai_stack.py`
  Show the AI hierarchy used in this project.
- `explore_artificial_intelligence.py`
  Run the reinforcement-learning, planning, code-analysis, and ethics demos.

## Requirements

- Python 3.11+ recommended
- No third-party Python dependencies are required for the current local version

## Setup

Clone the repository:

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

Optional but recommended virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks local scripts, use:

```powershell
powershell -ExecutionPolicy Bypass -File .\train_model.ps1
```

Or run the Python files directly with:

```powershell
.\.venv\Scripts\python.exe train.py
```

## Quick Start

Train the neural network:

```powershell
.\.venv\Scripts\python.exe train.py
```

Predict one point:

```powershell
.\.venv\Scripts\python.exe predict.py 0.1 0.2
```

Train the chatbot:

```powershell
.\.venv\Scripts\python.exe train_chatbot.py
```

Open chatbot mode:

```powershell
.\.venv\Scripts\python.exe chat.py
```

## Main Commands

Train the neural network:

```powershell
.\train_model.ps1
```

Train quietly:

```powershell
.\.venv\Scripts\python.exe train.py --quiet
```

Resume from a saved artifact:

```powershell
.\.venv\Scripts\python.exe train.py --resume-from artifacts/model_params.json
```

Export training history:

```powershell
.\.venv\Scripts\python.exe train.py --history-csv artifacts/training_history.csv
```

Predict from CSV:

```powershell
.\.venv\Scripts\python.exe predict.py --input-file sample_points.csv --output-file artifacts/predictions.csv
```

Compare classical ML models:

```powershell
.\compare_classical_models.ps1 --report-json artifacts/classical_ml_report.json
```

Explore the Artificial Intelligence layer:

```powershell
.\explore_artificial_intelligence.ps1 --rl
.\explore_artificial_intelligence.ps1 --plan "build a chatbot that answers user questions"
.\explore_artificial_intelligence.ps1 --analyze-code predict.py
.\explore_artificial_intelligence.ps1 --ethics "An AI hiring system stores face recordings and fully automated scores for candidates."
```

Show the AI stack:

```powershell
.\show_ai_stack.ps1
.\show_ai_stack.ps1 --concept "machine learning"
.\show_ai_stack.ps1 --json
```

Check runtime:

```powershell
.\check_runtime.ps1
```

Inspect the saved neural-network artifact:

```powershell
.\inspect_model.ps1
```

## Chatbot Notes

The chatbot can:

- Answer greetings and basic coding/help statements
- Remember a user name during the session
- Store short notes
- Explain concepts like AI, machine learning, deep learning, generative AI, and SLM

This is not a full cloud LLM. It is a local intent-based chatbot with limited knowledge compared with larger hosted models.

## AI Stack Coverage

The project currently covers these major layers:

- Artificial Intelligence
- Machine Learning
- Neural Networks
- Deep Learning
- Generative AI
- Small Language Models (SLM)

Implemented or partially implemented examples already exist for:

- Reinforcement Learning
- Algorithm Building
- Augmented Programming
- AI Ethics
- Logistic Regression
- K-Nearest Neighbours
- Decision Trees
- Neural-network training from scratch

## Testing

Run the full local test suite:

```powershell
.\.venv\Scripts\python.exe -m unittest test_model.py test_chatbot.py test_classical_ml.py test_ai_stack.py test_artificial_intelligence.py
```

## Before Pushing To GitHub

Before you push, review these items:

- Do not commit `.venv/`
- Do not commit `__pycache__/`
- Do not commit temporary folders such as `tmp*/`
- Only commit `artifacts/` if you intentionally want saved models and reports in the repository
- Make sure personal paths are not hardcoded in docs or code comments
- Make sure sample outputs are safe to publish

A root `.gitignore` file is included in this project. It ignores the virtual environment, Python cache files, generated artifacts, temporary folders, and editor metadata by default.

Main ignored entries:

```gitignore
.venv/
__pycache__/
*.pyc
artifacts/
tmp*/
```

## Push To GitHub

If this is a new repository:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

If the repository already exists on GitHub, usually you only need:

```bash
git add .
git commit -m "Update project"
git push
```

## Notes

- The neural-network demo learns whether a 2D point is inside or outside a shifted circle.
- The runtime currently works fully on `python + cpu`.
- Torch/CUDA support can be added later if you choose to extend the project in that direction.
