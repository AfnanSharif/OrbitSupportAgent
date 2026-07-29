<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=00c6ff,0072ff&height=200&section=header&text=OrbitSupportAgent&fontSize=66&fontColor=ffffff&animation=twinkling" width="100%" />

<img src="https://img.icons8.com/?id=wdcncLXBKPEU&format=png&size=100" width="90" />

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=600&size=22&duration=2500&pause=1000&color=00c6ff&center=true&vCenter=true&width=700&height=50&lines=Graph-routed%20customer%20support%20assistant;Python%20+%20Streamlit" alt="Typing SVG" />

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](#)
[![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)](#)
[![Transformers](https://img.shields.io/badge/Transformers-FFD21E?style=for-the-badge)](#)

</div>

---

## 📖 Overview

**OrbitSupportAgent** — Graph-routed customer support assistant.

Core logic lives in `src/support_copilot/`. Configuration is centralized in `config/settings.yaml`
and secrets/API keys are loaded from a local `.env` (see `.env.example`).

## 🏗️ Project Layout

```
OrbitSupportAgent/
├── app.py               # Streamlit entry point
├── src/support_copilot/
│   └── ...              # Core package — graph-routed customer support assistant
├── config/settings.yaml # App configuration
├── tests/                # Unit tests
├── scripts/setup.sh      # venv + install helper (macOS/Linux)
├── requirements.txt
```


## ⚡ Setup & Run

### 🪟 Windows (PowerShell / CMD)
```cmd
git clone https://github.com/AfnanSharif/OrbitSupportAgent.git
cd OrbitSupportAgent

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

copy .env.example .env
:: edit .env to add any API keys — the app runs fully offline without them

streamlit run app.py
```

### 🍎 macOS / 🐧 Linux
```bash
git clone https://github.com/AfnanSharif/OrbitSupportAgent.git
cd OrbitSupportAgent

./scripts/setup.sh                 # creates .venv and installs requirements.txt
source .venv/bin/activate

cp .env.example .env
# edit .env to add any API keys — the app runs fully offline without them

streamlit run app.py
```

Open **http://localhost:8501**.

```bash
make test    # run the test suite
make lint    # lint the codebase
```

---

<div align="center">

**Created by [AfnanSharif](https://github.com/AfnanSharif)** · ⭐ star this repo if it helped you

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=00c6ff,0072ff&height=80&section=footer" width="100%" />

</div>
