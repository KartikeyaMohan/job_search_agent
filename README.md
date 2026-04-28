# 🤖 AI Job Search Agent

An autonomous multi-agent system that finds jobs, scores their relevance to your profile, tailors your resume for each application, and drafts personalized recruiter messages — all from the terminal.

---

## Features

- **Multi-source job discovery** — scrapes LinkedIn, Wellfound, and JSearch (aggregates Indeed, Glassdoor, ZipRecruiter)
- **RAG-powered matching** — indexes your resume into a local vector store and retrieves relevant experience per job
- **ATS scoring** — pure-Python TF-IDF + keyword analysis to predict ATS pass rate before applying
- **Resume tailoring** — Claude rewrites your resume for each job, maximising keyword coverage while keeping it authentic
- **Outreach generation** — personalized LinkedIn connection requests, InMails, cold emails, and cover letters
- **Application tracker** — SQLite database tracking every job's status from found → applied → response
- **Feedback loop** — records which applications got responses to improve future matching

---

## Architecture

```
main.py              ← CLI (Typer + Rich)
orchestrator.py      ← CrewAI workflow coordinator
config.py            ← Settings (.env driven)

agents/
  research_agent.py  ← Discovers jobs from all sources
  matching_agent.py  ← Scores relevance via ATS + RAG
  resume_agent.py    ← Tailors resume with Claude
  outreach_agent.py  ← Writes recruiter messages

tools/
  ats_scorer.py            ← TF-IDF keyword scoring (no API cost)
  retry.py                 ← Exponential backoff for API calls
  rag/
    document_processor.py  ← Parses PDF/DOCX resumes
    vector_store.py        ← ChromaDB + sentence-transformers
  scrapers/
    linkedin_scraper.py    ← Playwright headless browser
    wellfound_scraper.py   ← Playwright + BeautifulSoup fallback
    jsearch_scraper.py     ← RapidAPI (free tier: 200 req/month)
    generic_scraper.py     ← Any company careers page URL

models/              ← Pydantic models (Job, CandidateProfile, Application)
database/db.py       ← SQLAlchemy + SQLite
```

### Multi-Agent Pipeline

```
┌─────────────────┐     ┌─────────────────┐
│  Research Agent │────▶│ Matching Agent  │
│  (finds jobs)   │     │ (scores jobs)   │
└─────────────────┘     └────────┬────────┘
                                  │ top N jobs
                    ┌─────────────┴──────────────┐
                    ▼                            ▼
          ┌─────────────────┐        ┌─────────────────────┐
          │  Resume Agent   │        │   Outreach Agent    │
          │ (tailors resume)│        │ (writes messages)   │
          └─────────────────┘        └─────────────────────┘
```

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Claude (Anthropic API) with prompt caching |
| Agents | CrewAI |
| RAG | ChromaDB + sentence-transformers (`all-MiniLM-L6-v2`) |
| Scraping | Playwright, BeautifulSoup4, RapidAPI JSearch |
| ATS Scoring | Pure Python (TF-IDF cosine + keyword extraction) |
| Database | SQLite via SQLAlchemy |
| CLI | Typer + Rich |
| Resume Parsing | pdfminer, PyPDF2, python-docx |

> **Cost note:** Embeddings use a local model (no API cost). Only resume tailoring and outreach generation call the Anthropic API.

---

## Setup

### 1. Clone & install dependencies

```bash
pip install -r requirements.txt
playwright install chromium
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Required
ANTHROPIC_API_KEY=sk-ant-...

# Optional — enables JSearch (aggregates Indeed/Glassdoor/ZipRecruiter)
# Free tier: 200 requests/month at rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch
RAPIDAPI_KEY=your_key_here

# Optional — LinkedIn scraping (use at your own risk, see note below)
LINKEDIN_EMAIL=your@email.com
LINKEDIN_PASSWORD=yourpassword
```

### 3. Initialize your profile

```bash
python main.py init --resume resume.pdf --name "Jane Doe" --email "jane@email.com"
```

You will be prompted for skills, tech stack, target roles, and locations. This only needs to be run **once** — your profile persists in `data/candidate_profile.json`.

---

## Usage

### Full autonomous pipeline (recommended)

Runs all four stages automatically and processes the top 5 matching jobs:

```bash
python main.py run
```

Options:
```bash
python main.py run --sources jsearch,linkedin --top-n 3 --message-types linkedin_connection,cover_letter
```

---

### Step-by-step

#### Search for jobs
```bash
python main.py search --role "ML Engineer" --location "Remote"

# Search without immediately scoring (faster)
python main.py search --role "Backend Engineer" --no-match
```

#### Score all discovered jobs
```bash
python main.py match
```

#### View the dashboard
```bash
python main.py status

# Filter by minimum match score
python main.py status --min-score 0.7
```

Example output:
```
╭─────────────────────── Job Search Dashboard ───────────────────────╮
│ Jobs found: 47  |  Applications: 5  |  Response rate: 40%          │
╰─────────────────────────────────────────────────────────────────────╯

╭──────────────────────────── Top Jobs ──────────────────────────────╮
│ ID       │ Title              │ Company    │ Match │ ATS  │ Status  │
│ a3f9c1   │ ML Engineer        │ OpenAI     │  87%  │  82% │ ready   │
│ b72e4d   │ Senior ML Engineer │ Anthropic  │  84%  │  79% │ -       │
│ c91a2f   │ AI Engineer        │ Cohere     │  76%  │  71% │ applied │
╰─────────────────────────────────────────────────────────────────────╯
```

#### View full job details
```bash
python main.py view a3f9c1
```

#### Tailor resume for a job
```bash
python main.py tailor a3f9c1
```

Tailored resume saved to `data/applications/resume_OpenAI_ML_Engineer_<timestamp>.md`

#### Generate outreach messages
```bash
# LinkedIn connection request + cover letter (default)
python main.py outreach a3f9c1

# All message types
python main.py outreach a3f9c1 --types linkedin_connection,linkedin_inmail,cover_letter,email

# With recruiter's name for personalization
python main.py outreach a3f9c1 --recruiter "Sarah"
```

#### Add a job manually
```bash
# Auto-scrape from URL
python main.py add-job --url "https://company.com/careers/ml-engineer"

# Interactive manual entry
python main.py add-job
```

#### Record feedback (improves future matching)
```bash
python main.py feedback a3f9c1 --response yes --days 5
python main.py feedback b72e4d --response no
```

---

## Data & Privacy

All data stays local:

| Path | Contents |
|---|---|
| `data/candidate_profile.json` | Your profile (skills, experience, target roles) |
| `data/job_search.db` | All jobs and applications (SQLite) |
| `data/chroma/` | Vector embeddings of your resume (ChromaDB) |
| `data/applications/` | Tailored resumes and outreach messages |
| `data/resumes/` | Original resume copies |

---

## Configuration

All settings can be overridden in `.env`:

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | **Required.** Your Anthropic API key |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Claude model to use |
| `RAPIDAPI_KEY` | — | RapidAPI key for JSearch |
| `LINKEDIN_EMAIL` | — | LinkedIn login (optional) |
| `LINKEDIN_PASSWORD` | — | LinkedIn password (optional) |
| `MAX_JOBS_PER_SOURCE` | `25` | Max jobs fetched per source |
| `RELEVANCE_THRESHOLD` | `0.6` | Minimum score to qualify for tailoring |

---

## Notes on LinkedIn Scraping

LinkedIn's Terms of Service prohibit automated scraping. The LinkedIn scraper is provided for **personal, non-commercial use only** and may break if LinkedIn changes its frontend. Consider these alternatives:

- **JSearch (RapidAPI)** — free tier covers LinkedIn job listings without direct scraping
- **`python main.py add-job --url <url>`** — manually add any job URL
- **CSV/JSON import** — use `add-job --file jobs.json` to bulk-import

---

## Troubleshooting

| Error | Fix |
|---|---|
| `529 Overloaded` | Anthropic servers are busy — retry logic handles this automatically (up to 6 attempts) |
| `No profile found` | Run `python main.py init` first |
| `No jobs found` | Check your `.env` has `RAPIDAPI_KEY`, or try `--sources linkedin` |
| `Playwright not installed` | Run `playwright install chromium` |
| HuggingFace rate limit warning | Set `HF_TOKEN` in `.env` or ignore — downloads still work |
