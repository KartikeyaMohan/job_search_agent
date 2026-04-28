"""ATS (Applicant Tracking System) scoring using keyword extraction and TF-IDF similarity."""
import re
import math
from collections import Counter
from typing import NamedTuple

# Common stop words to ignore during keyword extraction
_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must", "you", "we",
    "they", "it", "this", "that", "these", "those", "i", "he", "she",
    "our", "your", "their", "its", "about", "as", "if", "then", "than",
    "so", "such", "not", "no", "more", "also", "any", "all", "each",
    "both", "few", "other", "some", "what", "which", "who", "how", "when",
    "where", "why", "very", "just", "up", "out", "into", "through", "during",
    "including", "work", "working", "use", "using", "used", "ability",
    "strong", "excellent", "good", "great", "new", "high", "large", "small",
}

# Tech/role keyword boosts – these matter more for ATS
_TECH_PATTERNS = re.compile(
    r"\b("
    r"python|javascript|typescript|java|golang|go|rust|c\+\+|c#|ruby|php|scala|kotlin|swift"
    r"|react|angular|vue|nextjs|nuxt|svelte"
    r"|node\.?js|express|fastapi|django|flask|spring|rails|laravel"
    r"|postgresql|postgres|mysql|mongodb|redis|elasticsearch|cassandra|dynamodb|sqlite"
    r"|aws|gcp|azure|docker|kubernetes|k8s|terraform|ansible|jenkins|ci/cd|github.?actions"
    r"|machine.?learning|deep.?learning|nlp|llm|gpt|transformers|pytorch|tensorflow|keras"
    r"|langchain|llamaindex|openai|anthropic|hugging.?face|vector.?store|rag|embedding"
    r"|git|linux|unix|bash|sql|nosql|graphql|rest|api|microservices|kafka|rabbitmq"
    r"|agile|scrum|jira|ci|cd|devops|mlops|data.?engineer|data.?scientist|ml.?engineer"
    r"|pandas|numpy|scikit.?learn|spark|airflow|dbt|tableau|power.?bi"
    r")\b",
    re.IGNORECASE,
)


class ATSResult(NamedTuple):
    score: float                     # 0.0 – 1.0
    matched_keywords: list[str]
    missing_keywords: list[str]
    keyword_coverage: float          # fraction of JD keywords found in resume
    tfidf_similarity: float          # cosine similarity between JD and resume


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stop words."""
    tokens = re.findall(r"\b[a-z][a-z0-9+#./]*\b", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


def _extract_tech_keywords(text: str) -> list[str]:
    """Pull out specific tech/role keywords (weighted higher)."""
    matches = _TECH_PATTERNS.findall(text.lower())
    # Normalize compound matches
    return list({re.sub(r"[.\s/]", "", m) for m in matches})


def _tfidf_cosine(text_a: str, text_b: str) -> float:
    """Compute TF-IDF cosine similarity between two texts."""
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    if not tokens_a or not tokens_b:
        return 0.0

    vocab = list(set(tokens_a) | set(tokens_b))

    def tfidf_vec(tokens: list[str]) -> list[float]:
        tf = Counter(tokens)
        total = len(tokens)
        vec = []
        for word in vocab:
            tf_val = tf[word] / total
            # IDF: log((1+N)/(1+df)) + 1, simplified for 2-doc corpus
            df = 1 if word in set(tokens_a) and word in set(tokens_b) else 0
            idf = math.log((1 + 2) / (1 + df)) + 1
            vec.append(tf_val * idf)
        return vec

    va = tfidf_vec(tokens_a)
    vb = tfidf_vec(tokens_b)
    dot = sum(a * b for a, b in zip(va, vb))
    mag_a = math.sqrt(sum(a * a for a in va))
    mag_b = math.sqrt(sum(b * b for b in vb))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def score_resume_against_jd(resume_text: str, jd_text: str) -> ATSResult:
    """
    Compute ATS match score between a resume and job description.

    Scoring breakdown:
      - 60% keyword coverage (tech keywords from JD found in resume)
      - 40% TF-IDF cosine similarity
    """
    # Extract tech keywords from JD
    jd_tech = set(_extract_tech_keywords(jd_text))
    resume_tech = set(_extract_tech_keywords(resume_text))

    # General token keywords from JD (non-stop words, min freq 2)
    jd_tokens = _tokenize(jd_text)
    jd_token_freq = Counter(jd_tokens)
    # Keywords = tech terms + general terms appearing 2+ times
    jd_keywords = jd_tech | {t for t, c in jd_token_freq.items() if c >= 2 and len(t) > 3}

    if not jd_keywords:
        return ATSResult(0.0, [], [], 0.0, 0.0)

    resume_tokens_set = set(_tokenize(resume_text)) | resume_tech

    matched = sorted(jd_keywords & resume_tokens_set)
    missing = sorted(jd_keywords - resume_tokens_set)
    keyword_coverage = len(matched) / len(jd_keywords)
    tfidf_sim = _tfidf_cosine(resume_text, jd_text)

    # Weighted score
    score = min(1.0, 0.6 * keyword_coverage + 0.4 * tfidf_sim)

    return ATSResult(
        score=round(score, 4),
        matched_keywords=matched[:30],
        missing_keywords=missing[:30],
        keyword_coverage=round(keyword_coverage, 4),
        tfidf_similarity=round(tfidf_sim, 4),
    )


def top_missing_keywords(missing: list[str], jd_text: str, n: int = 15) -> list[str]:
    """Return the most impactful missing keywords, prioritising tech terms."""
    tech_missing = [k for k in missing if _TECH_PATTERNS.match(k)]
    other_missing = [k for k in missing if k not in tech_missing]
    combined = tech_missing + other_missing
    return combined[:n]
