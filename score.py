import os
import sys
import json
import logging
import csv
import requests
import feedparser
from pdf import PDFHandler
from github import fetch_and_display_github_info
from models import JSONResume, EvaluationData
from typing import List, Optional, Dict
from evaluator import ResumeEvaluator
from pathlib import Path
from prompt import DEFAULT_MODEL, MODEL_PARAMETERS
from transform import (
    transform_evaluation_response,
    convert_json_resume_to_text,
    convert_github_data_to_text,
    convert_blog_data_to_text,
)
from config import DEVELOPMENT_MODE
from urllib.parse import urlparse
from types import SimpleNamespace

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)5s - %(lineno)5d - %(funcName)33s - %(levelname)5s - %(message)s",
)


def print_evaluation_results(
    evaluation: EvaluationData, candidate_name: str = "Candidate"
):
    """Print evaluation results in a readable format."""
    print("\n" + "=" * 80)
    print(f"📊 RESUME EVALUATION RESULTS FOR: {candidate_name}")
    print("=" * 80)

    if not evaluation:
        print("❌ No evaluation data available")
        return

    # Calculate raw totals
    total_score = 0.0
    max_score = 0.0

    # accumulate category scores (from evaluation.scores)
    if hasattr(evaluation, "scores") and evaluation.scores:
        for category_name, category_data in evaluation.scores.model_dump().items():
            score_val = float(category_data.get("score", 0))
            max_val = float(category_data.get("max", 0))
            # Treat technical_blog_writing as an additive (contribution) but NOT part of the 100-point denominator
            # if category_name == "technical_blog_writing":
            #     total_score += score_val
            #     continue
            total_score += score_val
            max_score += max_val

    # no separate blog_score handling needed: blog lives under scores.technical_blog_writing

    # Add bonus points (include their max if provided)
    if hasattr(evaluation, "bonus_points") and evaluation.bonus_points:
        # add bonus to the numerator only (do not change denominator)
        total_score += float(getattr(evaluation.bonus_points, "total", 0))

    # Subtract deductions
    if hasattr(evaluation, "deductions") and evaluation.deductions:
        total_score -= float(getattr(evaluation.deductions, "total", 0))

    # Normalize to 100-point scale for display
    normalized_overall = 0.0
    if max_score > 0:
        normalized_overall = (total_score / max_score) * 100.0

    print(f"\n🎯 OVERALL SCORE: {normalized_overall:.1f}/100")

    # Detailed Scores
    print("\n📈 DETAILED SCORES:")
    print("-" * 60)

    if hasattr(evaluation, "scores") and evaluation.scores:
        # Open Source
        if hasattr(evaluation.scores, "open_source") and evaluation.scores.open_source:
            os_score = evaluation.scores.open_source
            print(f"🌐 Open Source:          {os_score.score}/{os_score.max}")
            print(f"   Evidence: {os_score.evidence}")
            print()

        # Self Projects
        if (
            hasattr(evaluation.scores, "self_projects")
            and evaluation.scores.self_projects
        ):
            sp_score = evaluation.scores.self_projects
            print(f"🚀 Self Projects:        {sp_score.score}/{sp_score.max}")
            print(f"   Evidence: {sp_score.evidence}")
            print()

        # Production Experience
        if hasattr(evaluation.scores, "production") and evaluation.scores.production:
            prod_score = evaluation.scores.production
            print(f"🏢 Production Experience: {prod_score.score}/{prod_score.max}")
            print(f"   Evidence: {prod_score.evidence}")
            print()

        # Technical Skills
        if (
            hasattr(evaluation, "scores") and evaluation.scores
            and hasattr(evaluation.scores, "technical_skills")
            and evaluation.scores.technical_skills
        ):
            tech_score = evaluation.scores.technical_skills
            print(f"💻 Technical Skills:     {tech_score.score}/{tech_score.max}")
            print(f"   Evidence: {tech_score.evidence}")
            print()


        # Blog Writing
        if (
            hasattr(evaluation, "scores") and evaluation.scores
            and hasattr(evaluation.scores, "technical_blog_writing")
            and evaluation.scores.technical_blog_writing
        ):
            blog_score = evaluation.scores.technical_blog_writing
            print(f" Blog Writing:     {blog_score.score}/{blog_score.max}")
            print(f"   Evidence: {blog_score.evidence}")
            print()



    # Bonus Points
    if hasattr(evaluation, "bonus_points") and evaluation.bonus_points:
        print(f"\n⭐ BONUS POINTS: {evaluation.bonus_points.total}")
        print("-" * 30)
        print(f"   {evaluation.bonus_points.breakdown}")

    # Deductions
    if (
        hasattr(evaluation, "deductions")
        and evaluation.deductions
        and getattr(evaluation.deductions, "total", 0) > 0
    ):
        print(f"\n⚠️  DEDUCTIONS: -{evaluation.deductions.total}")
        print("-" * 30)
        if getattr(evaluation.deductions, "reasons", None):
            print(f"   {evaluation.deductions.reasons}")

    # Key Strengths
    if hasattr(evaluation, "key_strengths") and evaluation.key_strengths:
        print(f"\n✅ KEY STRENGTHS:")
        print("-" * 30)
        for i, strength in enumerate(evaluation.key_strengths, 1):
            print(f"  {i}. {strength}")

    # Areas for Improvement
    if (
        hasattr(evaluation, "areas_for_improvement")
        and evaluation.areas_for_improvement
    ):
        print(f"\n🔧 AREAS FOR IMPROVEMENT:")
        print("-" * 30)
        for i, area in enumerate(evaluation.areas_for_improvement, 1):
            print(f"  {i}. {area}")

    # Blog Analysis block already printed above if present
    print("\n" + "=" * 80)


def find_blog_url_in_profiles(profiles):
    if not profiles:
        return None
    for p in profiles:
        url = p.get("url") if isinstance(p, dict) else getattr(p, "url", None)
        username = (p.get("username") if isinstance(p, dict) else getattr(p, "username", None)) or ""
        network = (p.get("network") if isinstance(p, dict) else getattr(p, "network", None)) or ""
        # direct label match
        if network and "blog" in network.lower():
            return url
        # domain detection
        if url and any(d in url.lower() for d in ("hashnode", "medium.com", "dev.to", "blog.")):
            return url
        # username that looks like host
        if username and any(d in username.lower() for d in ("hashnode", "medium", "dev", "blog")):
            if username.startswith("http"):
                return username
    return None


def fetch_blog_data_from_url(url, max_posts=10, timeout=8):
    """Try RSS first, fallback to feedparser on url, produce normalized dict."""
    if not url:
        return None
    try:
        # try common rss endpoints
        candidates = [url.rstrip("/") + p for p in ("/rss.xml", "/feed.xml", "/rss", "/atom.xml", "/feeds/latest")]
        feed = None
        for c in candidates:
            try:
                f = feedparser.parse(c)
                if getattr(f, "entries", None):
                    feed = f
                    source = c
                    break
            except Exception:
                continue
        if not feed:
            f = feedparser.parse(url)
            if getattr(f, "entries", None):
                feed = f
                source = url
        posts = []
        if feed and getattr(feed, "entries", None):
            for e in feed.entries[:max_posts]:
                posts.append(
                    {
                        "title": e.get("title", ""),
                        "url": e.get("link", ""),
                        "summary": e.get("summary", "") or e.get("description", ""),
                        "published": e.get("published", "") or e.get("updated", ""),
                    }
                )
            return {"source": source, "count": len(posts), "posts": posts}
    except Exception:
        pass

    # fallback: do a simple GET and capture first few links/titles
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(r.text, "html.parser")
        # heuristics: article links or h2/h3 titles linking to posts
        links = []
        for a in soup.select("a[href]"):
            href = a["href"]
            text = a.get_text(" ", strip=True)
            if ("/post/" in href or "/blog/" in href or href.endswith(".html")) and href not in links:
                full = href if href.startswith("http") else requests.compat.urljoin(url, href)
                links.append((text, full))
            if len(links) >= max_posts:
                break
        posts = [{"title": t or u, "url": u, "excerpt": ""} for t, u in links]
        return {"source": url, "count": len(posts), "posts": posts}
    except Exception:
        return None


def _evaluate_resume(
    resume_data: JSONResume, github_data: dict = None, blog_data: dict = None
) -> Optional[EvaluationData]:
    """Evaluate the resume using AI and display results."""
    model_params = MODEL_PARAMETERS.get(DEFAULT_MODEL)
    evaluator = ResumeEvaluator(model_name=DEFAULT_MODEL, model_params=model_params)

    # Convert JSON resume data to text
    resume_text = convert_json_resume_to_text(resume_data)

    # Add GitHub data if available
    if github_data:
        github_text = convert_github_data_to_text(github_data)
        resume_text += github_text

    # Add blog data if available
    if blog_data:
        blog_text = convert_blog_data_to_text(blog_data)
        resume_text += blog_text

    # Evaluate the enhanced resume
    evaluation_result = evaluator.evaluate_resume(resume_text)

    # attach blog_data so print_evaluation_results can show it later
    if blog_data:
        try:
            evaluation_result.blog_data = blog_data
        except Exception:
            pass

    return evaluation_result


def find_profile(profiles, network):
    if not profiles:
        return None
    return next(
        (p for p in profiles if p.network and p.network.lower() == network.lower()),
        None,
    )


def main(pdf_path):
    # Create cache filename based on PDF path
    cache_filename = (
        f"cache/resumecache_{os.path.basename(pdf_path).replace('.pdf', '')}.json"
    )
    github_cache_filename = (
        f"cache/githubcache_{os.path.basename(pdf_path).replace('.pdf', '')}.json"
    )
    blog_cache_filename = (
        f"cache/blogcache_{os.path.basename(pdf_path).replace('.pdf', '')}.json"
    )

    # Check if cache exists and we're in development mode
    if DEVELOPMENT_MODE and os.path.exists(cache_filename):
        print(f"Loading cached data from {cache_filename}")
        cached_data = json.loads(Path(cache_filename).read_text())
        resume_data = JSONResume(**cached_data)
    else:
        logger.debug(
            f"Extracting data from PDF"
            + (" and caching to " + cache_filename if DEVELOPMENT_MODE else "")
        )
        pdf_handler = PDFHandler()
        resume_data = pdf_handler.extract_json_from_pdf(pdf_path)
        if DEVELOPMENT_MODE:
            os.makedirs(os.path.dirname(cache_filename), exist_ok=True)
            Path(cache_filename).write_text(
                json.dumps(resume_data.model_dump(), indent=2, ensure_ascii=False)
            )

    # GitHub data fetching (unchanged)
    github_data = {}
    if DEVELOPMENT_MODE and os.path.exists(github_cache_filename):
        print(f"Loading cached data from {github_cache_filename}")
        github_data = json.loads(Path(github_cache_filename).read_text())
    else:
        print(
            f"Fetching GitHub data"
            + (" and caching to " + github_cache_filename if DEVELOPMENT_MODE else "")
        )

        profiles = []
        if resume_data and hasattr(resume_data, "basics") and resume_data.basics:
            profiles = resume_data.basics.profiles or []
        github_profile = find_profile(profiles, "Github")

        if github_profile:
            github_data = fetch_and_display_github_info(github_profile.url)
        if DEVELOPMENT_MODE:
            os.makedirs(os.path.dirname(github_cache_filename), exist_ok=True)
            Path(github_cache_filename).write_text(
                json.dumps(github_data, indent=2, ensure_ascii=False)
            )

    # BLOG data: load cache if present, else detect blog url and fetch
    blog_data = None
    if DEVELOPMENT_MODE and os.path.exists(blog_cache_filename):
        print(f"Loading cached blog data from {blog_cache_filename}")
        try:
            blog_data = json.loads(Path(blog_cache_filename).read_text())
        except Exception:
            blog_data = None
    else:
        profiles = []
        if resume_data and getattr(resume_data, "basics", None):
            profiles = getattr(resume_data.basics, "profiles", []) or []
        blog_url = find_blog_url_in_profiles(profiles)
        if blog_url:
            print(f"Fetching blog data from {blog_url}")
            blog_data = fetch_blog_data_from_url(blog_url)
            if DEVELOPMENT_MODE and blog_data:
                os.makedirs(os.path.dirname(blog_cache_filename), exist_ok=True)
                Path(blog_cache_filename).write_text(
                    json.dumps(blog_data, indent=2, ensure_ascii=False)
                )

    score = _evaluate_resume(resume_data, github_data, blog_data)

    # Get candidate name for display
    candidate_name = os.path.basename(pdf_path).replace(".pdf", "")
    if (
        resume_data
        and hasattr(resume_data, "basics")
        and resume_data.basics
        and resume_data.basics.name
    ):
        candidate_name = resume_data.basics.name

    # Print evaluation results in readable format
    print_evaluation_results(score, candidate_name)

    if DEVELOPMENT_MODE:
        csv_row = transform_evaluation_response(
            file_name=os.path.basename(pdf_path),
            evaluation=score,
            resume_data=resume_data,
            github_data=github_data,
        )

        # Write CSV row to file
        csv_path = "resume_evaluations.csv"
        file_exists = os.path.exists(csv_path)

        with open(csv_path, "a", newline="", encoding="utf-8") as csvfile:
            fieldnames = list(csv_row.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write headers if file doesn't exist
            if not file_exists:
                writer.writeheader()

            # Write the row
            writer.writerow(csv_row)

    return score


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python score.py <pdf_path>")
        exit(1)
    pdf_path = sys.argv[1]

    if not os.path.exists(pdf_path):
        print(f"Error: File '{pdf_path}' does not exist.")
        exit(1)

    main(pdf_path)
