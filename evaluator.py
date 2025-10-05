from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field, field_validator
from models import JSONResume, EvaluationData
from llm_utils import initialize_llm_provider, extract_json_from_response
import logging
import json
import re

MAX_BONUS_POINTS = 20
MIN_FINAL_SCORE = -20
MAX_FINAL_SCORE = 120

from prompt import (
    DEFAULT_MODEL,
    MODEL_PARAMETERS,
    MODEL_PROVIDER_MAPPING,
    GEMINI_API_KEY,
)
from prompts.template_manager import TemplateManager

logger = logging.getLogger(__name__)


class ResumeEvaluator:
    def __init__(self, model_name: str = DEFAULT_MODEL, model_params: dict = None):
        if not model_name:
            raise ValueError("Model name cannot be empty")

        self.model_name = model_name
        self.model_params = model_params or MODEL_PARAMETERS.get(
            model_name, {"temperature": 0.5, "top_p": 0.9}
        )
        self.template_manager = TemplateManager()
        self._initialize_llm_provider()

    def _initialize_llm_provider(self):
        """Initialize the appropriate LLM provider based on the model."""
        self.provider = initialize_llm_provider(self.model_name)

    def _load_evaluation_prompt(self, resume_text: str) -> str:
        criteria_template = self.template_manager.render_template(
            "resume_evaluation_criteria", text_content=resume_text
        )
        if criteria_template is None:
            raise ValueError("Failed to load resume evaluation criteria template")
        return criteria_template

    def _extract_blog_from_response(self, response_text: str) -> dict:
        """Try flexible extraction of blog_analysis/blog_score from raw response_text."""
        out = {}
        if not response_text:
            return out

        # JSON-style (multiline) blog_analysis and blog_score
        m = re.search(r'"blog_analysis"\s*:\s*"(.+?)"', response_text, re.IGNORECASE | re.DOTALL)
        if m:
            out["blog_analysis"] = m.group(1).strip()

        m2 = re.search(r'"blog_score"\s*:\s*([0-9]+(?:\.[0-9]+)?)', response_text, re.IGNORECASE)
        if m2:
            out["blog_score"] = float(m2.group(1))

        # Non-JSON pattern: "📝 Blog: 5.0/5" or "Blog: 5/5"
        m3 = re.search(r'📝\s*Blog\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*5', response_text)
        if not m3:
            m3 = re.search(r'Blog\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*5', response_text, re.IGNORECASE)
        if m3 and "blog_score" not in out:
            out["blog_score"] = float(m3.group(1))

        # Capture a short analysis following "Detected" lines or "Blog:" paragraph
        if "detected" in response_text.lower() and "blog_analysis" not in out:
            m4 = re.search(r'(Detected.+?blog.+?\.?.{0,200})', response_text, re.IGNORECASE | re.DOTALL)
            if m4:
                out["blog_analysis"] = m4.group(1).strip()

        # Fallback: find a "Blog" heading then next paragraph
        if "blog analysis" in response_text.lower() and "blog_analysis" not in out:
            m5 = re.search(r'(?:blog analysis[:\n\r]+)(.+?)(?:\n\s*\n|$)', response_text, re.IGNORECASE | re.DOTALL)
            if m5:
                out["blog_analysis"] = m5.group(1).strip()

        return out

    def evaluate_resume(self, resume_text: str) -> EvaluationData:
        self._last_resume_text = resume_text
        full_prompt = self._load_evaluation_prompt(resume_text)
        # logger.info(f"🔤 Evaluation prompt being sent: {full_prompt}")
        try:
            system_message = self.template_manager.render_template(
                "resume_evaluation_system_message"
            )
            if system_message is None:
                raise ValueError(
                    "Failed to load resume evaluation system message template"
                )

            # Prepare chat parameters
            chat_params = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": full_prompt},
                ],
                "options": {
                    "stream": False,
                    "temperature": self.model_params.get("temperature", 0.5),
                    "top_p": self.model_params.get("top_p", 0.9),
                },
            }

            # Add format parameter for structured output
            kwargs = {"format": EvaluationData.model_json_schema()}
            # Use the appropriate provider to make the API call
            response = self.provider.chat(**chat_params, **kwargs)

            # Keep the raw LLM response for flexible extraction, then extract JSON separately
            raw_response_text = response["message"]["content"]
            json_text = extract_json_from_response(raw_response_text)
            logger.error(f"🔤 Prompt raw response: {raw_response_text}")
            logger.error(f"🔤 Extracted JSON: {json_text}")

            evaluation_dict = json.loads(json_text)

            # flexible extraction: capture informal blog analysis if present in the raw response
            try:
                blog_fields = self._extract_blog_from_response(raw_response_text)
                if blog_fields:
                    if blog_fields.get("blog_analysis"):
                        evaluation_dict["blog_analysis"] = blog_fields["blog_analysis"]
                    if blog_fields.get("blog_score") is not None:
                        try:
                            score_val = int(round(float(blog_fields["blog_score"])))
                        except Exception:
                            score_val = 0
                        score_val = max(0, min(5, score_val))
                        evaluation_dict["blog_score"] = score_val

                        # Ensure scores -> technical_skills exists and attach nested blog object
                        scores = evaluation_dict.setdefault("scores", {})
                        tech = scores.setdefault("technical_skills", {})
                        tech["blog"] = {
                            "score": score_val,
                            "max": 5,
                            "evidence": blog_fields.get("blog_analysis", "").strip(),
                        }
            except Exception:
                pass

            evaluation_data = EvaluationData(**evaluation_dict)

            return evaluation_data

        except Exception as e:
            logger.error(f"Error evaluating resume: {str(e)}")
            raise
