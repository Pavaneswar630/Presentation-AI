"""Interactive Q&A and dynamic question-generation module using RAG context and LLM inference."""

import asyncio
import os
import re
import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Sequence

try:
    import httpx
except ImportError:
    httpx = None

from .indexer import FAISSHNSWIndexer
from .retriever import CoachingRetriever

logger = logging.getLogger("presentation_ai.rag_engine.question_generator")


def _transcript_text(value: Any) -> str:
    """Converts raw transcript text or timestamped word tokens to plain text."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return " ".join(
            word.strip()
            for item in value
            if (word := str(getattr(item, "word", item)).strip())
        )
    raise TypeError("transcript_text must be a string or a sequence of spoken-word tokens")


@dataclass(slots=True)
class PresentationQuestion:
    """An intelligent follow-up question generated for audience or evaluator Q&A."""
    question_text: str
    category: str  # Critical Thinking, Feasibility & Scalability, Clarification, Business Impact
    rationale: str  # Why an evaluator, investor, or audience member asks this
    suggested_answer_focus: str  # Tactical coaching drill on how to structure a strong answer
    difficulty: str = "probing"  # foundational, probing, challenging

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_text": self.question_text,
            "category": self.category,
            "rationale": self.rationale,
            "suggested_answer_focus": self.suggested_answer_focus,
            "difficulty": self.difficulty
        }


@dataclass(slots=True)
class LiveProbePrompt:
    """A real-speed probing question triggered in real-time during rehearsal practice."""
    prompt_text: str
    trigger_phrase: str
    context: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_text": self.prompt_text,
            "trigger_phrase": self.trigger_phrase,
            "context": self.context
        }


@dataclass(slots=True)
class QuestionGenerationResult:
    """Aggregated output from the question generator engine."""
    session_id: str
    transcript_summary: str
    targeted_questions: List[PresentationQuestion]
    live_probes: List[LiveProbePrompt]
    rag_context_used: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "transcript_summary": self.transcript_summary,
            "targeted_questions": [q.to_dict() for q in self.targeted_questions],
            "live_probes": [p.to_dict() for p in self.live_probes],
            "rag_context_used": self.rag_context_used
        }


class TranscriptSegmentAnalyzer:
    """Extracts central themes, numerical assertions, and core claims from presentation speech."""

    def __init__(self):
        # Patterns for quantifiable metrics (percentages, dollar amounts, multipliers, timelines)
        self._metric_regex = re.compile(
            r"(\b\d+(\.\d+)?%|\$\d+(\.\d+)?(\s?[kKmMbBtT]|(?:\s?million|\s?billion))?|\b\d+\s?(x|times|months|years|days|seconds|minutes)\b|\bincreased by\s?\d+|\breduced by\s?\d+)",
            re.IGNORECASE
        )
        # Patterns for assertive conviction/claims
        self._claim_indicators = [
            "we believe", "we will", "our strategy", "our solution", "we have proven",
            "we built", "the result is", "demonstrates", "outperforms", "key advantage",
            "competitive edge", "in order to scale", "fundamentally", "significantly"
        ]

    def analyze(self, transcript_text: Any) -> Dict[str, Any]:
        """Parses speech text and returns key concepts, numerical claims, and topic entities."""
        clean_text = _transcript_text(transcript_text)
        if not clean_text:
            return {
                "word_count": 0,
                "sentences": [],
                "metrics_claimed": [],
                "core_claims": [],
                "topics_detected": []
            }

        sentences = [s.strip() for s in re.split(r"[.!?]+", clean_text) if len(s.strip()) > 5]
        words = clean_text.split()

        # Extract quantifiable metrics
        metric_matches = self._metric_regex.findall(clean_text)
        metrics_claimed = [m[0] for m in metric_matches if isinstance(m, tuple) and m[0]]

        # Extract sentences with strong claims
        core_claims = []
        for s in sentences:
            s_lower = s.lower()
            if any(ind in s_lower for ind in self._claim_indicators) or any(m in s for m in metrics_claimed):
                core_claims.append(s)

        # Detect high-frequency topical keywords (excluding common filler/functional stopwords)
        stopwords = {
            "the", "a", "an", "is", "of", "and", "in", "to", "for", "with", "on", "at", "from",
            "by", "this", "that", "it", "as", "are", "was", "be", "or", "how", "what", "do",
            "does", "can", "should", "i", "you", "your", "my", "we", "our", "um", "uh", "like",
            "today", "going", "will", "just", "so", "have", "been", "there", "about"
        }
        word_freq: Dict[str, int] = {}
        for w in words:
            w_norm = re.sub(r"[^\w]", "", w.lower())
            if w_norm and len(w_norm) > 3 and w_norm not in stopwords:
                word_freq[w_norm] = word_freq.get(w_norm, 0) + 1

        sorted_topics = sorted(word_freq.items(), key=lambda item: item[1], reverse=True)
        top_topics = [t[0] for t in sorted_topics[:6]]

        return {
            "word_count": len(words),
            "sentences": sentences,
            "metrics_claimed": metrics_claimed[:8],
            "core_claims": core_claims[:6],
            "topics_detected": top_topics
        }


class LLMQuestionGenerator:
    """Asynchronous question generation engine combining transcript analysis with FAISS HNSW context."""

    def __init__(
        self,
        indexer: Optional[FAISSHNSWIndexer] = None,
        retriever: Optional[CoachingRetriever] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        provider: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        self.indexer = indexer
        self.retriever = retriever
        gemini_key = os.getenv("GEMINI_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        configured_provider = (provider or os.getenv("PRES_AI_LLM_PROVIDER") or "").lower()
        if configured_provider:
            self.provider = configured_provider
        elif gemini_key:
            self.provider = "gemini"
        elif openai_key:
            self.provider = "openai"
        elif api_key:
            self.provider = "gemini"
        else:
            self.provider = "offline"

        if self.provider not in {"gemini", "openai", "offline"}:
            raise ValueError("provider must be 'gemini', 'openai', or 'offline'")

        self.api_key = api_key or (
            gemini_key if self.provider == "gemini" else openai_key if self.provider == "openai" else None
        )
        self.model_name = model_name or os.getenv(
            "PRES_AI_LLM_MODEL",
            "gemini-1.5-flash" if self.provider == "gemini" else "gpt-4o-mini"
        )
        self.base_url = (
            base_url
            or os.getenv("PRES_AI_LLM_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
        )
        self.analyzer = TranscriptSegmentAnalyzer()

    async def generate_questions(
        self,
        session_id: str,
        transcript_text: Any,
        question_count: int = 3,
        difficulty: str = "probing",
        focus_area: Optional[str] = None
    ) -> QuestionGenerationResult:
        """Generates 3 targeted audience/evaluator Q&A questions and real-speed live practice prompts."""
        if not 1 <= question_count <= 10:
            raise ValueError("question_count must be between 1 and 10")
        if difficulty not in {"foundational", "probing", "challenging"}:
            raise ValueError("difficulty must be foundational, probing, or challenging")

        transcript_text = _transcript_text(transcript_text)

        # Step 1: Analyze Transcript Claims
        analysis = self.analyzer.analyze(transcript_text)
        topics = analysis["topics_detected"]
        claims = analysis["core_claims"]
        metrics = analysis["metrics_claimed"]

        # Step 2: Retrieve Domain Context from FAISS HNSW
        rag_context: List[Dict[str, Any]] = []
        if self.indexer is not None and (topics or focus_area):
            query = focus_area if focus_area else f"Presentation topics: {' '.join(topics)}"
            try:
                results = self.indexer.search(query=query, top_k=3)
                for chunk, score in results:
                    rag_context.append({
                        "chunk_id": chunk.chunk_id,
                        "topic": chunk.metadata.get("topic", "Coaching"),
                        "score": round(score, 3),
                        "text_snippet": chunk.text[:180] + ("..." if len(chunk.text) > 180 else "")
                    })
            except Exception:
                logger.exception("RAG context retrieval failed; continuing without retrieved context.")

        # Step 3: LLM Generation (External API or High-Precision Heuristic Fallback)
        if self.api_key and self.provider != "offline" and httpx is not None:
            try:
                questions, live_probes = await self._call_external_llm(
                    transcript_text=transcript_text,
                    analysis=analysis,
                    rag_context=rag_context,
                    question_count=question_count,
                    difficulty=difficulty
                )
            except Exception as ex:
                logger.warning(f"External LLM invocation failed ({ex}). Falling back to heuristic generator.")
                questions, live_probes = self._generate_heuristic_questions(
                    transcript_text=transcript_text,
                    analysis=analysis,
                    rag_context=rag_context,
                    question_count=question_count,
                    difficulty=difficulty
                )
        else:
            if self.api_key and httpx is None:
                logger.warning("httpx is unavailable; using offline question generation.")
            questions, live_probes = self._generate_heuristic_questions(
                transcript_text=transcript_text,
                analysis=analysis,
                rag_context=rag_context,
                question_count=question_count,
                difficulty=difficulty
            )

        summary = (
            f"Presentation covering {', '.join(topics[:3]) if topics else 'executive topics'}. "
            f"Analyzed {analysis['word_count']} words with {len(claims)} key claims."
        )

        return QuestionGenerationResult(
            session_id=session_id,
            transcript_summary=summary,
            targeted_questions=questions[:question_count],
            live_probes=live_probes,
            rag_context_used=rag_context
        )

    def generate_live_probe(self, recent_speech_snippet: Any) -> LiveProbePrompt:
        """Generates a rapid real-time probe during active speaking to test speaker agility."""
        recent_speech_snippet = _transcript_text(recent_speech_snippet)
        analysis = self.analyzer.analyze(recent_speech_snippet)
        metrics = analysis["metrics_claimed"]
        claims = analysis["core_claims"]

        if metrics:
            metric = metrics[0]
            return LiveProbePrompt(
                prompt_text=f"You just cited '{metric}'. What is the primary data source or baseline backing that figure?",
                trigger_phrase=metric,
                context="Quantitative Verification Drill"
            )
        elif claims:
            claim_snippet = claims[0][:60]
            return LiveProbePrompt(
                prompt_text=f"You asserted that '{claim_snippet}...'. How will you defend this against a skeptical competitor?",
                trigger_phrase=claim_snippet,
                context="Claim Defense Drill"
            )
        else:
            topic = analysis["topics_detected"][0] if analysis["topics_detected"] else "this strategy"
            return LiveProbePrompt(
                prompt_text=f"Can you give a concrete real-world example of how {topic} executes under pressure?",
                trigger_phrase=topic,
                context="Audience Clarity Drill"
            )

    async def _call_external_llm(
        self,
        transcript_text: str,
        analysis: Dict[str, Any],
        rag_context: List[Dict[str, Any]],
        question_count: int,
        difficulty: str
    ) -> Tuple[List[PresentationQuestion], List[LiveProbePrompt]]:
        """Makes an asynchronous request to Gemini or OpenAI-compatible endpoint."""
        prompt = (
            f"You are an executive presentation evaluator and audience Q&A simulator.\n"
            f"Speech Transcript:\n\"\"\"{transcript_text}\"\"\"\n\n"
            f"Key Claims: {json.dumps(analysis['core_claims'])}\n"
            f"Metrics Cited: {json.dumps(analysis['metrics_claimed'])}\n"
            f"RAG Coaching Context: {json.dumps([c['text_snippet'] for c in rag_context])}\n\n"
            f"Task:\n"
            f"Generate exactly {question_count} targeted Q&A questions at a '{difficulty}' difficulty level.\n"
            f"Use only these categories: Critical Thinking, Clarification, Feasibility, Business Impact.\n"
            f"Return a JSON object with a 'questions' array of objects with keys: "
            f"'question_text', 'category', 'rationale', 'suggested_answer_focus', 'difficulty'."
        )

        headers = {"Content-Type": "application/json"}
        if self.provider == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
            headers["x-goog-api-key"] = self.api_key or ""
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.3, "maxOutputTokens": 1200}
            }
        else:
            base_url = (self.base_url or "https://api.openai.com/v1").rstrip("/")
            url = base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"
            headers["Authorization"] = f"Bearer {self.api_key}"
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "Return only valid JSON matching the requested schema."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 1200,
                "response_format": {"type": "json_object"}
            }

        response_data = None
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=8.0) as client:
                    response = await client.post(url, json=payload, headers=headers)
                    response.raise_for_status()
                    response_data = response.json()
                break
            except httpx.HTTPStatusError as ex:
                if ex.response.status_code < 500 and ex.response.status_code != 429:
                    raise
                if attempt == 2:
                    raise
            except httpx.TransportError:
                if attempt == 2:
                    raise
            await asyncio.sleep(0.25 * (2 ** attempt))

        if self.provider == "gemini":
            raw_text = response_data["candidates"][0]["content"]["parts"][0]["text"]
        else:
            raw_text = response_data["choices"][0]["message"]["content"]

        parsed_text = raw_text.strip()
        fenced_match = re.search(r"```(?:json)?\s*(.*?)```", parsed_text, re.DOTALL | re.IGNORECASE)
        if fenced_match:
            parsed_text = fenced_match.group(1).strip()
        try:
            parsed = json.loads(parsed_text)
        except json.JSONDecodeError:
            json_match = re.search(r"\[.*\]", parsed_text, re.DOTALL)
            if not json_match:
                raise ValueError("Could not parse JSON from external LLM")
            parsed = json.loads(json_match.group(0))

        if isinstance(parsed, dict):
            parsed = parsed.get("questions", parsed.get("targeted_questions"))
        if not isinstance(parsed, list) or len(parsed) != question_count:
            raise ValueError(f"External LLM must return exactly {question_count} questions")

        questions = []
        for item in parsed:
            if not isinstance(item, dict) or not item.get("question_text"):
                raise ValueError("External LLM returned a question with an invalid schema")
            category = str(item.get("category") or "Critical Thinking")
            if category not in {"Critical Thinking", "Clarification", "Feasibility", "Business Impact"}:
                raise ValueError("External LLM returned an unsupported question category")
            questions.append(PresentationQuestion(
                question_text=str(item["question_text"]),
                category=category,
                rationale=str(item.get("rationale") or "Tests depth of knowledge"),
                suggested_answer_focus=str(
                    item.get("suggested_answer_focus") or "Cite verifiable empirical evidence"
                ),
                difficulty=str(item.get("difficulty") or difficulty)
            ))

        live_probe = self.generate_live_probe(transcript_text[-200:])
        return questions, [live_probe]

    def _generate_heuristic_questions(
        self,
        transcript_text: str,
        analysis: Dict[str, Any],
        rag_context: List[Dict[str, Any]],
        question_count: int,
        difficulty: str
    ) -> Tuple[List[PresentationQuestion], List[LiveProbePrompt]]:
        """Intelligent semantic question synthesizer for reliable, testable execution."""
        topics = analysis["topics_detected"]
        claims = analysis["core_claims"]
        metrics = analysis["metrics_claimed"]

        primary_topic = topics[0].capitalize() if topics else "Your Strategy"
        secondary_topic = topics[1].capitalize() if len(topics) > 1 else "Execution"

        questions: List[PresentationQuestion] = []

        # 1. Critical Thinking / Assumption Challenging Question
        if claims:
            claim_text = claims[0].strip().rstrip(".")
            q1 = PresentationQuestion(
                question_text=f"In your presentation, you stated that '{claim_text}'. What underlying assumptions must hold true for this to succeed, and what is your contingency if they fail?",
                category="Critical Thinking",
                rationale="Audience evaluators want to see if the speaker has stress-tested their thesis against market disruptions.",
                suggested_answer_focus="Acknowledge the core dependency, quantify the risk factor, and outline a 2-step mitigation plan.",
                difficulty=difficulty
            )
        else:
            q1 = PresentationQuestion(
                question_text=f"How does your framework for {primary_topic} account for edge cases where resources or team capacity are constrained?",
                category="Critical Thinking",
                rationale="Probes resilience and defensive depth of the presented proposal.",
                suggested_answer_focus="Outline operational prioritization rules and resource threshold triggers.",
                difficulty=difficulty
            )
        questions.append(q1)

        # 2. Feasibility & Scalability / Quantitative Question
        if metrics:
            metric_val = metrics[0]
            q2 = PresentationQuestion(
                question_text=f"You cited an achievement of {metric_val}. How was this metric benchmarked, and how linearly does this scale as volume increases by 10x?",
                category="Feasibility",
                rationale="Investors and technical evaluators immediately verify whether metrics are defensible or cherry-picked.",
                suggested_answer_focus="Explain the measurement methodology, state baseline figures, and discuss scaling bottlenecks.",
                difficulty=difficulty
            )
        else:
            q2 = PresentationQuestion(
                question_text=f"What are the primary operational bottlenecks when scaling {primary_topic} across diverse regional teams?",
                category="Feasibility",
                rationale="Tests if the speaker understands execution friction beyond the initial prototype.",
                suggested_answer_focus="Address infrastructure, compliance, and latency considerations.",
                difficulty=difficulty
            )
        questions.append(q2)

        # 3. Business Impact & Competitive Differentiation Question
        q3 = PresentationQuestion(
            question_text=f"If an executive stakeholder asks why we should invest in {primary_topic} over existing legacy solutions, what is the single decisive differentiator?",
            category="Business Impact",
            rationale="Audience members want clear strategic justification and payback timeline before approving initiatives.",
            suggested_answer_focus="Lead with the primary economic advantage (cost, speed, retention) before mentioning technical architecture.",
            difficulty=difficulty
        )
        questions.append(q3)

        additional_questions = [
            (
                f"What does success for {primary_topic} mean in measurable terms, and which baseline should the audience use?",
                "Clarification",
                "Clarifies the outcome and baseline so the audience can evaluate the core claim.",
                "Define the outcome, establish a starting point, and state how progress will be measured."
            ),
            (
                f"What evidence would convince a skeptical stakeholder that {primary_topic} is a better choice than {secondary_topic}?",
                "Critical Thinking",
                "Tests whether the proposal is supported by evidence rather than preference.",
                "Compare alternatives against measurable decision criteria and acknowledge trade-offs."
            ),
            (
                f"What is the first milestone for delivering {primary_topic}, and how will you know it is on track?",
                "Feasibility",
                "Checks whether the proposal has an actionable implementation path.",
                "Name an owner, a measurable milestone, and an early-warning indicator."
            ),
            (
                f"What is the most likely failure mode for {primary_topic}, and how quickly could the team detect it?",
                "Critical Thinking",
                "Probes operational readiness and the ability to limit downside.",
                "Describe the failure signal, response owner, and recovery or rollback plan."
            ),
            (
                f"Which customer or business outcome should improve first if {primary_topic} succeeds?",
                "Business Impact",
                "Connects the proposal to a concrete, observable business outcome.",
                "Link the leading indicator to a baseline, owner, and review timeframe."
            ),
            (
                f"What assumptions about adoption could prevent {primary_topic} from delivering its expected value?",
                "Feasibility",
                "Surfaces adoption constraints that can undermine projected impact.",
                "Identify the adoption barrier and explain how usage and value realization will be measured."
            ),
            (
                f"How would your recommendation for {primary_topic} change if the available budget were reduced by 20%?",
                "Business Impact",
                "Tests prioritization and resilience under a realistic constraint.",
                "Protect the highest-value capability and explain the impact of deferred work."
            ),
            (
                f"What new evidence would cause you to revise your position on {primary_topic}?",
                "Critical Thinking",
                "Assesses openness to disconfirming evidence and sound decision-making.",
                "State a falsifiable assumption, a decision threshold, and the next review point."
            )
        ]
        for question_text, category, rationale, answer_focus in additional_questions:
            questions.append(PresentationQuestion(
                question_text=question_text,
                category=category,
                rationale=rationale,
                suggested_answer_focus=answer_focus,
                difficulty=difficulty
            ))

        # Live Probes for Real-Time Practice
        probes = [
            self.generate_live_probe(transcript_text[-300:] if len(transcript_text) > 300 else transcript_text),
            LiveProbePrompt(
                prompt_text=f"Quick probe: How does {secondary_topic} directly link to client ROI?",
                trigger_phrase=secondary_topic,
                context="Fast-Paced Rehearsal Interjection"
            )
        ]

        return questions[:question_count], probes
