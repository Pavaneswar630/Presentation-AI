"""Synthetic and benchmark corpus generator for 1,000 presentation coaching transcripts."""

import random
from typing import List, Tuple, Dict, Any
from .chunker import TranscriptDocument


class TranscriptCorpusGenerator:
    """Generates a domain-specific corpus of 1,000 speech coaching transcripts with ground-truth test queries."""

    COACHING_DOMAINS = [
        {
            "topic": "Upper Torso Alignment & Spine Posture",
            "keywords": ["spine", "shoulders", "slouching", "upright", "plumb-line", "thoracic", "back", "posture"],
            "coaching_templates": [
                "When presenting from a podium or standing on stage, maintain a neutral spine plumb-line. Align your shoulder girdle directly over your hip axis to project authority and prevent vocal compression.",
                "Avoid dropping one shoulder while leaning on a single hip. Asymmetric shoulder tilt projects unease and destabilizes your diaphragmatic breathing apparatus during keynotes.",
                "Combat slumping by consciously rolling your shoulder blades downward toward your back pockets. This opens your thoracic chest cavity and increases vocal projection and presence by up to twenty percent.",
                "Keep your chin level with the horizon and spine upright. Proper spinal alignment directly improves diaphragmatic breath support, vocal projection, and stage presence."
            ],
            "query_templates": [
                "How do I prevent slouching and keep my spine upright while presenting?",
                "What is the best way to correct uneven shoulder tilt during a speech?",
                "Techniques for maintaining good posture and chest alignment on stage.",
                "How does spinal alignment affect vocal projection and presence?"
            ]
        },
        {
            "topic": "Cadence, Pacing & Deliberate Pausing",
            "keywords": ["cadence", "words per minute", "pacing", "pause", "speed", "rushing", "tempo"],
            "coaching_templates": [
                "The optimal presentation speaking cadence is between 130 and 160 words per minute. Exceeding 170 WPM overwhelms listeners and induces cognitive overload during technical briefings.",
                "Use the deliberate two-second pause before delivering a pivotal statistic or punchline. Strategic silence focuses audience attention and allows complex ideas to settle.",
                "When transitioning between slides, stop speaking for a full breath cycle. Never talk over your slide transition animations as it makes you appear hurried and undisciplined.",
                "Tips for regulating speech rate and avoiding fast talking include anchoring your pacing to one key idea per breath cycle and varying your speaking tempo dynamically."
            ],
            "query_templates": [
                "What is the optimal speaking cadence in words per minute for a keynote?",
                "How to stop rushing through slides and manage speech pacing?",
                "When should a speaker use deliberate silence and pausing?",
                "Tips for regulating speech rate and avoiding fast talking."
            ]
        },
        {
            "topic": "Eliminating Verbal Fillers & Speech Disfluencies",
            "keywords": ["fillers", "um", "uh", "like", "you know", "disfluency", "crutch words"],
            "coaching_templates": [
                "Verbal fillers and crutch words such as 'um', 'uh', and 'you know' typically arise when your vocal apparatus attempts to fill silence while your brain searches for the next concept.",
                "Replace vocalized fillers with silent pauses. A two-second silence conveys deep executive gravitas, whereas repeating crutch words undermines professional credibility.",
                "To reduce filler frequency below two percent, practice chunking your thoughts into discrete seven-word phrases and consciously swallow at the comma.",
                "Why do speakers use crutch words? Insecurity with silence is the primary reason. Track crutch words by recording your rehearsals and mapping your filler triggers."
            ],
            "query_templates": [
                "How to eliminate filler words like um, uh, and like during presentations?",
                "Techniques to replace verbal fillers with confident silent pauses.",
                "Why do speakers use crutch words and how can I track them?",
                "Best practices for reducing disfluency rate below two percent."
            ]
        },
        {
            "topic": "Hand Gestures, Micro-Movements & Kinetic Jitter",
            "keywords": ["hands", "gestures", "fidgeting", "wrists", "kinetic", "podium", "movement"],
            "coaching_templates": [
                "Anchor your hands in the neutral readiness zone between your navel and sternum. Avoid clasping your hands in the fig-leaf position or tucking them into pockets.",
                "Use open-palm gestures when presenting solutions. Showing your palms triggers involuntary subconscious trust mechanisms in executive decision-makers.",
                "Eliminate kinetic jitter and repetitive hand fidgeting. Rapid wrist twitching and ring-twisting distract listeners and signal internal acute stress.",
                "Scale your gesture amplitude to your audience size. In a boardroom, keep gestures within your torso silhouette; on a convention hall stage, extend past your shoulders."
            ],
            "query_templates": [
                "Where should I place my hands when speaking publicly?",
                "How do I eliminate nervous hand fidgeting and pocket fiddling?",
                "What do open palm gestures communicate to an executive audience?",
                "How to use purposeful hand gestures rather than repetitive jitter."
            ]
        },
        {
            "topic": "Vocal Pitch Modulation & Verbal Clarity",
            "keywords": ["vocal", "pitch", "monotone", "articulation", "inflection", "up-talk", "resonance"],
            "coaching_templates": [
                "Avoid uptalk—the habit of ending declarative statements with a rising question inflection. Practice ending declarative sentences with resolute vocal confidence and downward inflection.",
                "Techniques to eliminate uptalk and end statements with confidence require practicing dropping your pitch at the final syllable of every sentence.",
                "Warm up your vocal cords with lip trills and tongue twisters prior to speaking. Proper articulatory precision prevents mumbling and slurred diction.",
                "Breathe from your diaphragm rather than your upper chest. Deep diaphragmatic breathing provides continuous breath support and lowers nervous vocal pitch."
            ],
            "query_templates": [
                "How to avoid a monotone delivery and add vocal inflection?",
                "Techniques to eliminate uptalk and end statements with confidence.",
                "Vocal warm-ups for articulation and diction before a keynote.",
                "How does diaphragmatic breathing improve voice pitch and resonance?"
            ]
        }
    ]

    SPEAKERS = [
        "Dr. Elena Vance (Executive Speech Coach)",
        "Marcus Sterling (Keynote Speaker)",
        "Aisha Patel (Leadership Communications Consultant)",
        "David O'Connor (TEDx Presentation Mentor)",
        "Sarah Lin (Corporate Rhetoric Specialist)"
    ]

    @classmethod
    def generate_corpus(cls, count: int = 1000, seed: int = 42) -> Tuple[List[TranscriptDocument], List[Dict[str, Any]]]:
        """Generates `count` structured transcripts with annotated ground-truth test queries."""
        rng = random.Random(seed)
        documents: List[TranscriptDocument] = []
        evaluation_pairs: List[Dict[str, Any]] = []

        for i in range(count):
            domain_idx = i % len(cls.COACHING_DOMAINS)
            domain = cls.COACHING_DOMAINS[domain_idx]
            speaker = cls.SPEAKERS[i % len(cls.SPEAKERS)]
            transcript_id = f"transcript_{i+1:04d}"

            # Assemble realistic coaching transcript paragraph
            core_tip = domain["coaching_templates"][i % len(domain["coaching_templates"])]
            secondary_tip = domain["coaching_templates"][(i + 1) % len(domain["coaching_templates"])]
            keywords = ", ".join(domain["keywords"])

            content = (
                f"Transcript ID {transcript_id}: Coaching Module on {domain['topic']}. "
                f"Instructor: {speaker}. Key focus areas include {keywords}. "
                f"Diagnostic Guidance: {core_tip} "
                f"Furthermore, in high-stakes presentation environments: {secondary_tip} "
                f"Remember that non-verbal body language and speech delivery form sixty percent "
                f"of audience retention. Practice these diagnostic drills daily to internalize optimal muscle memory."
            )

            doc = TranscriptDocument(
                transcript_id=transcript_id,
                title=f"Coaching Session {i+1:04d}: {domain['topic']}",
                speaker=speaker,
                topic=domain["topic"],
                content=content,
                metadata={
                    "domain_index": domain_idx,
                    "session_number": i + 1,
                    "keywords": domain["keywords"]
                }
            )
            documents.append(doc)

            # Ground truth query for evaluation
            query = domain["query_templates"][i % len(domain["query_templates"])]
            evaluation_pairs.append({
                "query": query,
                "target_transcript_id": transcript_id,
                "target_topic": domain["topic"],
                "expected_domain_idx": domain_idx
            })

        return documents, evaluation_pairs
