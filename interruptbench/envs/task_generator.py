import random
from dataclasses import dataclass
from typing import List

@dataclass
class Task:
    title: str
    description: str
    subtasks: List[str]
    domain: str
    difficulty: str

TASK_TEMPLATES = [
    Task("Technical Report",
         "Write a structured technical report on transformer attention mechanisms.",
         ["Write an executive summary","Describe the problem background",
          "Explain self-attention mathematically",
          "Compare multi-head vs single-head attention",
          "Summarise key findings and limitations"],
         "writing", "medium"),
    Task("Bug Investigation",
         "Investigate and document a memory leak in a Python web server.",
         ["Reproduce the memory leak","Profile heap allocations",
          "Identify the root cause","Propose a fix","Write a regression test"],
         "engineering", "medium"),
    Task("Research Literature Review",
         "Survey recent work on reinforcement learning from human feedback.",
         ["Search for relevant papers (2022-2025)","Summarise the top 5 papers",
          "Identify common themes","Note open problems","Write a conclusion paragraph"],
         "research", "hard"),
    Task("Product Specification",
         "Write a product spec for a new mobile notifications system.",
         ["Define user personas","List functional requirements",
          "List non-functional requirements","Describe edge cases",
          "Propose a success metric"],
         "product", "easy"),
    Task("Data Pipeline Design",
         "Design a batch data pipeline for processing 10M daily events.",
         ["Define ingestion layer","Design transformation logic",
          "Specify storage schema","Describe error handling strategy",
          "Outline monitoring and alerting"],
         "engineering", "hard"),
    Task("Incident Postmortem",
         "Write a postmortem for a 2-hour production database outage.",
         ["Describe the timeline of events","Identify the root cause",
          "List contributing factors","Document immediate mitigations",
          "Propose long-term preventions"],
         "engineering", "medium"),
    Task("API Documentation",
         "Document a REST API for a user authentication service.",
         ["Document POST /register endpoint","Document POST /login endpoint",
          "Document POST /refresh-token endpoint",
          "Describe error codes and meanings","Add a quickstart usage example"],
         "writing", "easy"),
    Task("Competitive Analysis",
         "Analyse three competing AI coding assistants for an internal report.",
         ["Define evaluation criteria","Evaluate Competitor A",
          "Evaluate Competitor B","Evaluate Competitor C","Write a recommendation"],
         "research", "medium"),
    Task("Security Audit Plan",
         "Plan a security audit for a SaaS application handling medical data.",
         ["Identify attack surface areas","Define audit methodology",
          "List tools to use","Specify compliance requirements",
          "Draft a reporting template"],
         "engineering", "hard"),
    Task("Onboarding Guide",
         "Write an onboarding guide for new ML engineers joining the team.",
         ["Describe the development environment setup","Explain the codebase structure",
          "Document the experiment tracking workflow","List key internal tools",
          "Write a first-week checklist"],
         "writing", "easy"),
]

class TaskGenerator:
    def __init__(self, difficulty="any", domain="any", seed=None):
        self.difficulty = difficulty
        self.domain = domain
        self.rng = random.Random(seed)

    def sample(self) -> Task:
        pool = TASK_TEMPLATES
        if self.difficulty != "any":
            pool = [t for t in pool if t.difficulty == self.difficulty] or pool
        if self.domain != "any":
            pool = [t for t in pool if t.domain == self.domain] or pool
        return self.rng.choice(pool)
