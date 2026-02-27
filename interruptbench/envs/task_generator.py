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
     Task("Style Guide",
          "Write a company-wide coding style guide for Python projects.",
          ["Define naming conventions", "Specify import ordering rules",
          "Document string formatting standards", "Write docstring requirements",
          "Add a linting and enforcement section"],
          "writing", "hard"),
     Task("Environment Setup Script",
          "Write a shell script to set up a new developer machine from scratch.",
          ["Install system dependencies", "Configure git and SSH keys",
          "Set up Python with pyenv", "Install project dependencies",
          "Verify the setup works end-to-end"],
          "engineering", "easy"),
     Task("Dataset Audit",
          "Audit a machine learning training dataset for quality issues.",
          ["Check for class imbalance", "Identify duplicate records",
          "Flag missing or null values", "Assess label noise",
          "Write a summary report with recommendations"],
          "research", "easy"),
     Task("Quarterly Roadmap",
          "Draft a product roadmap for the next quarter for an API platform.",
          ["Review last quarter outcomes", "Gather stakeholder input",
          "Prioritise features by impact and effort",
          "Define success metrics per feature", "Write the roadmap document"],
          "product", "medium"),
     Task("Migration Plan",
          "Plan a zero-downtime database migration from PostgreSQL to a new schema.",
          ["Audit current schema and dependencies", "Design the new schema",
          "Write a forwards-compatible migration script",
          "Plan a rollback strategy", "Define a staging validation checklist"],
          "engineering", "hard"),
]

class TaskGenerator:
    """
    Samples structured multi-step tasks for use in InterruptEnv episodes.

    Draws from a fixed pool of task templates, optionally filtered by
    difficulty tier and domain. All sampling is seeded for reproducibility.

    Available difficulties: "easy" | "medium" | "hard" | "any"
    Available domains:      "writing" | "engineering" | "research" | "product" | "any"

    If the filtered pool is empty (e.g. no hard product tasks exist),
    the filter is silently dropped and the full pool is used. This prevents
    training from stalling on misconfigured filters.

    Args:
        difficulty: str — filter tasks by difficulty tier, or "any" for no filter
        domain:     str — filter tasks by domain, or "any" for no filter
        seed:       int | None — RNG seed for reproducible sampling
    """
    def __init__(self, difficulty="any", domain="any", seed=None):
        self.difficulty = difficulty
        self.domain = domain
        self.rng = random.Random(seed)

    def sample(self) -> Task:
        """
        Sample one task from the filtered template pool.

        Applies difficulty and domain filters in sequence. If either filter
        produces an empty pool, falls back to the unfiltered pool to avoid
        raising an error mid-episode.

        Returns:
            Task — a randomly selected task matching the configured filters
        """
        pool = TASK_TEMPLATES
        if self.difficulty != "any":
            pool = [t for t in pool if t.difficulty == self.difficulty] or pool
        if self.domain != "any":
            pool = [t for t in pool if t.domain == self.domain] or pool
        return self.rng.choice(pool)
    
    @classmethod
    def from_config(cls, config: dict) -> "TaskGenerator":
     """
     Construct a TaskGenerator from a configuration dictionary.

     Read the keys "difficulty", "domain", and "seed" from the dict,
     falling back to defaults if any are absent. Intended for use with
     YAML-loaded config dicts.

     Args:
          config: dict - configuration dict, typically from default.yaml
     
     Returns:
          TaskGenerator instance initialized according to the provided settings
     """
     return cls(
          difficulty=config.get("difficulty", "any"),
          domain=config.get("domain", "any"),
          seed=config.get("seed", None),
     )

