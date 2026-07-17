"""Shared skills vocabulary used by both the resume parser (to find skills in
a resume even without an explicit "Skills" section) and the ATS scoring engine
(to auto-extract required skills from a pasted job description).

Deliberately a plain, extendable list rather than an NLP model: keeps the
pipeline fast, offline, and easy for a non-engineer to extend by editing this
file. Matching is case-insensitive; multi-word entries are matched as phrases.
"""

SKILLS_TAXONOMY = [
    # Programming languages
    "python", "java", "javascript", "typescript", "c++", "c#", "c", "go", "golang",
    "rust", "ruby", "php", "swift", "kotlin", "scala", "r", "matlab", "perl",
    "objective-c", "dart", "shell scripting", "bash", "powershell",
    # Web frontend
    "html", "css", "sass", "less", "react", "react.js", "angular", "vue", "vue.js",
    "next.js", "nuxt.js", "redux", "webpack", "tailwind css", "bootstrap", "jquery",
    "svelte",
    # Web backend / frameworks
    "node.js", "express.js", "django", "flask", "fastapi", "spring", "spring boot",
    "asp.net", ".net", "ruby on rails", "laravel", "nestjs",
    # Mobile
    "android", "ios", "react native", "flutter", "xamarin",
    # Data / ML / AI
    "machine learning", "deep learning", "artificial intelligence", "nlp",
    "natural language processing", "computer vision", "data science",
    "data analysis", "data engineering", "data visualization", "pandas", "numpy",
    "scikit-learn", "tensorflow", "pytorch", "keras", "opencv", "power bi",
    "tableau", "excel", "spark", "hadoop", "airflow", "etl", "statistics",
    # Databases
    "sql", "mysql", "postgresql", "sqlite", "mongodb", "redis", "oracle",
    "sql server", "cassandra", "elasticsearch", "dynamodb", "firebase",
    "nosql",
    # Cloud / DevOps
    "aws", "amazon web services", "azure", "gcp", "google cloud platform",
    "docker", "kubernetes", "terraform", "ansible", "jenkins", "ci/cd",
    "devops", "linux", "unix", "nginx", "microservices", "serverless",
    "cloudformation",
    # APIs / architecture
    "rest api", "restful api", "graphql", "grpc", "api development",
    "system design", "object-oriented programming", "oop", "design patterns",
    "agile", "scrum", "kanban", "tdd", "unit testing",
    # Version control / tools
    "git", "github", "gitlab", "bitbucket", "jira", "confluence",
    # QA / testing
    "selenium", "cypress", "junit", "pytest", "manual testing",
    "automation testing", "quality assurance",
    # Security
    "cybersecurity", "penetration testing", "network security", "cryptography",
    # Business / soft skills (kept lightweight -- resumes often list these)
    "project management", "product management", "business analysis",
    "communication", "leadership", "team management", "stakeholder management",
    "problem solving", "critical thinking", "customer service",
    # Other common resume terms
    "salesforce", "sap", "erp", "crm", "seo", "digital marketing",
    "content writing", "ui/ux", "ui design", "ux design", "figma",
    "adobe photoshop", "adobe illustrator",
    # Sales / business development
    "business development", "lead generation", "cold calling", "cold emailing",
    "sales", "b2b sales", "b2c sales", "inside sales", "outside sales",
    "field sales", "telesales", "channel sales", "outbound sales",
    "inbound sales", "sales operations", "account management",
    "client relationship management", "customer relationship management",
    "relationship management", "negotiation", "sales strategy",
    "sales pipeline", "sales forecasting", "prospecting", "market research",
    "competitor analysis", "closing deals", "deal closing", "revenue growth",
    "partnership development", "networking", "presentation skills",
    "powerpoint", "ms office", "microsoft office", "tally", "tally erp",
    "hubspot", "zoho crm", "pipedrive", "customer acquisition",
    "customer retention", "upselling", "cross-selling", "kpi tracking",
    "crm management", "b2b", "b2c", "consultative selling", "consultative sales",
    "client acquisition", "vendor management", "key account management",
    "sales presentations", "business strategy", "market analysis",
]

# Longer/more-specific phrases must be checked before their substrings
# (e.g. "react native" before "react") so sort once, longest first.
SKILLS_TAXONOMY_SORTED = sorted(set(SKILLS_TAXONOMY), key=len, reverse=True)
