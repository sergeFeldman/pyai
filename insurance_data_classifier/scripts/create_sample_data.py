"""
Creation of testing sample data for "Insurance Document Classification" process.
"""

from datetime import datetime, timedelta
import os
import pandas as pd
import random

import src.utils as utl

DOCUMENT_NUMBER = 100
SAMPLE_DATA_PATH = 'data/in/insurance_docs.csv'

# Document type configurations
DOCUMENT_CONFIGS = {
    "claim_form": {
        "templates": [
            "CLAIM FORM\nPolicy Number: {policy_num}\nDate of Loss: {date}\nDescription: {description}",
            "CLAIM SUBMISSION\nClaim Number: {policy_num}\nPolicy Holder: {name}\nIncident Date: {date}\nDamage: {description}",
            "INSURANCE CLAIM\nReference: {policy_num}\nLoss Date: {date}\nClaim Details: {description}\nClaimant: {name}"
        ],
        "descriptions": ["Fire damage", "Theft", "Vehicle accident", "Water damage"]
    },
    "invoice": {
        "templates": [
            "BILLING STATEMENT\nAccount: {policy_num}\nTotal Due: {amount}\nDue Date: {date}",
            "INVOICE\nInvoice Number: {policy_num}\nAmount: {amount}\nServices: {description}",
            "STATEMENT OF ACCOUNT\nInvoice: {policy_num}\nAmount Due: {amount}\nIssue Date: {date}\nDescription: {description}"
        ],
        "descriptions": ["Legal services", "Medical services", "Property repairs", "Vehicle repairs"]
    },
    "legal_correspondence": {
        "templates": [
            "ATTORNEY CORRESPONDENCE\nRe: Claim {policy_num}\nDear Sir/Madam,\nWe represent {name} in the matter of the insurance claim...",
            "LEGAL NOTICE\nReference: {policy_num}\nDate: {date}\nThis letter confirms the receipt of coverage and outlines the terms...",
            "SETTLEMENT OFFER\nClaim Number: {policy_num}\nDate: {date}\nWe are pleased to offer a settlement amount of {amount}...",
            "To Whom It May Concern,\nWe are writing regarding our recent claim submission for policy {policy_num}..."
        ]
    },
    "medical_report": {
        "templates": [
            "HEALTH ASSESSMENT\nPatient ID: {policy_num}\nName: {name}\nAssessment Date: {date}\nDiagnosis: {description}",
            "MEDICAL EXAMINATION REPORT\nPatient: {name}\nFindings: {description}\nDate: {date}",
            "MEDICAL REPORT\nHPW Number: {policy_num}\nPatient: {name}\nDiagnosis: {description}\nTreatment: Physical therapy",
            "TREATMENT SUMMARY\nMedical Record: {policy_num}\nPatient: {name}\nTreatment Date: {date}\nProcedures: {description}"
        ],
        "descriptions": ["Fractured arm", "Normal examination", "Soft tissue injury", "Whiplash injury"]
    },
    "policy_form": {
        "templates": [
            "COVERAGE APPLICATION\nApplicant Name: {name}\nSubmission Date: {date}\nPolicy Type: {description}",
            "INSURANCE APPLICATION\nProposed Insured: {name}\nApplication Date: {date}\nReference: {policy_num}",
            "POLICY APPLICATION FORM\nApplicant: {name}\nPolicy Number: {policy_num}\nCoverage Type: {description}"
        ],
        "descriptions": ["Auto Insurance", "Homeowners Insurance"]
    },
    "police_report": {
        "templates": [
            "INCIDENT REPORT\nReport Number: {policy_num}\nDate: {date}\nIncident Type: {description}",
            "LAW ENFORCEMENT REPORT\nCase: {policy_num}\nReporting Officer: {name}\nIncident Details: {description}\nDate: {date}",
            "POLICE REPORT\nCase Number: {policy_num}\nIncident: {description}\nOfficer: {name}"
        ],
        "descriptions": ["Theft incident", "Traffic collision", "Vandalism report"]
    }
}

# Prefixes for different document types.
PREFIXES = {
    "claim_form": ["CL", "IC"],
    "invoice": ["IC", "INV"],
    "legal_correspondence": ["LC", "LEG"],
    "medical_report": ["MED", "MR"],
    "policy_form": ["CV", "POL"],
    "police_report": ["IR", "PR"]
}

# Name lists
FIRST_NAMES = ["Anna", "John", "Ilan", "Lana", "Robert"]
LAST_NAMES = ["Baker", "Davis", "Jones", "Smith", "Williams"]


def create_sample_data():
    """
    Create sample data file for testing.
    """
    logger = utl.config_logging(__name__)

    sample_data = []

    for doc_type, config in DOCUMENT_CONFIGS.items():
        for idx in range(DOCUMENT_NUMBER):
            # Generate random data.
            description = random.choice(config["descriptions"]) if "descriptions" in config and config["descriptions"] else ""
            template = random.choice(config["templates"])

            # Format the string.
            doc_string = template.format(
                policy_num=_random_policy_number(random.choice(PREFIXES[doc_type])),
                name=_random_name(),
                date=_random_date(),
                amount=_random_amount(),
                description=description
            )
            sample_data.append({
                "string": doc_string,
                "label": doc_type
            })

    os.makedirs('data/in', exist_ok=True)

    df = pd.DataFrame(sample_data)
    df.to_csv(SAMPLE_DATA_PATH, index=False)
    logger.info(f"Created sample data with {len(df)} records at {SAMPLE_DATA_PATH}")


def _random_amount(min_amount=500, max_amount=1500):
    """
    Generate a random dollar amount.
    """
    amount = random.uniform(min_amount, max_amount)
    return f"${amount:,.2f}"


def _random_date(start_year=2020, end_year=2024):
    """
    Generate a random date between start_year and end_year.
    """
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    random_days = random.randint(0, (end_date - start_date).days)
    return (start_date + timedelta(days=random_days)).strftime("%Y-%m-%d")


def _random_name():
    """
    Generate a random name.
    """
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


def _random_policy_number(prefix, length=5):
    """
    Generate a random policy/claim number.
    """
    numbers = ''.join([str(random.randint(0, 9)) for _ in range(length)])
    return f"{prefix}-{numbers}"


if __name__ == "__main__":
    create_sample_data()