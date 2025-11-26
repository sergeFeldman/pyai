# Insurance Documents Classification POC

Automatic classification of insurance documents into specific categories using fine-tuned transformer models.

## Document Types

The system classifies documents into 6 categories:

- **`claim_form`** - Insurance claim forms and submissions
- **`invoice`** - Billing statements and invoices
- **`legal_correspondence`** - Legal notices and attorney correspondence
- **`medical_report`** - Medical examinations and health assessments
- **`police_report`** - Law enforcement incident reports
- **`policy_form`** - Insurance policy applications and forms

## 📊 Workflow

1. Create Sample Data

Generate synthetic sample strings, which represent insurance documents for testing purposes:

```bash 

python -m scripts.create_sample_data
```

### Sample Data Structure:
```csv

string,label
"CLAIM FORM\nPolicy Number: CLM-28374\nDate of Loss: 2023-08-12\nDescription: Water damage","claim_form"
"INVOICE\nInvoice Number: INV-84726\nAmount: $3,847.50\nServices: Property repairs","invoice"
"To Whom It May Concern,\nWe are writing regarding our recent claim submission...","legal_correspondence"
"MEDICAL REPORT\nPatient: Mary Johnson\nDiagnosis: Soft tissue injury\nTreatment: Physical therapy","medical_report"
"POLICE REPORT\nCase Number: CR-2023-428\nIncident: Theft incident\nOfficer: Smith, J.","police_report"
"POLICY APPLICATION FORM\nApplicant: Robert Brown\nCoverage Type: Auto Insurance","policy_form"
```

2. Train the Model

Train the document classification model (and persist its artifacts)

```bash 

python -m scripts.train
```

### Model Artifacts Created:

```text
models/insurance_classifier/
├── config.json
├── config.yaml
├── label_mappings.pt
├── model.safetensors
├── special_tokens_map.json
├── tokenizer_config.json
├── tokenizer.json
├── training_args.bin
└── vocab.txt
```

3. Predict Document Types

Classify provided insurance documents

```bash

python -m scripts.predict --text "POLICE REPORT Case Number: CR-2024-015 Incident: Traffic collision"
# Output: Predicted: police_report (Confidence: 0.8678)
```
