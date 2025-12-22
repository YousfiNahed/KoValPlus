# KoValPlus
Persona-based Cultural & Value Alignment Evaluation for LLMs

This repository provides an implementation for evaluating cultural and value alignment between Large Language Model (LLM) responses and real Korean human responses, with a focus on persona-based prompting.

The system compares LLM-generated responses to Korean response distributions derived from the World Values Survey (WVS) and quantitatively measures alignment using statistical similarity metrics.

## Repository Structure
```bash
KoValPlus/
├── code/
│ ├── survey.py # LLM survey response generation
│ └── eval.py # Similarity evaluation
│
├── dataset/
│ └── KoValPlus.json # WVS-based Korean value survey dataset
│
├── outputs/
│ └── gpt-4o-mini/
│ ├── kovalplus_responses_.json
│ └── kovalplus_similarity_.csv
│
├── main.sh # End-to-end execution script
└── README.md
```

## Dataset

The dataset used in this project is based on **World Values Survey (WVS) Wave 7**.  
A subset of value-related survey questions was selected and translated into Korean using **GPT-3.5-Turbo**, and then manually organized into a structured survey format.

The dataset is designed to elicit cultural and value-oriented responses from LLMs and to enable direct comparison with real Korean response distributions provided in the original WVS data.

## Workflow

1. Generate LLM responses to WVS-based Korean value questions  
2. Aggregate responses into distributions  
3. Compare model distributions with real Korean response distributions  
4. Compute similarity scores (e.g., Jensen–Shannon Distance, Cosine Similarity)

## Usage

Run the full pipeline:
```
bash main.sh
```
Or run each step separately:

Generate responses:
```
python code/survey.py
```

Evaluate alignment:
```
python code/eval.py
```

## License
This project is licensed under the MIT License
