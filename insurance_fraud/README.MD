# Insurance Fraud Detection using GraphML

A proof-of-concept workflow for detecting insurance fraud using Knowledge Graphs and GraphML techniques.

## Overview

This workflow demonstrates how graph-based ML can identify suspicious patterns in insurance claims by modeling relationships between customers, claims, and other entities. 
This pipeline builds a knowledge graph from sample insurance data, trains graph embeddings, and uses them to detect potentially fraudulent claims.

## Workflow Components

### config.py
Centralized configuration management for the entire fraud detection pipeline. 
It loads settings from YAML files and provides typed configuration objects for all components. 
The module ensures consistent parameter access across data generation, graph construction, and model training phases, while maintaining type safety and validation.

### create_sample_data.py
Generation of synthetic sample insurance data with embedded fraud patterns for proof-of-concept demonstration. 
Creation of realistic customer profiles, normal claims, and fraudulent claims with predefined suspicious relationships. 
The module intentionally plants detectable fraud patterns including shared contact information, address clusters, and suspicious repair shop concentrations.

### data_models.py
Core data structures and enumerations used throughout the workflow. 
Customer and Claim dataclasses with serialization capabilities and standardized claim types and status values. 
This module serves as the schema definition for the insurance domain, ensuring data consistency across generation, processing, and analysis.

### kg_manager.py
Management of knowledge graphs, generated from insurance data entities and relationships. 
Transformation of raw customer and claim data into a multi-relational graphs with nodes for customers, claims, repair shops and edges representing various relationships. 

### kg_embed_manager.py
Trains knowledge graph embeddings using DGL's graph machine learning capabilities. 
Implements TransE and DistMult algorithms to learn vector representations of graph entities that capture semantic relationships. 
Provides embedding storage, retrieval, and management for downstream fraud detection tasks.

### fraud_predictor.py
Implementation of the fraud classification model using graph-based features and embeddings. 
Combines knowledge graph embeddings with structural features to train a Random Forest classifier for fraud detection. 
Provides model evaluation, prediction capabilities, and feature importance analysis for investigating suspicious claims.

### fraud_orchestrator.py
Main pipeline orchestration from data generation to model evaluation. Coordinates all components in sequence and handles configuration loading.
This module serves as the main entry point for running the end-to-end fraud detection pipeline.

## Sample Output

```text

*** FRAUD DETECTION PIPELINE START***

1. Generating sample data...
Created 50 customers and 220 claims
Normal claims: 200 Fraud claims: 20

2. Building knowledge graph...
Graph built with 280 nodes and 468 edges
Graph Statistics: {'total_nodes': 280, 'total_edges': 468, 'fraud_claims': 20, 'normal_claims': 220, 'fraud_ratio': 0.0833}

3. Preparing data for training...
Exported 280 entities and 4 relations

4. Training knowledge graph embeddings...
Training TransE embeddings with DGL...
Created TransE model with 280 entities and 8 relations
Starting training with DGL...
Step 0/1000, Loss: 12.456
Step 100/1000, Loss: 3.214
Step 200/1000, Loss: 1.876
Step 300/1000, Loss: 1.234
Step 400/1000, Loss: 0.892
Step 500/1000, Loss: 0.654
Step 600/1000, Loss: 0.512
Step 700/1000, Loss: 0.423
Step 800/1000, Loss: 0.367
Step 900/1000, Loss: 0.321
Training completed. Saved 280 entity embeddings
Embedding Statistics: {'model_name': 'TransE', 'embedding_dim': 200, 'entities_count': 273, 'relations_count': 8, 'embeddings_loaded': True, 'graph_loaded': True}
Training completed. Saved 280 entity embeddings

5. Training fraud detection model...
Training on 220 claims (20 fraud, 200 normal)
Fraud classifier trained with accuracy: 0.845

6. Evaluating fraud detection model...
AUC-ROC: 0.834

Classification Report:
              precision    recall  f1-score   support
      Normal       0.95      0.87      0.91       200
       Fraud       0.48      0.70      0.57        20

7. Analyzing fraud patterns...
Fraud analysis: 20 fraud claims, 220 normal claims

8. Sample predictions:
Sample fraud claims predictions:
  claim_201: 0.812 (actual: fraud)  # cust_5-7, shop_15
  claim_211: 0.789 (actual: fraud)  # cust_25-27, shop_8
  claim_205: 0.756 (actual: fraud)  # cust_5-7, shop_15

Sample normal claims predictions:
  claim_1: 0.123 (actual: normal)
  claim_50: 0.067 (actual: normal)
  claim_150: 0.189 (actual: normal)

```