from create_sample_data import create_sample_data
from fraud_predictor import FraudPredictor
from kg_manager import KnowledgeGraphManager
from kg_embed_manager import EmbeddingManager

from config import AppConfig


def main():
    """Fraud detection pipeline"""
    print("*** FRAUD DETECTION PIPELINE START***\n")

    config = AppConfig.read("config/config.yaml")

    # Step 1: Generate sample data.
    print("\n1. Generating sample data...")
    customers, claims = create_sample_data()

    # Step 2: Build knowledge graph using sample data.
    print("\n2. Building knowledge graph...")
    kg_manager = KnowledgeGraphManager(config)
    knowledge_graph = kg_manager.build(customers, claims)

    # Show graph statistics.
    graph_stats = kg_manager.get_stats()
    print(f"Graph Statistics: {graph_stats}")
    embed_manager = EmbeddingManager(config)

    # Step 3: Export for DGL-KE training.
    print("\n3. Preparing data for training...")
    kg_manager.export_for_dglke(config.data.data_path)

    # Step 4: Train knowledge graph embeddings using orchestrator.
    print("\n4. Training knowledge graph embeddings...")
    success = embed_manager.train_embeddings()

    if not success:
        print("Embedding training failed. Exiting pipeline.")
        return

    # Show embedding statistics.
    stats = embed_manager.get_embedding_stats()
    print(f"Embedding Statistics: {stats}")

    # Step 5: Train fraud predictor using embeddings from orchestrator.
    print("\n5. Training fraud detection model...")
    fraud_predictor = FraudPredictor(
        graph=knowledge_graph,
        embed_manager=embed_manager
    )
    fraud_predictor.train_fraud_classifier()

    # Step 6: Evaluate model performance.
    print("\n6. Evaluating fraud detection model...")
    eval_results = fraud_predictor.evaluate_model()

    # Step 7: Analyze fraud patterns.
    print("\n7. Analyzing fraud patterns...")
    fraud_claims = kg_manager.get_claims(is_fraud=True)
    normal_claims = kg_manager.get_claims(is_fraud=False)
    print(f"Fraud analysis: {len(fraud_claims)} fraud claims, {len(normal_claims)} normal claims")

    # Step 8: Show sample predictions.
    print("\n8. Sample predictions:")

    print(f"Sample fraud claims predictions:")
    for claim_id, claim_data in fraud_claims[:3]:
        score = fraud_predictor.predict_fraud_probability(claim_id)
        print(f"  {claim_id}: {score:.3f} (actual: fraud)")

    print(f"\nSample normal claims predictions:")
    for claim_id, claim_data in normal_claims[:3]:
        score = fraud_predictor.predict_fraud_probability(claim_id)
        print(f"  {claim_id}: {score:.3f} (actual: normal)")

    print("\n*** FRAUD DETECTION PIPELINE END***\n")


if __name__ == "__main__":
    main()