import src.utils as utl

logger = utl.config_logging(__name__)

from src.fraud_predictor import FraudPredictor
from src.kg_manager import KnowledgeGraphManager
from src.kg_embed_manager import EmbeddingManager

from src.config import AppConfig
import src.data_models as dm

SAMPLE_DATA_PATH = 'data\in'


def main():
    """Fraud detection pipeline"""
    logger.info("*** FRAUD DETECTION PIPELINE START***\n")

    config = AppConfig.read("config/config.yaml")

    # Step 1: Extract sample data.
    logger.info("1. Extracting sample data...")
    customers = dm.Customer.read(instance_type='dict')
    claims = dm.Claim.read(instance_type='dict')

    # Step 2: Build knowledge graph using sample data.
    logger.info("2. Building knowledge graph...")
    kg_manager = KnowledgeGraphManager(config)
    knowledge_graph = kg_manager.build(customers, claims)

    # Show graph statistics.
    graph_stats = kg_manager.get_stats()
    logger.info(f"Graph Statistics: {graph_stats}")
    embed_manager = EmbeddingManager(config)

    # Step 3: Export for DGL-KE training.
    logger.info("3. Preparing data for training...")
    kg_manager.export_for_dglke(config.data.data_path)

    # Step 4: Train knowledge graph embeddings using orchestrator.
    logger.info("4. Training knowledge graph embeddings...")
    success = embed_manager.train_embeddings()

    if not success:
        logger.info("Embedding training failed. Exiting pipeline.")
        return

    # Show embedding statistics.
    stats = embed_manager.get_embedding_stats()
    logger.info(f"Embedding Statistics: {stats}")

    # Step 5: Train fraud predictor using embeddings from orchestrator.
    logger.info("5. Training fraud detection model...")
    fraud_predictor = FraudPredictor(graph=knowledge_graph, embed_manager=embed_manager)
    fraud_predictor.train_fraud_classifier()

    # Step 6: Evaluate model performance.
    logger.info("6. Evaluating fraud detection model...")
    eval_results = fraud_predictor.evaluate_model()

    # Step 7: Analyze fraud patterns.
    logger.info("7. Analyzing fraud patterns...")
    fraud_claims = kg_manager.get_claims(is_fraud=True)
    normal_claims = kg_manager.get_claims(is_fraud=False)
    logger.info(f"Fraud analysis: {len(fraud_claims)} fraud claims, {len(normal_claims)} normal claims")

    # Step 8: Show sample predictions.
    logger.info("8. Sample predictions:")

    logger.info(f"Sample fraud claims predictions:")
    for claim_id, claim_data in fraud_claims[:3]:
        score = fraud_predictor.predict_fraud_probability(claim_id)
        logger.info(f"  {claim_id}: {score:.3f} (actual: fraud)")

    logger.info(f"Sample normal claims predictions:")
    for claim_id, claim_data in normal_claims[:3]:
        score = fraud_predictor.predict_fraud_probability(claim_id)
        logger.info(f"  {claim_id}: {score:.3f} (actual: normal)")

    logger.info("*** FRAUD DETECTION PIPELINE END***\n")


if __name__ == "__main__":
    main()