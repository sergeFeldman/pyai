import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)  # Module-level logger


class FraudPredictor:
    def __init__(self, graph, embed_manager):
        self.graph = graph
        self.embed_manager = embed_manager
        self.entity_embeddings = self._load_embeddings()
        self.classifier = None
        self.scaler = StandardScaler()

    def _load_embeddings(self):
        """
        Load pre-trained embeddings from EmbeddingManager.
        """
        try:
            if self.embed_manager.entity_embeddings is not None:
                # Get embeddings from the embedding manager
                embeddings = self.embed_manager.entity_embeddings

                # Get mapping from the embedding manager
                entity_to_idx = self.embed_manager.entity_mapping

                return {
                    'embeddings': embeddings,
                    'mapping': entity_to_idx
                }
            else:
                logger.info("Embeddings not found in manager, using random embeddings")
                return self._create_random_embeddings()
        except Exception as e:
            logger.error(f"Error loading embeddings from manager: {e}")
            return self._create_random_embeddings()

    def _create_random_embeddings(self):
        """
        Create random embeddings as fallback.
        """
        entities = list(self.graph.nodes())
        entity_to_idx = {entity: i for i, entity in enumerate(entities)}
        embedding_dim = self.embed_manager.config.kg_embedding.hidden_dim
        embeddings = np.random.randn(len(entities), embedding_dim)
        return {
            'embeddings': embeddings,
            'mapping': entity_to_idx
        }

    def extract_claim_features(self, claim_id):
        """
        Extract features for a claim.
        """
        if claim_id not in self.graph:
            return None

        features = []

        embedding = self.embed_manager.get_entity_embedding(claim_id)
        if embedding is not None:
            features.extend(embedding)
        else:
            embedding_dim = self.embed_manager.config.kg_embedding.hidden_dim
            features.extend([0] * embedding_dim)

        # Graph structure features.
        node_data = self.graph.nodes[claim_id]

        amount = node_data.get('amount', 0)
        features.append(np.log1p(amount))

        neighbors = list(self.graph.neighbors(claim_id))
        features.append(len(neighbors))

        customer = None
        for neighbor in neighbors:
            if self.graph.nodes[neighbor].get('entity_type') == 'customer':
                customer = neighbor
                break

        if customer:
            customer_claims = [n for n in self.graph.neighbors(customer)
                               if self.graph.nodes[n].get('entity_type') == 'claim']
            features.append(len(customer_claims))

            shared_connections = 0
            for neighbor in self.graph.neighbors(customer):
                if self.graph.nodes[neighbor].get('entity_type') == 'customer':
                    shared_connections += 1
            features.append(shared_connections)
        else:
            features.extend([0, 0])

        return np.array(features)

    def train_fraud_classifier(self):
        """
        Train a fraud classification model.
        """
        logger.info("Training fraud classifier...")

        X = []
        y = []

        for node, data in self.graph.nodes(data=True):
            if data.get('entity_type') == 'claim':
                features = self.extract_claim_features(node)
                if features is not None:
                    X.append(features)
                    y.append(1 if data.get('is_fraud') else 0)

        X = np.array(X)
        y = np.array(y)

        logger.info(f"Training on {len(X)} claims ({sum(y)} fraud, {len(y) - sum(y)} normal)")

        if len(X) == 0:
            logger.info("No claim data available for training")
            return 0.0

        X_scaled = self.scaler.fit_transform(X)

        self.classifier = RandomForestClassifier(
            n_estimators=self.embed_manager.config.model.n_estimators,
            max_depth=self.embed_manager.config.model.max_depth,
            random_state=self.embed_manager.config.model.random_state
        )

        self.classifier.fit(X_scaled, y)

        train_accuracy = self.classifier.score(X_scaled, y)
        logger.info(f"Fraud classifier trained with accuracy: {train_accuracy:.3f}")

        return train_accuracy

    def predict_fraud_probability(self, claim_id):
        """
        Predict fraud probability for a claim.
        """
        if self.classifier is None:
            self.train_fraud_classifier()

        features = self.extract_claim_features(claim_id)
        if features is None:
            return 0.0

        features_scaled = self.scaler.transform(features.reshape(1, -1))
        probability = self.classifier.predict_proba(features_scaled)[0, 1]

        return probability

    def evaluate_model(self):
        """
        Evaluate model performance.
        """
        if self.classifier is None:
            logger.info("Model not trained yet. Call train_fraud_classifier() first.")
            return None

        X = []
        y_true = []
        claim_ids = []

        for node, data in self.graph.nodes(data=True):
            if data.get('entity_type') == 'claim':
                features = self.extract_claim_features(node)
                if features is not None:
                    X.append(features)
                    y_true.append(1 if data.get('is_fraud') else 0)
                    claim_ids.append(node)

        if len(X) == 0:
            logger.info("No claim data available for evaluation")
            return None

        X_scaled = self.scaler.transform(X)
        y_pred = self.classifier.predict(X_scaled)
        y_prob = self.classifier.predict_proba(X_scaled)[:, 1]

        logger.info("Fraud Detection Model Evaluation:")
        logger.info(f"AUC-ROC: {roc_auc_score(y_true, y_prob):.3f}")
        logger.info("\nClassification Report:")
        logger.raw(classification_report(y_true, y_pred, target_names=['Normal', 'Fraud']))

        return {
            'auc_roc': roc_auc_score(y_true, y_prob),
            'predictions': list(zip(claim_ids, y_true, y_pred, y_prob))
        }
