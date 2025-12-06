import dgl
import dgl.nn as dglnn
from dgl import DGLGraph
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Dict, Optional, Tuple

from config import Configurable, AppConfig


class EmbeddingManager(Configurable):
    """
    Management of knowledge graph embeddings using DGL.
    """

    def __init__(self, config: AppConfig):
        super().__init__(config)

        self.model: Optional[nn.Module] = None
        self.entity_emb: Optional[nn.Embedding] = None  # ADD THIS
        self.entity_embeddings: Optional[np.ndarray] = None
        self.relation_embeddings: Optional[np.ndarray] = None
        self.entity_mapping: Dict[str, int] = {}
        self.relation_mapping: Dict[str, int] = {}
        self.graph: Optional[DGLGraph] = None

        # Ensure directories exist.
        os.makedirs(self.config.data.data_path, exist_ok=True)
        os.makedirs(self.config.data.embed_path, exist_ok=True)

    def train_embeddings(self) -> bool:
        """
        Train knowledge graph embeddings using DGL.
        """
        kg_embed_config = self.config.kg_embedding
        print(f"Training {kg_embed_config.model_name} embeddings with DGL...")

        try:
            # Load the knowledge graph
            self._load_knowledge_graph()

            if self.graph is None:
                print("Failed to load knowledge graph data.")
                return False

            num_entities = len(self.entity_mapping)
            num_relations = len(self.relation_mapping)
            print(f"Training on {self.graph.num_edges()} triples with {num_entities} entities and {num_relations} relations")

            # Create model using DGL's implementations.
            if kg_embed_config.model_name == 'DistMult':
                self.model = dglnn.DistMult(num_entities, num_relations, kg_embed_config.hidden_dim)
            else:  # TransE
                self.model = dglnn.TransE(num_rels=num_relations, feats=kg_embed_config.hidden_dim, p=1)

            # Add entity embeddings layer.
            self.entity_emb = nn.Embedding(num_entities, kg_embed_config.hidden_dim)

            # Combine entity emb + model params.
            self.optimizer = optim.Adam(
                list(self.entity_emb.parameters()) + list(self.model.parameters()),
                lr=kg_embed_config.learning_rate)

            # Train the model using custom training loop.
            print("Starting training with DGL...")
            self._train_model(kg_embed_config.max_step)

            # Extract and save embeddings.
            self._extract_embeddings()
            self._write()

            print(f"Training completed. Saved {len(self.entity_embeddings)} entity embeddings")
            return True

        except Exception as e:
            print(f"Training failed: {str(e)}")  # Convert to string
            import traceback
            traceback.print_exc()  # Print full traceback
            return False

    def _train_model(self, max_steps: int):
        """
        Custom training loop for DGL models.
        """
        kg_embed_config = self.config.kg_embedding

        self.model.train()
        self.entity_emb.train()

        for step in range(max_steps):
            self.optimizer.zero_grad()

            # Get edge data.
            edges = torch.arange(self.graph.num_edges())
            head, tail = self.graph.edges()
            rel = self.graph.edata['etype']

            # Sample a batch.
            batch_size = min(1024, len(edges))
            batch_edges = edges[torch.randperm(len(edges))[:batch_size]]

            batch_head = head[batch_edges]
            batch_tail = tail[batch_edges]
            batch_rel = rel[batch_edges]

            # Get entity embeddings for the batch.
            emb_head = self.entity_emb(batch_head)
            emb_tail = self.entity_emb(batch_tail)

            # Generate positive and negative scores.
            pos_scores = self.model(emb_head, emb_tail, batch_rel)
            neg_head, neg_rel, neg_tail = self._generate_negative_samples(batch_head, batch_rel, batch_tail)

            # Get embeddings for negative samples.
            neg_emb_head = self.entity_emb(neg_head)
            neg_emb_tail = self.entity_emb(neg_tail)

            # Generate negative scores.
            neg_scores = self.model(neg_emb_head, neg_emb_tail, neg_rel)

            # Compute loss.
            if kg_embed_config.model_name == 'DistMult':
                # DistMult uses BCE loss.
                pos_labels = torch.ones_like(pos_scores)
                neg_labels = torch.zeros_like(neg_scores)

                pos_loss = nn.functional.binary_cross_entropy_with_logits(
                    pos_scores, pos_labels)
                neg_loss = nn.functional.binary_cross_entropy_with_logits(
                    neg_scores, neg_labels)
                loss = (pos_loss + neg_loss) / 2
            else:
                # TransE uses margin ranking loss.
                loss = torch.clamp(neg_scores - pos_scores + self.config.kg_embedding.gamma, min=0).mean()

            loss.backward()
            self.optimizer.step()

            if step % 100 == 0:
                print(f"Step {step}/{max_steps}, Loss: {loss.item():.4f}")

    def _generate_negative_samples(self, head, rel, tail):
        """
        Generate negative samples and compute their scores.
        """
        batch_size = head.size(0)

        # Corrupt either head or tail.
        corrupt_head = torch.rand(batch_size) > 0.5
        neg_head = head.clone()
        neg_tail = tail.clone()
        neg_rel = rel.clone()

        # Randomly replace heads or tails.
        num_entities = len(self.entity_mapping)

        if corrupt_head.sum() > 0:
            neg_head[corrupt_head] = torch.randint(0, num_entities, (corrupt_head.sum().item(),))

        if (~corrupt_head).sum() > 0:
            neg_tail[~corrupt_head] = torch.randint(0, num_entities, ((~corrupt_head).sum().item(),))

        return neg_head, neg_rel, neg_tail

    def _load_knowledge_graph(self):
        """
        Load knowledge graph data using DGL's format.
        """
        try:
            # Use your original file naming convention
            train_file = f"{self.config.data.data_path}/train.txt"
            entity_file = f"{self.config.data.data_path}/entity.dict"  # FIXED: Use original naming
            relation_file = f"{self.config.data.data_path}/relation.dict"  # FIXED: Use original naming

            if not all(os.path.exists(f) for f in [train_file, entity_file, relation_file]):
                print("KG files not found. Please export data first.")
                return

            # Read mappings with original naming
            self._read_mapping("entity")
            self._read_mapping("relation")

            # Create DGL graph from triples.
            triples = []
            with open(train_file, 'r') as f:
                for line in f:
                    if line.strip():
                        head, relation, tail = line.strip().split('\t')
                        triples.append((self.entity_mapping[head],
                                        self.relation_mapping[relation],
                                        self.entity_mapping[tail]))

            # Create DGL graph.
            heads, relations, tails = map(list, zip(*triples))

            self.graph = dgl.graph((heads, tails), num_nodes=len(self.entity_mapping))
            self.graph.edata['etype'] = torch.tensor(relations)

            print(f"Created DGL graph with {self.graph.num_nodes()} nodes and {self.graph.num_edges()} edges")

        except Exception as e:
            print(f"Error loading knowledge graph: {e}")
            self.graph = None

    def _read_mapping(self, part_type):
        """
        Read mapping from file - keeping your original method.
        """
        try:
            mapping: Dict[str, int] = {}
            file = f"{self.config.data.data_path}/{part_type}.dict"
            if os.path.exists(file):
                with open(file, 'r') as f:
                    for line in f:
                        idx, part_value = line.strip().split('\t')
                        mapping[part_value] = int(idx)
                    setattr(self, f"{part_type}_mapping", mapping)
                print(f"Read {len(mapping)} {part_type} mapping")
            else:
                raise FileNotFoundError(f"{part_type} mapping file not found: {file}")
        except Exception as e:
            print(f"Error reading {part_type} mapping: {e}")
            raise

    def _extract_embeddings(self):
        """
        Extract embeddings from trained DGL model.
        """
        if self.model is None:
            print("No trained model available.")
            return

        try:
            if isinstance(self.model, dglnn.TransE):
                # TransE: entity embeddings are in our separate layer
                if self.entity_emb is not None:
                    self.entity_embeddings = self.entity_emb.weight.data.cpu().numpy()
                else:
                    print("Warning: entity_emb is None for TransE")
            else:
                # DistMult: check model for entity embeddings
                if hasattr(self.model, 'emb'):
                    self.entity_embeddings = self.model.emb.weight.data.cpu().numpy()
                elif hasattr(self.model, 'entity_emb'):
                    self.entity_embeddings = self.model.entity_emb.weight.data.cpu().numpy()
                else:
                    print("Warning: Could not find entity embeddings in DistMult model")

            # Extract relation embeddings (TransE has rel_emb, DistMult has w_relation)
            if hasattr(self.model, 'w_relation'):
                self.relation_embeddings = self.model.w_relation.weight.data.cpu().numpy()
            elif hasattr(self.model, 'rel_emb'):
                self.relation_embeddings = self.model.rel_emb.weight.data.cpu().numpy()
            else:
                print("Warning: Could not find relation embeddings in model")

            print(f"Extracted {len(self.entity_embeddings) if self.entity_embeddings is not None else 0} entity embeddings and {len(self.relation_embeddings) if self.relation_embeddings is not None else 0} relation embeddings")

        except Exception as e:
            print(f"Error extracting embeddings: {e}")

    def _read(self):
        """
        Read embeddings.
        """
        try:
            entity_path = f"{self.config.data.embed_path}/entity_emb.npy"
            relation_path = f"{self.config.data.embed_path}/relation_emb.npy"

            if os.path.exists(entity_path):
                self.entity_embeddings = np.load(entity_path)
                print(f"Read {len(self.entity_embeddings)} entity embeddings")

            if os.path.exists(relation_path):
                self.relation_embeddings = np.load(relation_path)
                print(f"Read {len(self.relation_embeddings)} relation embeddings")

            # Read mappings for lookup
            self._read_mapping("entity")
            self._read_mapping("relation")

        except Exception as e:
            print(f"Error reading embeddings: {e}")
            self.entity_embeddings = None

    def _write(self):
        """
        Write embeddings.
        """
        try:
            embed_path = self.config.data.embed_path
            np.save(f"{embed_path}/entity_emb.npy", self.entity_embeddings)
            np.save(f"{embed_path}/relation_emb.npy", self.relation_embeddings)
            print(f"Embeddings written to {embed_path}")
        except Exception as e:
            print(f"Error writing embeddings: {e}")

    def get_entity_embedding(self, entity_id: str) -> Optional[np.ndarray]:
        """
        Get embedding for a specific entity.
        """
        if self.entity_embeddings is None:
            self._read()

        if self.entity_embeddings is not None and entity_id in self.entity_mapping:
            idx = self.entity_mapping[entity_id]
            return self.entity_embeddings[idx]
        return None

    def get_relation_embedding(self, relation_id: str) -> Optional[np.ndarray]:
        """
        Get embedding for a specific relation.
        """
        if self.relation_embeddings is None:
            self._read()

        if self.relation_embeddings is not None and relation_id in self.relation_mapping:
            idx = self.relation_mapping[relation_id]
            return self.relation_embeddings[idx]
        return None

    def get_embedding_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the trained embeddings.
        """
        stats = {
            'model_name': self.config.kg_embedding.model_name,
            'embedding_dim': self.config.kg_embedding.hidden_dim,
            'entities_count': len(self.entity_mapping) if self.entity_mapping else 0,
            'relations_count': len(self.relation_mapping) if self.relation_mapping else 0,
            'embeddings_loaded': self.entity_embeddings is not None,
            'graph_loaded': self.graph is not None
        }
        return stats
