from collections import defaultdict
import networkx as nx
import os

from config import Configurable, AppConfig


class KnowledgeGraphManager(Configurable):
    def __init__(self, config: AppConfig):
        super().__init__(config)
        self.graph = nx.MultiDiGraph()

    def build(self, customers: list, claims: list):
        """
        Build a knowledge graph from provided data.

        Args:
            customers (list): List of customer dictionaries
            claims (list): List of claim dictionaries
        """
        print("Building knowledge graph...")

        self.graph.add_nodes_from(
            (
                customer['customer_id'],
                {
                    'label': 'Customer',
                    'name': customer['name'],
                    'address': customer['address'],
                    'phone': customer['phone'],
                    'entity_type': 'customer'
                }
            ) for customer in customers
        )

        self.graph.add_nodes_from(
            (
                claim['claim_id'],
                {
                    'label': 'Claim',
                    'claim_type': claim['claim_type'],
                    'amount': float(claim['amount']),
                    'date': claim['date'],
                    'status': claim['status'],
                    'repair_shop': claim['repair_shop'],
                    'is_fraud': claim['is_fraud'],
                    'entity_type': 'claim'
                }
            ) for claim in claims
        )

        self.graph.add_edges_from(
            (
                claim['customer_id'],
                claim['claim_id'],
                {
                    'label': 'FILED_CLAIM',
                    'relation_type': 'filed_claim'
                }
            ) for claim in claims
        )

        # Repair shop connections.
        self._add_repair_shop_connections(claims)

        # Use config to conditionally add inferred relations.
        self._add_inferred_relations(customers)

        print(f"Graph built with {self.graph.number_of_nodes()} nodes "
              f"and {self.graph.number_of_edges()} edges")
        return self.graph

    def _add_repair_shop_connections(self, claims):
        """
        Add repair shop nodes and connections.
        """
        shop_to_claims = {}
        for claim in claims:
            shop = claim['repair_shop']
            if shop not in shop_to_claims:
                shop_to_claims[shop] = []
            shop_to_claims[shop].append(claim)

        self.graph.add_nodes_from(
            (
                f"shop_{shop.split('_')[-1]}",
                {
                    'label': 'RepairShop',
                    'name': shop,
                    'entity_type': 'repair_shop'
                }
            ) for shop in shop_to_claims.keys()
        )

        self.graph.add_edges_from(
            (
                claim['claim_id'],
                f"shop_{claim['repair_shop'].split('_')[-1]}",
                {
                    'label': 'REPAIRED_AT',
                    'relation_type': 'repaired_at'
                }
            ) for claim in claims
        )

    def _add_inferred_relations(self, customers):
        """
        Add inferred suspicious relations based on config.
        """
        print("Adding inferred relations...")

        if self.config.kg.enable_phone_relations:
            self._add_phone_relations(customers)

        if self.config.kg.enable_address_relations:
            self._add_address_relations(customers)

        print("Inferred relations added")

    def _add_phone_relations(self, customers):
        """
        Add relations based on shared phone numbers.
        """
        phone_groups = KnowledgeGraphManager._group_by_attr(customers, 'phone')

        # Add phone relation edges.
        edges_to_add = [
            (
                customer_group[i]['customer_id'],
                customer_group[j]['customer_id'],
                {
                    'label': 'SHARED_PHONE',
                    'relation_type': 'shared_contact',
                    'weight': self.config.kg.relation_weights.get('shared_contact', 0.8),
                    'evidence': 'same_phone'
                }
            ) for phone, customer_group in phone_groups.items()
            if len(customer_group) > 1
            for i in range(len(customer_group))
            for j in range(i + 1, len(customer_group))
        ]

        self.graph.add_edges_from(edges_to_add)

    def _add_address_relations(self, customers):
        """
        Add relations based on shared addresses.
        """
        address_groups = KnowledgeGraphManager._group_by_attr(customers, 'address')

        # Add address relation edges.
        edges_to_add = [
            (
                customer_group[i]['customer_id'],
                customer_group[j]['customer_id'],
                {
                    'label': 'SHARED_ADDRESS',
                    'relation_type': 'shared_location',
                    'weight': self.config.kg.relation_weights.get('shared_location', 0.9),
                    'evidence': 'same_address'
                }
            ) for address, customer_group in address_groups.items()
            if len(customer_group) > 1
            for i in range(len(customer_group))
            for j in range(i + 1, len(customer_group))
        ]

        self.graph.add_edges_from(edges_to_add)

    def export_for_dglke(self, out_path):
        """
        Export graph in DGL-KE format.
        """
        print("Exporting graph for DGL-KE training...")

        try:
            os.makedirs(out_path, exist_ok=True)

            # Write training triples.
            with open(f"{out_path}/train.txt", 'w') as f:
                for edge in self.graph.edges(data=True):
                    head, tail, data = edge
                    relation = data.get('label', 'RELATED_TO')
                    f.write(f"{head}\t{relation}\t{tail}\n")

            # Create entity and relation mappings.
            entities = set(self.graph.nodes())
            relations = set()

            for _, _, data in self.graph.edges(data=True):
                relations.add(data.get('label', 'RELATED_TO'))

            with open(f"{out_path}/entity.dict", 'w') as f:
                for idx, entity in enumerate(entities):
                    f.write(f"{idx}\t{entity}\n")

            with open(f"{out_path}/relation.dict", 'w') as f:
                for idx, relation in enumerate(relations):
                    f.write(f"{idx}\t{relation}\n")

            print(f"Exported {len(entities)} entities and {len(relations)} relations")

        except Exception as e:
            print(f"Error exporting graph for DGL-KE: {e}")
            raise

    def get_claims(self, is_fraud: bool = False):
        """
        Get claim nodes.
        """
        target_claims = []
        for node, data in self.graph.nodes(data=True):
            if data.get('entity_type') == 'claim' and data.get('is_fraud') == is_fraud:
                target_claims.append((node, data))
        return target_claims

    def get_stats(self):
        """
        Get graph related statistics.
        """
        fraud_claims = self.get_claims(is_fraud=True)
        normal_claims = self.get_claims()

        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'fraud_claims': len(fraud_claims),
            'normal_claims': len(normal_claims),
            'fraud_ratio': round(len(fraud_claims) / (len(fraud_claims) + len(normal_claims)), 2)
        }

    @staticmethod
    def _group_by_attr(input_list: list, attribute: str) -> dict:
        grouped = defaultdict(list)
        for item in input_list:
            grouped[item[attribute]].append(item)
        return dict(grouped)
