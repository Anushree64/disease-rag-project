"""
graph_rag.py — GraphRAG Biomedical Knowledge Graph Extraction Engine.

Constructs structured entity-relation knowledge graphs from retrieved PubMed literature
using NetworkX and extracts graph-centrality knowledge paths.
"""

from typing import Dict, List, Tuple
import networkx as nx


class GraphRAGKnowledgeExtractor:
    """Biomedical Knowledge Graph Extractor & GraphRAG Engine."""

    def __init__(self, disease_name: str):
        self.disease_name = disease_name.replace('_', ' ').title()
        self.graph = nx.DiGraph()

    def build_knowledge_graph(self, evidence_chunks: List[Dict], predicted_class: str) -> Dict:
        """
        Builds directed knowledge graph G = (V, E) from PubMed evidence passages.
        """
        self.graph.clear()
        disease_node = self.disease_name
        class_node = f"{self.disease_name} ({predicted_class})"

        self.graph.add_node(disease_node, type="Disease")
        self.graph.add_node(class_node, type="Subtype")
        self.graph.add_edge(disease_node, class_node, relation="HAS_SUBTYPE")

        for idx, chunk in enumerate(evidence_chunks, 1):
            title = chunk.get('title', f'PubMed_{idx}')
            passage = chunk.get('passage', '')
            pmid = chunk.get('pmid', 'Unknown')
            pmid_node = f"PMID:{pmid}"

            self.graph.add_node(pmid_node, type="Literature", title=title)
            self.graph.add_edge(class_node, pmid_node, relation="SUPPORTED_BY_EVIDENCE")

            # Extract biomedical keyphrase entities
            words = [w.strip('.,;()').lower() for w in passage.split() if len(w) > 5]
            top_words = list(dict.fromkeys(words))[:3]
            for word in top_words:
                ent_node = f"Concept:{word.capitalize()}"
                self.graph.add_node(ent_node, type="BiomedicalConcept")
                self.graph.add_edge(pmid_node, ent_node, relation="ASSOCIATED_WITH")

        # Compute GraphRAG centrality metrics
        degree_centrality = nx.degree_centrality(self.graph)
        top_central_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]

        triples = []
        for u, v, data in self.graph.edges(data=True):
            triples.append((u, data.get('relation', 'RELATED_TO'), v))

        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'extracted_triples': triples[:8],
            'top_central_concepts': [n[0] for n in top_central_nodes if 'Concept:' in n[0]],
            'graph_density': round(float(nx.density(self.graph)), 4),
        }


if __name__ == '__main__':
    print("GraphRAG module ready.")
