"""
Entity Resolution using Name Matching Model

This script performs entity resolution on transaction data using the trained
name matching model to identify and consolidate entities across transactions.
"""

import os
import time
import warnings
from itertools import combinations
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import structlog
from networkx.algorithms import community as nx_community

from name_matching.config import read_config
from name_matching.log.logging import configure_structlog
from name_matching.models.predict_model import NameMatchingPredictor
from name_matching.utils.utils import process_text_standard

# Disable tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress warnings
warnings.filterwarnings("ignore")

# Read configuration
config = read_config()


class EntityResolver:
    """
    Entity resolution system using name matching model and graph-based community detection.
    """

    def __init__(self, logger=None):
        """
        Initialize the entity resolver.

        :param logger: Logger instance (optional)
        """

        self.logger = logger if logger is not None else structlog.get_logger()
        self.predictor = NameMatchingPredictor(logger=self.logger)

    def load_transaction_data(self, file_path: str) -> pd.DataFrame:
        """
        Load transaction data from CSV file.

        :param file_path: Path to the CSV file
        :return: DataFrame containing transaction data
        """

        self.logger.info("LOADING_TRANSACTION_DATA", path=file_path)
        df_txn = pd.read_csv(file_path)
        self.logger.info("LOADED_TRANSACTIONS", count=len(df_txn))

        return df_txn

    def preprocess_names(self, df_txn: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess name columns in the transaction DataFrame.

        :param df_txn: Input DataFrame with Cust_Name and Counterpart_Name columns
        :return: DataFrame with additional preprocessed columns
        """

        self.logger.info("PREPROCESSING_NAMES")
        # Create a copy to avoid modifying the original
        df_txn = df_txn.copy()

        # Preprocess customer names
        df_txn["Cust_Name_Processed"] = df_txn["Cust_Name"].apply(
            lambda x: process_text_standard(x.upper(), remove_stopwords=False)
        )

        # Preprocess counterparty names
        df_txn["Counterpart_Name_Processed"] = df_txn["Counterpart_Name"].apply(
            lambda x: process_text_standard(x.upper(), remove_stopwords=False)
        )

        self.logger.info("PREPROCESSING_COMPLETE")
        return df_txn

    def deduplicate_transactions(self, df_txn: pd.DataFrame) -> pd.DataFrame:
        """
        Deduplicate transactions by preprocessed name pairs.

        :param df_txn: DataFrame with preprocessed name columns
        :return: Deduplicated DataFrame
        """

        self.logger.info("DEDUPLICATING_TRANSACTIONS", original_count=len(df_txn))
        # Drop duplicates based on preprocessed names
        df_dedup = df_txn.drop_duplicates(
            subset=["Cust_Name_Processed", "Counterpart_Name_Processed"]
        ).reset_index(drop=True)

        self.logger.info(
            "DEDUPLICATION_COMPLETE",
            original_count=len(df_txn),
            deduplicated_count=len(df_dedup),
        )

        return df_dedup

    def create_transaction_graph(
        self, df_txn: pd.DataFrame, name_col_x: str, name_col_y: str
    ) -> nx.Graph:
        """
        Create a NetworkX graph from transaction pairs.

        :param df_txn: DataFrame containing transaction data
        :param name_col_x: Column name for first entity in pair
        :param name_col_y: Column name for second entity in pair
        :return: NetworkX Graph object
        """

        self.logger.info("CREATING_TRANSACTION_GRAPH")
        G = nx.Graph()

        # Add edges for each transaction pair
        for _, row in df_txn.iterrows():
            name_x = row[name_col_x]
            name_y = row[name_col_y]
            G.add_edge(name_x, name_y)

        self.logger.info(
            "GRAPH_CREATED",
            nodes=G.number_of_nodes(),
            edges=G.number_of_edges(),
        )

        return G

    def visualize_graph(
        self, G: nx.Graph, output_path: str, title: str = "Transaction Graph"
    ) -> None:
        """
        Visualize and save a NetworkX graph.

        :param G: NetworkX Graph object
        :param output_path: Path to save the visualization
        :param title: Title for the graph
        """

        self.logger.info("VISUALIZING_GRAPH", output_path=output_path)
        plt.figure(figsize=(14, 10))

        # Use spring layout for better visualization
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=800, alpha=0.9)
        nx.draw_networkx_edges(G, pos, alpha=0.5, width=2)
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")

        plt.title(title, fontsize=16, fontweight="bold")
        plt.axis("off")
        plt.tight_layout()

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save the figure
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info("GRAPH_SAVED", path=output_path)

    def generate_pairwise_comparisons(self, unique_names: List[str]) -> pd.DataFrame:
        """
        Generate all pairwise comparisons of unique names.

        :param unique_names: List of unique name strings
        :return: DataFrame with NAME_X and NAME_Y columns
        """

        self.logger.info(
            "GENERATING_PAIRWISE_COMPARISONS", unique_names_count=len(unique_names)
        )
        # Generate all combinations (n choose 2)
        pairs = list(combinations(unique_names, 2))

        # Create DataFrame
        df_pairs = pd.DataFrame(pairs, columns=["NAME_X", "NAME_Y"])
        self.logger.info("PAIRWISE_COMPARISONS_GENERATED", pair_count=len(df_pairs))

        return df_pairs

    def predict_matches(
        self, df_pairs: pd.DataFrame, threshold: float = 0.85
    ) -> pd.DataFrame:
        """
        Predict matches for all name pairs using the trained model.

        :param df_pairs: DataFrame with NAME_X and NAME_Y columns
        :param threshold: Classification threshold
        :return: DataFrame with prediction results
        """

        self.logger.info("PREDICTING_MATCHES", pair_count=len(df_pairs))
        # Convert DataFrame to list of dictionaries for batch prediction
        name_pairs = [
            {"name_x": row["NAME_X"], "name_y": row["NAME_Y"]}
            for _, row in df_pairs.iterrows()
        ]

        # Perform batch prediction
        results = self.predictor.predict_batch(name_pairs, threshold=threshold)

        # Add results to DataFrame
        df_pairs = df_pairs.copy()
        df_pairs["prediction"] = [r.get("prediction", 0) for r in results]
        df_pairs["probability"] = [r.get("probability", 0.0) for r in results]
        df_pairs["match_label"] = [r.get("match_label", "NO_MATCH") for r in results]

        # Count the number of matches
        matches = df_pairs[df_pairs["prediction"] == 1]
        self.logger.info(
            "PREDICTION_COMPLETE",
            total_pairs=len(df_pairs),
            matches=len(matches),
        )

        return df_pairs

    def create_matched_graph(self, df_matches: pd.DataFrame) -> nx.Graph:
        """
        Create a graph from matched pairs.

        :param df_matches: DataFrame containing matched pairs
        :return: NetworkX Graph object
        """

        self.logger.info("CREATING_MATCHED_GRAPH")
        G = nx.Graph()

        # Add edges for each matched pair
        for _, row in df_matches.iterrows():
            G.add_edge(row["NAME_X"], row["NAME_Y"])

        self.logger.info(
            "MATCHED_GRAPH_CREATED",
            nodes=G.number_of_nodes(),
            edges=G.number_of_edges(),
        )

        return G

    def detect_communities(self, G: nx.Graph) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        Detect communities using Louvain method for modularity optimization.

        :param G: NetworkX Graph object
        :return: Tuple of (entity_mapping, resolved_names)
                 - entity_mapping: Dictionary mapping node names to entity IDs
                 - resolved_names: Dictionary mapping entity IDs to resolved names (longest string)
        """

        self.logger.info("DETECTING_COMMUNITIES", method="Louvain")
        # Use Louvain method for community detection
        # This optimizes modularity to find better community structure
        communities = nx_community.louvain_communities(G, seed=42)

        # Assign entity IDs to each node
        entity_mapping = {}
        resolved_names = {}

        for entity_id, community in enumerate(communities):
            # Select the longest name in the community as the resolved name
            longest_name = max(community, key=len)
            resolved_names[entity_id] = longest_name

            for node in community:
                entity_mapping[node] = entity_id

        self.logger.info(
            "COMMUNITY_DETECTION_COMPLETE",
            method="Louvain",
            num_communities=len(communities),
        )

        return entity_mapping, resolved_names

    def assign_entity_ids(
        self,
        df_txn: pd.DataFrame,
        entity_mapping: Dict[str, int],
        resolved_names: Dict[int, str],
    ) -> pd.DataFrame:
        """
        Assign entity IDs and resolved names to original transaction data.

        :param df_txn: DataFrame with preprocessed name columns
        :param entity_mapping: Dictionary mapping names to entity IDs
        :param resolved_names: Dictionary mapping entity IDs to resolved names
        :return: DataFrame with ENTITY_X, ENTITY_Y, RESOLVED_NAME_X, RESOLVED_NAME_Y columns
        """

        self.logger.info("ASSIGNING_ENTITY_IDS")
        df_txn = df_txn.copy()

        # Assign entity IDs or use unique negative IDs for unmapped names
        next_entity_id = max(entity_mapping.values()) + 1 if entity_mapping else 0

        def get_entity_id(name: str) -> int:
            nonlocal next_entity_id

            if name in entity_mapping:
                return entity_mapping[name]
            else:
                # Assign a new unique entity ID for unmapped names
                # Because standalone entities are not in any detected community
                entity_id = next_entity_id
                next_entity_id += 1
                entity_mapping[name] = entity_id

                # Use the original name as resolved name for standalone entities
                resolved_names[entity_id] = name

                return entity_id

        df_txn["ENTITY_X"] = df_txn["Cust_Name_Processed"].apply(get_entity_id)
        df_txn["ENTITY_Y"] = df_txn["Counterpart_Name_Processed"].apply(get_entity_id)

        # Map entity IDs to resolved names
        df_txn["RESOLVED_NAME_X"] = df_txn["ENTITY_X"].map(resolved_names)
        df_txn["RESOLVED_NAME_Y"] = df_txn["ENTITY_Y"].map(resolved_names)
        self.logger.info(
            "ENTITY_IDS_ASSIGNED", unique_entities=len(set(entity_mapping.values()))
        )

        return df_txn

    def create_resolved_graph(self, df_resolved: pd.DataFrame) -> nx.Graph:
        """
        Create the final resolved graph using resolved entity names, excluding self-loops.

        :param df: DataFrame with RESOLVED_NAME_X and RESOLVED_NAME_Y columns
        :return: NetworkX Graph object
        """

        self.logger.info("CREATING_RESOLVED_GRAPH")
        G = nx.Graph()

        # Add edges only for non-self connections using resolved names
        for _, row in df_resolved.iterrows():
            resolved_name_x = row["RESOLVED_NAME_X"]
            resolved_name_y = row["RESOLVED_NAME_Y"]

            # Skip self-loops
            if resolved_name_x != resolved_name_y:
                G.add_edge(resolved_name_x, resolved_name_y)

        self.logger.info(
            "RESOLVED_GRAPH_CREATED",
            nodes=G.number_of_nodes(),
            edges=G.number_of_edges(),
        )

        return G

    def run_entity_resolution(
        self,
        input_txn: str,
        output_orig_graph: str,
        output_resolved_graph: str,
        threshold: float = 0.85,
    ) -> Tuple[pd.DataFrame, nx.Graph, nx.Graph]:
        """
        Run the complete entity resolution pipeline.

        :param input_txn: Path to input transaction CSV file
        :param output_orig_graph: Path to save original transaction graph
        :param output_resolved_graph: Path to save resolved graph
        :param threshold: Classification threshold for name matching
        :return: Tuple of (resolved_df, original_graph, resolved_graph)
        """

        self.logger.info("STARTING_ENTITY_RESOLUTION")
        # Step 1: Load transaction data
        df_txn = self.load_transaction_data(input_txn)

        # Step 2: Preprocess names
        df_txn = self.preprocess_names(df_txn)

        # Step 3: Deduplicate transactions
        df_dedup = self.deduplicate_transactions(df_txn)

        # Step 4: Create and visualize original transaction graph
        orig_graph = self.create_transaction_graph(
            df_dedup, "Cust_Name_Processed", "Counterpart_Name_Processed"
        )
        self.visualize_graph(
            orig_graph, output_orig_graph, "Original Transaction Graph"
        )

        # Step 5: Get all unique names
        unique_names = list(
            set(df_dedup["Cust_Name_Processed"].unique()).union(
                set(df_dedup["Counterpart_Name_Processed"].unique())
            )
        )

        # Step 6: Generate pairwise comparisons
        df_pairs = self.generate_pairwise_comparisons(unique_names)

        # Step 7: Predict matches
        df_predictions = self.predict_matches(df_pairs, threshold=threshold)

        # Step 8: Create graph from matched pairs
        df_matches = df_predictions[df_predictions["prediction"] == 1]
        matched_graph = self.create_matched_graph(df_matches)

        # Step 9: Detect communities and get resolved names
        entity_mapping, resolved_names = self.detect_communities(matched_graph)

        # Step 10: Assign entity IDs and resolved names to original transactions
        df_resolved = self.assign_entity_ids(df_dedup, entity_mapping, resolved_names)

        # Step 11: Create and visualize resolved graph (using resolved names)
        resolved_graph = self.create_resolved_graph(df_resolved)
        self.visualize_graph(
            resolved_graph, output_resolved_graph, "Resolved Entity Graph"
        )
        self.logger.info("ENTITY_RESOLUTION_COMPLETE")

        return df_resolved, orig_graph, resolved_graph


def main():
    """
    Main execution function.
    """

    # Configure logging
    configure_structlog(silent=False)
    logger = structlog.get_logger()

    # Initialize entity resolver
    resolver = EntityResolver(logger=logger)

    # Define file paths
    input_csv = "data/raw/sample_txns.csv"
    output_orig_graph = "reports/figures/orig_txn_graph.png"
    output_resolved_graph = "reports/figures/resolved_txn_graph.png"

    # Start clocking the time
    start_time = time.time()

    # Run entity resolution
    df_resolved, orig_graph, resolved_graph = resolver.run_entity_resolution(
        input_txn=input_csv,
        output_orig_graph=output_orig_graph,
        output_resolved_graph=output_resolved_graph,
        threshold=0.85,
    )

    # Display summary
    logger.info(
        "SUMMARY",
        original_graph_nodes=orig_graph.number_of_nodes(),
        original_graph_edges=orig_graph.number_of_edges(),
        resolved_graph_nodes=resolved_graph.number_of_nodes(),
        resolved_graph_edges=resolved_graph.number_of_edges(),
    )

    # Save resolved transactions to CSV
    output_csv = "data/processed/resolved_txns.csv"
    df_resolved.to_csv(output_csv, index=False)
    logger.info("RESOLVED_TRANSACTIONS_SAVED", path=output_csv)

    # Display sample of resolved transactions
    print("\nSample of Resolved Transactions:")
    print(
        df_resolved[
            [
                "FT_No",
                "Cust_Name",
                "Counterpart_Name",
                "ENTITY_X",
                "ENTITY_Y",
                "RESOLVED_NAME_X",
                "RESOLVED_NAME_Y",
            ]
        ].head(10)
    )

    # Compute the time elapsed
    end_time = time.time()
    time_taken = float(end_time - start_time) / 60
    logger.info("TOTAL_RUNTIME", time_taken=round(time_taken, 4), unit="minutes")


if __name__ == "__main__":
    main()
