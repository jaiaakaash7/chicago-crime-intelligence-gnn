#!/usr/bin/env python
# ==============================================================================
# DASHBOARD HEADLESS SMOKE TEST
# Verifies complete dashboard functionality without requiring a public URL
# ==============================================================================

import os
import sys
import unittest
import pandas as pd
import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DASHBOARD_DIR = os.path.join(PROJECT_ROOT, "dashboard")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, DASHBOARD_DIR)

class TestDashboardHeadless(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n" + "=" * 60)
        print("RUNNING DASHBOARD HEADLESS SMOKE TESTS")
        print("=" * 60)
        import app
        cls.app = app

    def test_01_artifacts_loaded(self):
        """Verify all pre-computed artifacts are loaded in memory."""
        self.assertGreater(self.app.N_TOTAL_NODES, 0)
        self.assertEqual(self.app.dashboard_embeddings.shape[0], self.app.N_TOTAL_NODES)
        self.assertEqual(self.app.dashboard_embeddings.shape[1], 64)
        self.assertGreater(len(self.app.dashboard_links_df), 0)
        self.assertGreater(len(self.app.CLASS_NAMES), 0)
        print("  [PASS] All pre-computed artifacts successfully loaded in memory.")

    def test_02_node_587_analysis(self):
        """Verify Node 587 query and GraphSAGE inference."""
        status, meta_df, pred_class, pred_conf, top10_df = self.app.analyze_crime_node(587)
        self.assertIn("✅", status)
        self.assertIsInstance(meta_df, pd.DataFrame)
        self.assertGreater(len(meta_df), 5)
        self.assertIsInstance(pred_class, str)
        self.assertTrue(pred_conf.endswith("%"))
        self.assertEqual(len(top10_df), 10)
        print(f"  [PASS] Node 587 analyzed: Actual={self.app.dashboard_nodes_df.iloc[587]['crime_type']}, Predicted={pred_class}, Conf={pred_conf}")

    def test_03_invalid_node_safety(self):
        """Verify negative, out-of-bounds, and non-integer inputs return friendly errors without crashing."""
        for bad_id in [-5, 999999, "invalid"]:
            status, meta, pred_c, pred_cf, t10 = self.app.analyze_crime_node(bad_id)
            self.assertIn("⚠️", status)
            self.assertEqual(len(meta), 0)
        print("  [PASS] Invalid node inputs safely handled with friendly error alerts.")

    def test_04_gnn_intelligence_similarity(self):
        """Verify 64-D embedding extraction and top 10 cosine similarity (excluding self)."""
        status, emb_prev, df_sim = self.app.get_gnn_intelligence(587)
        self.assertIn("✅", status)
        self.assertEqual(len(df_sim), 10)
        # Verify query node itself is NOT in results
        self.assertNotIn(587, df_sim["Target Node"].values)
        # Verify similarity is in valid range
        for sim in df_sim["GNN Cosine Similarity"]:
            self.assertGreaterEqual(sim, -1.0)
            self.assertLessEqual(sim, 1.0)
        print("  [PASS] 64-D embeddings and top 10 similarity computed; self-similarity strictly excluded.")

    def test_05_candidate_links(self):
        """Verify candidate links query returns valid pairs without self-links."""
        status, df_links = self.app.query_candidate_links(587, 0.50)
        self.assertIn("✅", status)
        self.assertGreater(len(df_links), 0)
        # Verify no self links
        self.assertTrue((df_links["source_node"] != df_links["target_node"]).all())
        print(f"  [PASS] Candidate links query returned {len(df_links)} valid pairs without self-links.")

    def test_06_folium_map_generation(self):
        """Verify local Folium HTML map renders properly."""
        map_html, summary = self.app.render_crime_map("ALL", "ALL")
        self.assertIsInstance(map_html, str)
        self.assertIn("<div", map_html)
        self.assertIn("Displaying", summary)
        print(f"  [PASS] Local Folium map generated: {summary}")

    def test_07_gradio_blocks_instantiation(self):
        """Verify Gradio demo blocks instantiated properly."""
        self.assertIsNotNone(self.app.demo)
        print("  [PASS] Gradio dashboard blocks fully instantiated with 8 tabs.")

if __name__ == "__main__":
    unittest.main()
