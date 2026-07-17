import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepositoryContractTests(unittest.TestCase):
    def test_configs_contain_reported_core_settings(self):
        for name in ("dlpfc4.json", "dlpfc12.json", "mouse_embryo.json"):
            config = json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))
            self.assertEqual(config["hidden_dims"], [512, 30])
            self.assertEqual(config["n_epochs"], 1000)
            self.assertEqual(config["pretrain_epochs"], 500)
            self.assertEqual(config["update_interval"], 100)
        for name in ("dlpfc4.json", "dlpfc12.json"):
            config = json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))
            self.assertEqual(config["mclust_model"], "EEE")

    def test_training_default_is_label_free_and_named_sparcl(self):
        source = (ROOT / "SpaRCL_GitHub" / "SpaRCL" / "train_SpaRCL.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "train_sparcl"
        )
        defaults = dict(zip((arg.arg for arg in function.args.kwonlyargs), function.args.kw_defaults))
        self.assertFalse(ast.literal_eval(defaults["use_label_filter"]))
        self.assertEqual(ast.literal_eval(defaults["key_added"]), "SpaRCL")
        self.assertIn("negative_distances < positive_distance + margin", source)

    def test_mouse_embryo_settings_match_the_executed_workflow(self):
        config = json.loads((ROOT / "configs" / "mouse_embryo.json").read_text(encoding="utf-8"))
        self.assertEqual(config["radius_cutoff"], 1.3)
        self.assertEqual(config["common_genes_in_revision_run"], 693)
        self.assertEqual(config["mnn_iter_comb"], [[0, 3], [1, 3], [2, 3]])
        self.assertEqual(config["clustering_algorithm"], "Louvain")
        self.assertEqual(config["training_seed"], 666)
        self.assertEqual(config["clustering_seed"], 666)
        self.assertEqual(config["louvain_resolution"], 0.5)
        self.assertEqual(config["expected_cluster_count"], 15)
        source = (ROOT / "examples" / "run_mouse_embryo.py").read_text(encoding="utf-8")
        self.assertIn("sc.tl.louvain", source)
        self.assertNotIn("mclust", source.lower())


if __name__ == "__main__":
    unittest.main()
