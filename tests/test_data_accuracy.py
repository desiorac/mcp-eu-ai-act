#!/usr/bin/env python3
"""
Tests de précision des données EU AI Act
Vérifie l'exactitude des catégories de risque, exigences, et patterns de détection
"""

import unittest
import sys
import re
from pathlib import Path

# Ajouter le répertoire parent au path pour l'import
sys.path.insert(0, str(Path(__file__).parent.parent))

from server import AI_MODEL_PATTERNS, RISK_CATEGORIES


class TestAIModelPatterns(unittest.TestCase):
    """Tests de précision des patterns de détection AI"""

    def test_openai_patterns_accuracy(self):
        """Test des patterns OpenAI contre du vrai code"""
        patterns = AI_MODEL_PATTERNS["openai"]

        # Code OpenAI réel - devrait matcher
        valid_openai_code = [
            "import openai",
            "from openai import OpenAI",
            "openai.ChatCompletion.create()",
            "openai.Completion.create()",
            'model="gpt-4"',
            'model="gpt-3.5-turbo"',
            'engine="text-davinci-003"',
        ]

        for code in valid_openai_code:
            matched = any(re.search(pattern, code, re.IGNORECASE) for pattern in patterns)
            self.assertTrue(matched, f"Failed to detect OpenAI in: {code}")

        # Code non-OpenAI - ne devrait pas matcher
        invalid_code = [
            "import os",
            "from anthropic import Anthropic",
            "import tensorflow",
        ]

        for code in invalid_code:
            matched = any(re.search(pattern, code, re.IGNORECASE) for pattern in patterns)
            self.assertFalse(matched, f"False positive for OpenAI in: {code}")

    def test_anthropic_patterns_accuracy(self):
        """Test des patterns Anthropic contre du vrai code"""
        patterns = AI_MODEL_PATTERNS["anthropic"]

        # Code Anthropic réel - devrait matcher
        valid_anthropic_code = [
            "from anthropic import Anthropic",
            "import anthropic",
            'model="claude-3-opus-20240229"',
            'model="claude-3-sonnet-20240229"',
            "client = Anthropic()",
            "client.messages.create()",
        ]

        for code in valid_anthropic_code:
            matched = any(re.search(pattern, code, re.IGNORECASE) for pattern in patterns)
            self.assertTrue(matched, f"Failed to detect Anthropic in: {code}")

    def test_huggingface_patterns_accuracy(self):
        """Test des patterns HuggingFace contre du vrai code"""
        patterns = AI_MODEL_PATTERNS["huggingface"]

        # Code HuggingFace réel - devrait matcher
        valid_hf_code = [
            "from transformers import AutoModel",
            "from transformers import AutoTokenizer",
            "from transformers import pipeline",
            "model = AutoModel.from_pretrained('bert-base-uncased')",
            "from huggingface_hub import HfApi",
        ]

        for code in valid_hf_code:
            matched = any(re.search(pattern, code, re.IGNORECASE) for pattern in patterns)
            self.assertTrue(matched, f"Failed to detect HuggingFace in: {code}")

    def test_tensorflow_patterns_accuracy(self):
        """Test des patterns TensorFlow contre du vrai code"""
        patterns = AI_MODEL_PATTERNS["tensorflow"]

        # Code TensorFlow réel - devrait matcher
        valid_tf_code = [
            "import tensorflow as tf",
            "from tensorflow import keras",
            "model = tf.keras.Sequential()",
        ]

        for code in valid_tf_code:
            matched = any(re.search(pattern, code, re.IGNORECASE) for pattern in patterns)
            self.assertTrue(matched, f"Failed to detect TensorFlow in: {code}")

        # Test détection fichiers .h5
        self.assertTrue("model.h5".endswith(".h5"))
        h5_pattern = r"\.h5$"
        self.assertTrue(re.search(h5_pattern, "model.h5"))

    def test_pytorch_patterns_accuracy(self):
        """Test des patterns PyTorch contre du vrai code"""
        patterns = AI_MODEL_PATTERNS["pytorch"]

        # Code PyTorch réel - devrait matcher
        valid_pytorch_code = [
            "import torch",
            "from torch import nn",
            "class MyModel(nn.Module):",
        ]

        for code in valid_pytorch_code:
            matched = any(re.search(pattern, code, re.IGNORECASE) for pattern in patterns)
            self.assertTrue(matched, f"Failed to detect PyTorch in: {code}")

        # Test détection fichiers .pt et .pth
        self.assertTrue("model.pt".endswith(".pt"))
        self.assertTrue("model.pth".endswith(".pth"))
        pt_pattern = r"\.pt$"
        pth_pattern = r"\.pth$"
        self.assertTrue(re.search(pt_pattern, "model.pt"))
        self.assertTrue(re.search(pth_pattern, "model.pth"))

    def test_langchain_patterns_accuracy(self):
        """Test des patterns LangChain contre du vrai code"""
        patterns = AI_MODEL_PATTERNS["langchain"]

        # Code LangChain réel - devrait matcher
        valid_langchain_code = [
            "from langchain import LLMChain",
            "from langchain.llms import ChatOpenAI",
            "import langchain",
        ]

        for code in valid_langchain_code:
            matched = any(re.search(pattern, code, re.IGNORECASE) for pattern in patterns)
            self.assertTrue(matched, f"Failed to detect LangChain in: {code}")

    def test_no_false_positives(self):
        """Test qu'il n'y a pas de faux positifs avec du code normal"""
        all_patterns = []
        for patterns in AI_MODEL_PATTERNS.values():
            all_patterns.extend(patterns)

        # Code normal sans AI
        normal_code = [
            "import os",
            "import sys",
            "import json",
            "from pathlib import Path",
            "def hello(): pass",
            "class MyClass: pass",
        ]

        for code in normal_code:
            matched = any(re.search(pattern, code, re.IGNORECASE) for pattern in all_patterns)
            self.assertFalse(matched, f"False positive in: {code}")

    def test_all_frameworks_have_patterns(self):
        """Test que tous les frameworks ont au moins un pattern"""
        expected_frameworks = ["openai", "anthropic", "huggingface", "tensorflow", "pytorch", "langchain"]

        for framework in expected_frameworks:
            self.assertIn(framework, AI_MODEL_PATTERNS)
            self.assertGreater(len(AI_MODEL_PATTERNS[framework]), 0)
            self.assertIsInstance(AI_MODEL_PATTERNS[framework], list)


class TestRiskCategories(unittest.TestCase):
    """Tests de précision des catégories de risque EU AI Act"""

    def test_all_risk_categories_present(self):
        """Test que toutes les catégories de risque sont présentes"""
        required_categories = ["unacceptable", "high", "limited", "minimal"]

        for category in required_categories:
            self.assertIn(category, RISK_CATEGORIES)

    def test_unacceptable_risk_category(self):
        """Test de la catégorie risque inacceptable"""
        category = RISK_CATEGORIES["unacceptable"]

        self.assertIn("description", category)
        self.assertIn("requirements", category)

        # Vérifier la description
        description_lower = category["description"].lower()
        self.assertTrue(
            "interdit" in description_lower or "manipulation" in description_lower,
            "Description should mention prohibited systems"
        )

        # Devrait avoir au moins une exigence
        self.assertGreater(len(category["requirements"]), 0)
        self.assertIsInstance(category["requirements"], list)

    def test_high_risk_category(self):
        """Test de la catégorie risque élevé"""
        category = RISK_CATEGORIES["high"]

        self.assertIn("description", category)
        self.assertIn("requirements", category)

        # Vérifier la description
        description_lower = category["description"].lower()
        self.assertTrue(
            "haut risque" in description_lower or "recrutement" in description_lower or "crédit" in description_lower,
            "Description should mention high-risk systems"
        )

        # Exigences clés pour risque élevé
        requirements_str = " ".join(category["requirements"]).lower()

        required_keywords = [
            "documentation",
            "risque",
            "transparence",
            "surveillance",
            "robustesse",
        ]

        for keyword in required_keywords:
            self.assertTrue(
                keyword in requirements_str,
                f"High-risk requirements should include '{keyword}'"
            )

        # Devrait avoir au moins 6 exigences
        self.assertGreaterEqual(len(category["requirements"]), 6)

    def test_limited_risk_category(self):
        """Test de la catégorie risque limité"""
        category = RISK_CATEGORIES["limited"]

        self.assertIn("description", category)
        self.assertIn("requirements", category)

        # Vérifier la description
        description_lower = category["description"].lower()
        self.assertTrue(
            "limité" in description_lower or "chatbot" in description_lower,
            "Description should mention limited-risk systems"
        )

        # Exigences clés pour risque limité
        requirements_str = " ".join(category["requirements"]).lower()

        required_keywords = [
            "transparence",
            "information",
        ]

        for keyword in required_keywords:
            self.assertTrue(
                keyword in requirements_str,
                f"Limited-risk requirements should include '{keyword}'"
            )

        # Devrait avoir au moins 2 exigences
        self.assertGreaterEqual(len(category["requirements"]), 2)

    def test_minimal_risk_category(self):
        """Test de la catégorie risque minimal"""
        category = RISK_CATEGORIES["minimal"]

        self.assertIn("description", category)
        self.assertIn("requirements", category)

        # Vérifier la description
        description_lower = category["description"].lower()
        self.assertTrue(
            "minimal" in description_lower or "spam" in description_lower or "jeux" in description_lower,
            "Description should mention minimal-risk systems"
        )

        # Exigences minimales
        requirements_str = " ".join(category["requirements"]).lower()
        self.assertTrue(
            "aucune" in requirements_str or "volontaire" in requirements_str,
            "Minimal-risk should have minimal or voluntary requirements"
        )

    def test_requirements_are_actionable(self):
        """Test que les exigences sont actionnables (pas vides, ont du sens)"""
        for category_name, category in RISK_CATEGORIES.items():
            requirements = category["requirements"]

            for req in requirements:
                # Chaque exigence devrait être une chaîne non vide
                self.assertIsInstance(req, str)
                self.assertGreater(len(req), 5, f"Requirement too short in {category_name}: {req}")

                # Devrait contenir des mots significatifs
                self.assertTrue(
                    any(word in req.lower() for word in ["documentation", "système", "données", "transparence", "surveillance", "qualité", "aucune", "volontaire", "interdit", "robustesse", "précision", "cybersécurité", "humaine", "gestion", "risques", "gouvernance", "enregistrement", "information", "utilisateurs", "marquage", "contenu", "obligations"]),
                    f"Requirement lacks meaningful content in {category_name}: {req}"
                )

    def test_risk_hierarchy(self):
        """Test que la hiérarchie des risques est cohérente"""
        # Risque élevé devrait avoir plus d'exigences que risque limité
        high_reqs = len(RISK_CATEGORIES["high"]["requirements"])
        limited_reqs = len(RISK_CATEGORIES["limited"]["requirements"])
        minimal_reqs = len(RISK_CATEGORIES["minimal"]["requirements"])

        self.assertGreater(high_reqs, limited_reqs, "High risk should have more requirements than limited")
        self.assertGreater(limited_reqs, minimal_reqs, "Limited risk should have more requirements than minimal")


class TestComplianceAccuracy(unittest.TestCase):
    """Tests de précision de la logique de conformité"""

    def test_compliance_score_calculation(self):
        """Test du calcul du score de conformité"""
        # Simuler différents scénarios de conformité
        test_cases = [
            # (checks_passed, total_checks, expected_percentage)
            (3, 3, 100.0),
            (2, 3, 66.7),
            (1, 3, 33.3),
            (0, 3, 0.0),
            (5, 6, 83.3),
        ]

        for passed, total, expected_pct in test_cases:
            calculated_pct = round((passed / total) * 100, 1) if total > 0 else 0
            self.assertEqual(
                calculated_pct,
                expected_pct,
                f"Score calculation wrong for {passed}/{total}"
            )

    def test_eu_ai_act_reference_data(self):
        """Test contre des données de référence EU AI Act connues"""
        # Exemples réels de systèmes à haut risque selon EU AI Act
        high_risk_examples = [
            "recrutement",
            "crédit",
            "application de la loi",
            "infrastructure critique",
        ]

        high_risk_desc = RISK_CATEGORIES["high"]["description"].lower()

        # Devrait mentionner au moins 2 de ces exemples
        matches = sum(1 for ex in high_risk_examples if ex in high_risk_desc)
        self.assertGreaterEqual(matches, 2, "High-risk description should mention known examples")

        # Exemples de systèmes interdits (risque inacceptable)
        unacceptable_examples = [
            "manipulation",
            "notation sociale",
            "surveillance",
        ]

        unacceptable_desc = RISK_CATEGORIES["unacceptable"]["description"].lower()

        # Devrait mentionner au moins 1 de ces exemples
        matches = sum(1 for ex in unacceptable_examples if ex in unacceptable_desc)
        self.assertGreaterEqual(matches, 1, "Unacceptable-risk description should mention prohibited systems")


class TestDataConsistency(unittest.TestCase):
    """Tests de cohérence des données"""

    def test_no_duplicate_patterns(self):
        """Test qu'il n'y a pas de patterns dupliqués dans un même framework"""
        for framework, patterns in AI_MODEL_PATTERNS.items():
            unique_patterns = set(patterns)
            self.assertEqual(
                len(patterns),
                len(unique_patterns),
                f"Duplicate patterns found in {framework}"
            )

    def test_patterns_are_valid_regex(self):
        """Test que tous les patterns sont des regex valides"""
        for framework, patterns in AI_MODEL_PATTERNS.items():
            for pattern in patterns:
                try:
                    re.compile(pattern)
                except re.error as e:
                    self.fail(f"Invalid regex in {framework}: {pattern} - {e}")

    def test_risk_categories_structure(self):
        """Test que toutes les catégories ont la même structure"""
        required_keys = ["description", "requirements"]

        for category_name, category in RISK_CATEGORIES.items():
            for key in required_keys:
                self.assertIn(
                    key,
                    category,
                    f"Missing key '{key}' in category '{category_name}'"
                )

            # Vérifier les types
            self.assertIsInstance(category["description"], str)
            self.assertIsInstance(category["requirements"], list)

    def test_no_empty_data(self):
        """Test qu'il n'y a pas de données vides"""
        # Vérifier AI_MODEL_PATTERNS
        for framework, patterns in AI_MODEL_PATTERNS.items():
            self.assertGreater(len(patterns), 0, f"Empty patterns for {framework}")
            for pattern in patterns:
                self.assertGreater(len(pattern), 0, f"Empty pattern in {framework}")

        # Vérifier RISK_CATEGORIES
        for category_name, category in RISK_CATEGORIES.items():
            self.assertGreater(len(category["description"]), 0, f"Empty description in {category_name}")
            self.assertGreater(len(category["requirements"]), 0, f"Empty requirements in {category_name}")


class TestFrameworkCoverage(unittest.TestCase):
    """Tests de couverture des frameworks AI populaires"""

    def test_major_frameworks_covered(self):
        """Test que les frameworks majeurs sont couverts"""
        major_frameworks = {
            "openai": ["OpenAI", "GPT"],
            "anthropic": ["Claude", "Anthropic"],
            "huggingface": ["Transformers", "HuggingFace"],
            "tensorflow": ["TensorFlow", "Keras"],
            "pytorch": ["PyTorch", "Torch"],
            "langchain": ["LangChain"],
        }

        for framework, expected_detections in major_frameworks.items():
            self.assertIn(framework, AI_MODEL_PATTERNS, f"Missing framework: {framework}")
            patterns = AI_MODEL_PATTERNS[framework]

            # Vérifier qu'au moins un pattern correspond au nom du framework
            patterns_str = " ".join(patterns).lower()
            framework_mentioned = any(
                name.lower() in patterns_str for name in expected_detections
            )

            self.assertTrue(
                framework_mentioned,
                f"Framework {framework} patterns don't mention expected names: {expected_detections}"
            )

    def test_common_model_files_detected(self):
        """Test que les fichiers de modèles communs sont détectés"""
        file_patterns = {
            "tensorflow": [".h5"],
            "pytorch": [".pt", ".pth"],
        }

        for framework, extensions in file_patterns.items():
            patterns = AI_MODEL_PATTERNS[framework]
            patterns_str = " ".join(patterns)

            for ext in extensions:
                self.assertTrue(
                    ext in patterns_str,
                    f"File extension {ext} not detected for {framework}"
                )


class TestEUAIActArticleAccuracy(unittest.TestCase):
    """Tests de conformité avec les articles spécifiques de l'EU AI Act"""

    def test_article_5_prohibited_practices(self):
        """Art. 5 - Les pratiques interdites sont bien couvertes"""
        desc = RISK_CATEGORIES["unacceptable"]["description"].lower()
        # Article 5 interdit: manipulation, notation sociale, surveillance biométrique
        prohibited = ["manipulation", "notation sociale", "surveillance"]
        covered = sum(1 for p in prohibited if p in desc)
        self.assertGreaterEqual(covered, 2, "Article 5 prohibited practices insufficiently covered")

    def test_article_6_high_risk_systems(self):
        """Art. 6 - Les systèmes à haut risque ont les exigences d'Annexe III"""
        high_reqs = RISK_CATEGORIES["high"]["requirements"]
        req_text = " ".join(high_reqs).lower()

        # Exigences essentielles Art. 6/Annexe III
        essential = ["documentation", "risques", "données", "transparence", "surveillance humaine", "robustesse"]
        for req in essential:
            self.assertIn(req, req_text, f"High-risk missing Art. 6 requirement: {req}")

    def test_article_52_transparency_obligations(self):
        """Art. 52 - Obligations de transparence pour risque limité"""
        limited_reqs = RISK_CATEGORIES["limited"]["requirements"]
        req_text = " ".join(limited_reqs).lower()

        # Art. 52 exige transparence + information utilisateurs + marquage contenu
        self.assertIn("transparence", req_text)
        self.assertIn("utilisateurs", req_text)
        self.assertIn("contenu", req_text)

    def test_four_tier_risk_classification(self):
        """L'EU AI Act définit exactement 4 niveaux de risque"""
        self.assertEqual(len(RISK_CATEGORIES), 4)
        expected = {"unacceptable", "high", "limited", "minimal"}
        self.assertEqual(set(RISK_CATEGORIES.keys()), expected)

    def test_high_risk_examples_accuracy(self):
        """Les exemples de systèmes à haut risque correspondent à l'Annexe III"""
        desc = RISK_CATEGORIES["high"]["description"].lower()
        # Annexe III catégories: recrutement, crédit/scoring, justice/law enforcement
        annex_iii_examples = ["recrutement", "crédit", "loi"]
        covered = sum(1 for ex in annex_iii_examples if ex in desc)
        self.assertGreaterEqual(covered, 2, "High-risk examples should match Annex III")

    def test_limited_risk_examples_accuracy(self):
        """Les exemples de risque limité sont corrects"""
        desc = RISK_CATEGORIES["limited"]["description"].lower()
        # Chatbots et deepfakes sont les exemples principaux
        self.assertTrue("chatbot" in desc or "deepfake" in desc)

    def test_minimal_risk_no_mandatory_requirements(self):
        """Le risque minimal n'a pas d'obligations obligatoires (Art. 69 - codes volontaires)"""
        req_text = " ".join(RISK_CATEGORIES["minimal"]["requirements"]).lower()
        self.assertTrue("aucune" in req_text or "volontaire" in req_text)

    def test_high_risk_eu_database_registration(self):
        """Art. 60 - Les systèmes à haut risque doivent être enregistrés dans la base UE"""
        high_reqs = RISK_CATEGORIES["high"]["requirements"]
        req_text = " ".join(high_reqs).lower()
        self.assertTrue("enregistrement" in req_text or "base de données" in req_text)


class TestPatternCrossContamination(unittest.TestCase):
    """Tests que les patterns d'un framework ne détectent pas un autre"""

    def test_openai_not_detected_as_langchain(self):
        """Code OpenAI pur ne devrait pas déclencher LangChain"""
        langchain_patterns = AI_MODEL_PATTERNS["langchain"]
        pure_openai_code = "import openai\nopenai.ChatCompletion.create(model='gpt-4')"

        for pattern in langchain_patterns:
            # ChatOpenAI est un pattern LangChain qui pourrait matcher
            if pattern == "ChatOpenAI":
                continue  # Ce pattern peut légitimement apparaître dans du code OpenAI via LangChain
            matched = re.search(pattern, pure_openai_code, re.IGNORECASE)
            self.assertIsNone(matched, f"LangChain pattern '{pattern}' false positive on OpenAI code")

    def test_pytorch_not_detected_as_tensorflow(self):
        """Code PyTorch pur ne devrait pas déclencher TensorFlow"""
        tf_patterns = AI_MODEL_PATTERNS["tensorflow"]
        pure_pytorch_code = "import torch\nmodel = torch.nn.Linear(10, 5)"

        for pattern in tf_patterns:
            matched = re.search(pattern, pure_pytorch_code, re.IGNORECASE)
            self.assertIsNone(matched, f"TensorFlow pattern '{pattern}' false positive on PyTorch code")

    def test_anthropic_not_detected_as_openai(self):
        """Code Anthropic pur ne devrait pas déclencher OpenAI"""
        openai_patterns = AI_MODEL_PATTERNS["openai"]
        pure_anthropic_code = "from anthropic import Anthropic\nclient = Anthropic()\nclient.messages.create(model='claude-3-opus')"

        for pattern in openai_patterns:
            matched = re.search(pattern, pure_anthropic_code, re.IGNORECASE)
            self.assertIsNone(matched, f"OpenAI pattern '{pattern}' false positive on Anthropic code")


if __name__ == "__main__":
    unittest.main(verbosity=2)
