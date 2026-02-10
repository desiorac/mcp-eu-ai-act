#!/usr/bin/env python3
"""
Tests unitaires pour le serveur MCP EU AI Act
Teste les outils MCP individuellement
"""

import unittest
import sys
import os
import tempfile
import shutil
from pathlib import Path

# Ajouter le répertoire parent au path pour l'import
sys.path.insert(0, str(Path(__file__).parent.parent))

from server import (
    EUAIActChecker,
    scan_project_tool,
    check_compliance_tool,
    generate_report_tool,
    MCPServer,
    AI_MODEL_PATTERNS,
    RISK_CATEGORIES,
)


class TestEUAIActChecker(unittest.TestCase):
    """Tests pour la classe EUAIActChecker"""

    def setUp(self):
        """Créer un projet de test temporaire"""
        self.test_dir = tempfile.mkdtemp()
        self.project_path = Path(self.test_dir) / "test_project"
        self.project_path.mkdir()

    def tearDown(self):
        """Nettoyer le projet de test"""
        shutil.rmtree(self.test_dir)

    def test_init(self):
        """Test d'initialisation du checker"""
        checker = EUAIActChecker(str(self.project_path))
        self.assertEqual(checker.project_path, self.project_path)
        self.assertEqual(checker.files_scanned, 0)
        self.assertEqual(checker.detected_models, {})
        self.assertEqual(checker.ai_files, [])

    def test_scan_empty_project(self):
        """Test de scan d'un projet vide"""
        checker = EUAIActChecker(str(self.project_path))
        results = checker.scan_project()

        self.assertEqual(results["files_scanned"], 0)
        self.assertEqual(results["ai_files"], [])
        self.assertEqual(results["detected_models"], {})

    def test_scan_project_with_openai(self):
        """Test de détection de code OpenAI"""
        # Créer un fichier avec du code OpenAI
        test_file = self.project_path / "main.py"
        test_file.write_text("""
import openai

def chat():
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}]
    )
    return response
""")

        checker = EUAIActChecker(str(self.project_path))
        results = checker.scan_project()

        self.assertEqual(results["files_scanned"], 1)
        self.assertIn("openai", results["detected_models"])
        self.assertEqual(len(results["ai_files"]), 1)
        self.assertEqual(results["ai_files"][0]["file"], "main.py")
        self.assertIn("openai", results["ai_files"][0]["frameworks"])

    def test_scan_project_with_anthropic(self):
        """Test de détection de code Anthropic"""
        test_file = self.project_path / "ai.py"
        test_file.write_text("""
from anthropic import Anthropic

client = Anthropic()
message = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}]
)
""")

        checker = EUAIActChecker(str(self.project_path))
        results = checker.scan_project()

        self.assertEqual(results["files_scanned"], 1)
        self.assertIn("anthropic", results["detected_models"])

    def test_scan_project_multiple_frameworks(self):
        """Test de détection de plusieurs frameworks"""
        # Fichier avec OpenAI
        (self.project_path / "openai_code.py").write_text("import openai")
        # Fichier avec Anthropic
        (self.project_path / "anthropic_code.py").write_text("from anthropic import Anthropic")
        # Fichier avec HuggingFace
        (self.project_path / "hf_code.py").write_text("from transformers import AutoModel")

        checker = EUAIActChecker(str(self.project_path))
        results = checker.scan_project()

        self.assertEqual(results["files_scanned"], 3)
        self.assertIn("openai", results["detected_models"])
        self.assertIn("anthropic", results["detected_models"])
        self.assertIn("huggingface", results["detected_models"])
        self.assertEqual(len(results["ai_files"]), 3)

    def test_scan_project_non_existent(self):
        """Test de scan d'un projet inexistant"""
        checker = EUAIActChecker("/non/existent/path")
        results = checker.scan_project()

        self.assertIn("error", results)
        self.assertIn("does not exist", results["error"])

    def test_check_compliance_invalid_category(self):
        """Test de vérification avec catégorie invalide"""
        checker = EUAIActChecker(str(self.project_path))
        results = checker.check_compliance("invalid_category")

        self.assertIn("error", results)
        self.assertIn("Invalid risk category", results["error"])

    def test_check_compliance_limited_risk(self):
        """Test de vérification pour risque limité"""
        # Créer un README
        (self.project_path / "README.md").write_text("""
# Test Project

This project uses AI and machine learning.
""")

        checker = EUAIActChecker(str(self.project_path))
        results = checker.check_compliance("limited")

        self.assertEqual(results["risk_category"], "limited")
        self.assertIn("compliance_status", results)
        self.assertIn("transparence", results["compliance_status"])
        self.assertIn("compliance_score", results)
        self.assertIn("compliance_percentage", results)

    def test_check_compliance_high_risk(self):
        """Test de vérification pour risque élevé"""
        # Créer documentation minimale
        (self.project_path / "README.md").write_text("# Test")

        checker = EUAIActChecker(str(self.project_path))
        results = checker.check_compliance("high")

        self.assertEqual(results["risk_category"], "high")
        self.assertIn("documentation_technique", results["compliance_status"])
        self.assertIn("gestion_risques", results["compliance_status"])
        self.assertIn("transparence", results["compliance_status"])

    def test_check_compliance_minimal_risk(self):
        """Test de vérification pour risque minimal"""
        (self.project_path / "README.md").write_text("# Basic project")

        checker = EUAIActChecker(str(self.project_path))
        results = checker.check_compliance("minimal")

        self.assertEqual(results["risk_category"], "minimal")
        self.assertIn("documentation_basique", results["compliance_status"])
        self.assertTrue(results["compliance_status"]["documentation_basique"])

    def test_check_technical_docs(self):
        """Test de vérification de documentation technique"""
        checker = EUAIActChecker(str(self.project_path))

        # Sans documentation
        self.assertFalse(checker._check_technical_docs())

        # Avec README
        (self.project_path / "README.md").write_text("# Docs")
        self.assertTrue(checker._check_technical_docs())

        # Avec dossier docs
        (self.project_path / "docs").mkdir()
        self.assertTrue(checker._check_technical_docs())

    def test_check_file_exists(self):
        """Test de vérification d'existence de fichier"""
        checker = EUAIActChecker(str(self.project_path))

        # Fichier inexistant
        self.assertFalse(checker._check_file_exists("NONEXISTENT.md"))

        # Fichier à la racine
        (self.project_path / "TEST.md").write_text("test")
        self.assertTrue(checker._check_file_exists("TEST.md"))

        # Fichier dans docs/
        (self.project_path / "docs").mkdir()
        (self.project_path / "docs" / "TEST2.md").write_text("test")
        self.assertTrue(checker._check_file_exists("TEST2.md"))

    def test_check_ai_disclosure(self):
        """Test de vérification de divulgation AI"""
        checker = EUAIActChecker(str(self.project_path))

        # Sans README
        self.assertFalse(checker._check_ai_disclosure())

        # Avec README sans mention AI
        (self.project_path / "README.md").write_text("# Project")
        self.assertFalse(checker._check_ai_disclosure())

        # Avec README mentionnant AI
        (self.project_path / "README.md").write_text("# AI Project using GPT-4")
        self.assertTrue(checker._check_ai_disclosure())

    def test_check_content_marking(self):
        """Test de vérification de marquage de contenu"""
        checker = EUAIActChecker(str(self.project_path))

        # Sans marqueurs
        self.assertFalse(checker._check_content_marking())

        # Avec marqueur
        (self.project_path / "generated.py").write_text("""
# This code is generated by AI
def hello():
    pass
""")
        self.assertTrue(checker._check_content_marking())

    def test_generate_report(self):
        """Test de génération de rapport complet"""
        # Créer un projet avec AI
        (self.project_path / "main.py").write_text("import openai")
        (self.project_path / "README.md").write_text("# AI Project using AI")

        checker = EUAIActChecker(str(self.project_path))
        scan_results = checker.scan_project()
        compliance_results = checker.check_compliance("limited")
        report = checker.generate_report(scan_results, compliance_results)

        # Vérifier la structure du rapport
        self.assertIn("report_date", report)
        self.assertIn("project_path", report)
        self.assertIn("scan_summary", report)
        self.assertIn("compliance_summary", report)
        self.assertIn("detailed_findings", report)
        self.assertIn("recommendations", report)

        # Vérifier le contenu
        self.assertEqual(report["scan_summary"]["files_scanned"], 1)
        self.assertEqual(report["compliance_summary"]["risk_category"], "limited")
        self.assertGreater(len(report["recommendations"]), 0)

    def test_generate_recommendations(self):
        """Test de génération de recommandations"""
        checker = EUAIActChecker(str(self.project_path))

        # Toutes les vérifications passées
        compliance_results = {
            "risk_category": "limited",
            "compliance_status": {
                "transparence": True,
                "information_utilisateurs": True,
                "marquage_contenu": True,
            }
        }
        recommendations = checker._generate_recommendations(compliance_results)
        self.assertIn("✅", recommendations[0])

        # Vérifications échouées
        compliance_results["compliance_status"]["transparence"] = False
        recommendations = checker._generate_recommendations(compliance_results)
        self.assertTrue(any("❌" in r for r in recommendations))

    def test_scan_file_with_error(self):
        """Test de scan de fichier avec erreur de lecture"""
        # Créer un fichier binaire qui causera une erreur
        binary_file = self.project_path / "test.py"
        binary_file.write_bytes(b'\x00\x01\x02\x03')

        checker = EUAIActChecker(str(self.project_path))
        # Ne devrait pas lever d'exception
        results = checker.scan_project()
        self.assertEqual(results["files_scanned"], 1)

    def test_scan_file_permission_error(self):
        """Test de scan de fichier inaccessible (couvre except branch dans _scan_file)"""
        import os
        test_file = self.project_path / "secret.py"
        test_file.write_text("import openai")
        os.chmod(str(test_file), 0o000)

        checker = EUAIActChecker(str(self.project_path))
        results = checker.scan_project()
        # Fichier compté mais pas de détection (erreur de lecture)
        self.assertEqual(results["files_scanned"], 1)
        self.assertEqual(len(results["ai_files"]), 0)

        # Restaurer les permissions pour le tearDown
        os.chmod(str(test_file), 0o644)

    def test_content_marking_unreadable_file(self):
        """Test _check_content_marking avec fichier illisible (couvre bare except)"""
        import os
        test_file = self.project_path / "gen.py"
        test_file.write_text("# generated by ai\ndef x(): pass")
        os.chmod(str(test_file), 0o000)

        checker = EUAIActChecker(str(self.project_path))
        # Ne devrait pas lever d'exception malgré fichier illisible
        result = checker._check_content_marking()
        self.assertFalse(result)

        os.chmod(str(test_file), 0o644)

    def test_check_compliance_unacceptable_risk(self):
        """Test de vérification pour risque inacceptable"""
        checker = EUAIActChecker(str(self.project_path))
        results = checker.check_compliance("unacceptable")

        self.assertEqual(results["risk_category"], "unacceptable")
        # Risque inacceptable: aucune check de compliance, score 0/0
        self.assertEqual(len(results["compliance_status"]), 0)

    def test_scan_ignores_non_code_files(self):
        """Test que le scan ignore les fichiers non-code"""
        (self.project_path / "data.json").write_text('{"key": "import openai"}')
        (self.project_path / "readme.txt").write_text("import openai")
        (self.project_path / "image.png").write_bytes(b'\x89PNG')

        checker = EUAIActChecker(str(self.project_path))
        results = checker.scan_project()

        self.assertEqual(results["files_scanned"], 0)
        self.assertEqual(len(results["ai_files"]), 0)

    def test_scan_all_supported_extensions(self):
        """Test que toutes les extensions supportées sont scannées"""
        extensions = [".py", ".js", ".ts", ".java", ".go", ".rs", ".cpp", ".c"]
        for ext in extensions:
            (self.project_path / f"file{ext}").write_text("import openai")

        checker = EUAIActChecker(str(self.project_path))
        results = checker.scan_project()

        self.assertEqual(results["files_scanned"], len(extensions))

    def test_scan_project_with_subdirectories(self):
        """Test que le scan est récursif dans les sous-répertoires"""
        (self.project_path / "src").mkdir()
        (self.project_path / "src" / "deep").mkdir()
        (self.project_path / "src" / "deep" / "model.py").write_text("import torch")

        checker = EUAIActChecker(str(self.project_path))
        results = checker.scan_project()

        self.assertIn("pytorch", results["detected_models"])

    def test_ai_disclosure_with_various_keywords(self):
        """Test de détection de tous les mots-clés AI dans le README"""
        keywords = ["ai", "artificial intelligence", "intelligence artificielle",
                     "machine learning", "deep learning", "gpt", "claude", "llm"]
        for keyword in keywords:
            (self.project_path / "README.md").write_text(f"# Project using {keyword}")
            checker = EUAIActChecker(str(self.project_path))
            self.assertTrue(
                checker._check_ai_disclosure(),
                f"Failed to detect AI disclosure for keyword: {keyword}"
            )

    def test_content_marking_french_marker(self):
        """Test de détection du marqueur français 'généré par ia'"""
        (self.project_path / "output.py").write_text("# Contenu généré par IA\ndef gen(): pass")

        checker = EUAIActChecker(str(self.project_path))
        self.assertTrue(checker._check_content_marking())

    def test_generate_recommendations_high_risk(self):
        """Test des recommandations spécifiques au risque élevé"""
        checker = EUAIActChecker(str(self.project_path))

        compliance_results = {
            "risk_category": "high",
            "compliance_status": {
                "documentation_technique": True,
                "gestion_risques": False,
            }
        }
        recommendations = checker._generate_recommendations(compliance_results)
        self.assertTrue(any("haut risque" in r.lower() for r in recommendations))
        self.assertTrue(any("❌" in r for r in recommendations))

    def test_high_risk_full_compliance(self):
        """Test risque élevé avec toute la documentation présente"""
        (self.project_path / "README.md").write_text("# Project")
        (self.project_path / "RISK_MANAGEMENT.md").write_text("# Risk")
        (self.project_path / "TRANSPARENCY.md").write_text("# Trans")
        (self.project_path / "DATA_GOVERNANCE.md").write_text("# Data")
        (self.project_path / "HUMAN_OVERSIGHT.md").write_text("# Human")
        (self.project_path / "ROBUSTNESS.md").write_text("# Robust")

        checker = EUAIActChecker(str(self.project_path))
        results = checker.check_compliance("high")

        self.assertTrue(all(results["compliance_status"].values()))
        self.assertEqual(results["compliance_percentage"], 100.0)
        self.assertEqual(results["compliance_score"], "6/6")


class TestMCPTools(unittest.TestCase):
    """Tests pour les outils MCP"""

    def setUp(self):
        """Créer un projet de test temporaire"""
        self.test_dir = tempfile.mkdtemp()
        self.project_path = Path(self.test_dir) / "test_project"
        self.project_path.mkdir()

    def tearDown(self):
        """Nettoyer le projet de test"""
        shutil.rmtree(self.test_dir)

    def test_scan_project_tool(self):
        """Test de l'outil scan_project"""
        (self.project_path / "test.py").write_text("import openai")

        result = scan_project_tool(str(self.project_path))

        self.assertEqual(result["tool"], "scan_project")
        self.assertIn("results", result)
        self.assertIn("files_scanned", result["results"])

    def test_check_compliance_tool(self):
        """Test de l'outil check_compliance"""
        (self.project_path / "README.md").write_text("# AI Project")

        result = check_compliance_tool(str(self.project_path), "limited")

        self.assertEqual(result["tool"], "check_compliance")
        self.assertIn("results", result)
        self.assertEqual(result["results"]["risk_category"], "limited")

    def test_check_compliance_tool_default_risk(self):
        """Test de l'outil check_compliance avec risque par défaut"""
        result = check_compliance_tool(str(self.project_path))

        self.assertEqual(result["tool"], "check_compliance")
        self.assertEqual(result["results"]["risk_category"], "limited")

    def test_generate_report_tool(self):
        """Test de l'outil generate_report"""
        (self.project_path / "main.py").write_text("from anthropic import Anthropic")
        (self.project_path / "README.md").write_text("# AI Project")

        result = generate_report_tool(str(self.project_path), "high")

        self.assertEqual(result["tool"], "generate_report")
        self.assertIn("results", result)
        self.assertIn("report_date", result["results"])
        self.assertEqual(result["results"]["compliance_summary"]["risk_category"], "high")


class TestMCPServer(unittest.TestCase):
    """Tests pour le serveur MCP"""

    def setUp(self):
        """Créer un serveur MCP de test"""
        self.server = MCPServer()
        self.test_dir = tempfile.mkdtemp()
        self.project_path = Path(self.test_dir) / "test_project"
        self.project_path.mkdir()

    def tearDown(self):
        """Nettoyer"""
        shutil.rmtree(self.test_dir)

    def test_init(self):
        """Test d'initialisation du serveur"""
        self.assertIn("scan_project", self.server.tools)
        self.assertIn("check_compliance", self.server.tools)
        self.assertIn("generate_report", self.server.tools)
        self.assertEqual(len(self.server.tools), 3)

    def test_list_tools(self):
        """Test de listage des outils"""
        result = self.server.list_tools()

        self.assertIn("tools", result)
        self.assertEqual(len(result["tools"]), 3)

        # Vérifier la structure de chaque outil
        for tool in result["tools"]:
            self.assertIn("name", tool)
            self.assertIn("description", tool)
            self.assertIn("parameters", tool)

    def test_handle_request_scan_project(self):
        """Test de requête scan_project"""
        (self.project_path / "test.py").write_text("import torch")

        result = self.server.handle_request("scan_project", {
            "project_path": str(self.project_path)
        })

        self.assertEqual(result["tool"], "scan_project")
        self.assertIn("pytorch", result["results"]["detected_models"])

    def test_handle_request_check_compliance(self):
        """Test de requête check_compliance"""
        (self.project_path / "README.md").write_text("# Test")

        result = self.server.handle_request("check_compliance", {
            "project_path": str(self.project_path),
            "risk_category": "minimal"
        })

        self.assertEqual(result["tool"], "check_compliance")
        self.assertEqual(result["results"]["risk_category"], "minimal")

    def test_handle_request_generate_report(self):
        """Test de requête generate_report"""
        result = self.server.handle_request("generate_report", {
            "project_path": str(self.project_path),
            "risk_category": "limited"
        })

        self.assertEqual(result["tool"], "generate_report")
        self.assertIn("report_date", result["results"])

    def test_handle_request_unknown_tool(self):
        """Test de requête avec outil inconnu"""
        result = self.server.handle_request("unknown_tool", {})

        self.assertIn("error", result)
        self.assertIn("Unknown tool", result["error"])
        self.assertIn("available_tools", result)

    def test_handle_request_with_exception(self):
        """Test de requête causant une exception"""
        result = self.server.handle_request("scan_project", {
            "invalid_param": "value"
        })

        self.assertIn("error", result)
        self.assertIn("Error executing", result["error"])


class TestConstants(unittest.TestCase):
    """Tests pour les constantes"""

    def test_ai_model_patterns(self):
        """Test des patterns de détection AI"""
        self.assertIn("openai", AI_MODEL_PATTERNS)
        self.assertIn("anthropic", AI_MODEL_PATTERNS)
        self.assertIn("huggingface", AI_MODEL_PATTERNS)
        self.assertIn("tensorflow", AI_MODEL_PATTERNS)
        self.assertIn("pytorch", AI_MODEL_PATTERNS)
        self.assertIn("langchain", AI_MODEL_PATTERNS)

        # Vérifier que chaque framework a des patterns
        for framework, patterns in AI_MODEL_PATTERNS.items():
            self.assertGreater(len(patterns), 0)
            self.assertIsInstance(patterns, list)

    def test_risk_categories(self):
        """Test des catégories de risque"""
        self.assertIn("unacceptable", RISK_CATEGORIES)
        self.assertIn("high", RISK_CATEGORIES)
        self.assertIn("limited", RISK_CATEGORIES)
        self.assertIn("minimal", RISK_CATEGORIES)

        # Vérifier la structure de chaque catégorie
        for category, info in RISK_CATEGORIES.items():
            self.assertIn("description", info)
            self.assertIn("requirements", info)
            self.assertIsInstance(info["requirements"], list)
            self.assertGreater(len(info["requirements"]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
