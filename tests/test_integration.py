#!/usr/bin/env python3
"""
Tests d'intégration pour le serveur MCP EU AI Act
Teste des scénarios utilisateur complets end-to-end
"""

import unittest
import sys
import tempfile
import shutil
import json
from pathlib import Path

# Ajouter le répertoire parent au path pour l'import
sys.path.insert(0, str(Path(__file__).parent.parent))

from server import MCPServer, EUAIActChecker


class TestEndToEndScenarios(unittest.TestCase):
    """Tests de scénarios utilisateur complets"""

    def setUp(self):
        """Créer un environnement de test"""
        self.server = MCPServer()
        self.test_dir = tempfile.mkdtemp()
        self.project_path = Path(self.test_dir) / "test_project"
        self.project_path.mkdir()

    def tearDown(self):
        """Nettoyer"""
        shutil.rmtree(self.test_dir)

    def test_scenario_simple_chatbot(self):
        """
        Scénario: Chatbot simple utilisant OpenAI
        - Risque limité
        - Doit avoir transparence et information utilisateurs
        """
        # Créer un chatbot simple
        (self.project_path / "chatbot.py").write_text("""
import openai

def chat(message):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": message}]
    )
    return response.choices[0].message.content
""")

        (self.project_path / "README.md").write_text("""
# Simple Chatbot

This chatbot uses OpenAI GPT-4 to answer questions.
Users interact with an AI system.
""")

        # 1. Scanner le projet
        scan_result = self.server.handle_request("scan_project", {
            "project_path": str(self.project_path)
        })

        self.assertEqual(scan_result["tool"], "scan_project")
        self.assertIn("openai", scan_result["results"]["detected_models"])
        self.assertEqual(scan_result["results"]["files_scanned"], 1)

        # 2. Vérifier la conformité (risque limité)
        compliance_result = self.server.handle_request("check_compliance", {
            "project_path": str(self.project_path),
            "risk_category": "limited"
        })

        self.assertEqual(compliance_result["results"]["risk_category"], "limited")
        # Devrait avoir transparence et information utilisateurs
        self.assertTrue(compliance_result["results"]["compliance_status"]["transparence"])
        self.assertTrue(compliance_result["results"]["compliance_status"]["information_utilisateurs"])

        # 3. Générer rapport complet
        report_result = self.server.handle_request("generate_report", {
            "project_path": str(self.project_path),
            "risk_category": "limited"
        })

        self.assertIn("report_date", report_result["results"])
        self.assertEqual(report_result["results"]["scan_summary"]["frameworks_detected"], ["openai"])
        self.assertGreater(len(report_result["results"]["recommendations"]), 0)

    def test_scenario_high_risk_recruitment_ai(self):
        """
        Scénario: Système de recrutement utilisant AI (risque élevé)
        - Doit avoir documentation complète, gestion des risques, etc.
        """
        # Créer un système de recrutement
        (self.project_path / "recruitment.py").write_text("""
from anthropic import Anthropic

def analyze_cv(cv_text):
    client = Anthropic()
    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"Analyze this CV and score the candidate: {cv_text}"
        }]
    )
    return message.content
""")

        (self.project_path / "README.md").write_text("""
# AI Recruitment System

Uses Claude AI to analyze CVs and score candidates.
""")

        # Scanner et vérifier
        report = self.server.handle_request("generate_report", {
            "project_path": str(self.project_path),
            "risk_category": "high"
        })

        # Devrait détecter Anthropic
        self.assertIn("anthropic", report["results"]["scan_summary"]["frameworks_detected"])

        # Devrait avoir score de conformité bas (documentation manquante)
        compliance_pct = report["results"]["compliance_summary"]["compliance_percentage"]
        self.assertLess(compliance_pct, 50)  # Probablement < 50%

        # Devrait avoir recommandations pour documentation
        recommendations = report["results"]["recommendations"]
        self.assertTrue(any("documentation" in r.lower() for r in recommendations))
        self.assertTrue(any("⚠️" in r or "haut risque" in r.lower() for r in recommendations))

    def test_scenario_multi_framework_project(self):
        """
        Scénario: Projet utilisant plusieurs frameworks AI
        - OpenAI, Anthropic, LangChain
        - Tous doivent être détectés
        """
        # Créer plusieurs fichiers avec différents frameworks
        (self.project_path / "openai_service.py").write_text("""
import openai
openai.ChatCompletion.create(model="gpt-4", messages=[])
""")

        (self.project_path / "anthropic_service.py").write_text("""
from anthropic import Anthropic
client = Anthropic()
""")

        (self.project_path / "langchain_agent.py").write_text("""
from langchain import LLMChain
from langchain.llms import ChatOpenAI
""")

        (self.project_path / "utils.py").write_text("""
def helper():
    pass
""")

        # Scanner
        scan_result = self.server.handle_request("scan_project", {
            "project_path": str(self.project_path)
        })

        detected = scan_result["results"]["detected_models"]
        self.assertIn("openai", detected)
        self.assertIn("anthropic", detected)
        self.assertIn("langchain", detected)
        self.assertEqual(scan_result["results"]["files_scanned"], 4)
        self.assertEqual(len(scan_result["results"]["ai_files"]), 3)

    def test_scenario_compliant_limited_risk_project(self):
        """
        Scénario: Projet à risque limité entièrement conforme
        - Transparence: README avec mention AI
        - Information utilisateurs: Divulgation claire
        - Marquage contenu: Code généré marqué
        """
        # Créer un projet conforme
        (self.project_path / "app.py").write_text("""
import openai

# This code is generated by AI
def generate_text(prompt):
    return openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt
    )
""")

        (self.project_path / "README.md").write_text("""
# AI Content Generator

This application uses OpenAI GPT models to generate content.

## AI Disclosure
Users are informed that they are interacting with an AI system.
All generated content is clearly marked as AI-generated.
""")

        # Vérifier conformité
        compliance_result = self.server.handle_request("check_compliance", {
            "project_path": str(self.project_path),
            "risk_category": "limited"
        })

        status = compliance_result["results"]["compliance_status"]
        self.assertTrue(status["transparence"])
        self.assertTrue(status["information_utilisateurs"])
        self.assertTrue(status["marquage_contenu"])

        # Score devrait être 3/3 = 100%
        self.assertEqual(compliance_result["results"]["compliance_score"], "3/3")
        self.assertEqual(compliance_result["results"]["compliance_percentage"], 100.0)

    def test_scenario_minimal_risk_game(self):
        """
        Scénario: Jeu vidéo avec AI (risque minimal)
        - Presque aucune exigence
        """
        (self.project_path / "game_ai.py").write_text("""
import torch
import torch.nn as nn

class GameAI(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
""")

        (self.project_path / "README.md").write_text("# Game AI")

        # Vérifier conformité minimale
        compliance_result = self.server.handle_request("check_compliance", {
            "project_path": str(self.project_path),
            "risk_category": "minimal"
        })

        self.assertEqual(compliance_result["results"]["risk_category"], "minimal")
        # Devrait avoir documentation basique (README existe)
        self.assertTrue(compliance_result["results"]["compliance_status"]["documentation_basique"])
        self.assertEqual(compliance_result["results"]["compliance_percentage"], 100.0)

    def test_scenario_no_ai_detected(self):
        """
        Scénario: Projet sans AI détecté
        - Devrait quand même vérifier la conformité
        """
        (self.project_path / "utils.py").write_text("""
def add(a, b):
    return a + b
""")

        (self.project_path / "README.md").write_text("# Utilities")

        # Scanner
        scan_result = self.server.handle_request("scan_project", {
            "project_path": str(self.project_path)
        })

        self.assertEqual(len(scan_result["results"]["detected_models"]), 0)
        self.assertEqual(len(scan_result["results"]["ai_files"]), 0)

        # Vérifier conformité quand même
        compliance_result = self.server.handle_request("check_compliance", {
            "project_path": str(self.project_path),
            "risk_category": "minimal"
        })

        # Devrait fonctionner même sans AI détecté
        self.assertEqual(compliance_result["results"]["risk_category"], "minimal")

    def test_scenario_nested_project_structure(self):
        """
        Scénario: Projet avec structure de dossiers imbriqués
        - AI code dans sous-dossiers
        - Documentation dans docs/
        """
        # Créer structure
        (self.project_path / "src").mkdir()
        (self.project_path / "src" / "ai").mkdir()
        (self.project_path / "docs").mkdir()

        (self.project_path / "src" / "ai" / "model.py").write_text("""
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
""")

        (self.project_path / "docs" / "TRANSPARENCY.md").write_text("""
# Transparency Documentation
This system uses AI models from HuggingFace.
""")

        # Scanner
        scan_result = self.server.handle_request("scan_project", {
            "project_path": str(self.project_path)
        })

        self.assertIn("huggingface", scan_result["results"]["detected_models"])
        self.assertIn("src/ai/model.py", scan_result["results"]["detected_models"]["huggingface"][0])

        # Vérifier conformité
        compliance_result = self.server.handle_request("check_compliance", {
            "project_path": str(self.project_path),
            "risk_category": "limited"
        })

        # Devrait détecter TRANSPARENCY.md dans docs/
        self.assertTrue(compliance_result["results"]["compliance_status"]["transparence"])

    def test_scenario_full_workflow(self):
        """
        Scénario: Workflow complet d'audit
        1. Liste les outils disponibles
        2. Scan du projet
        3. Vérification conformité pour chaque catégorie de risque
        4. Génération de rapport final
        """
        # Créer un projet
        (self.project_path / "ai_app.py").write_text("""
import openai

def process():
    return openai.ChatCompletion.create(model="gpt-4", messages=[])
""")

        (self.project_path / "README.md").write_text("# AI Application using GPT-4")

        # 1. Lister les outils
        tools = self.server.list_tools()
        self.assertEqual(len(tools["tools"]), 3)

        # 2. Scanner
        scan = self.server.handle_request("scan_project", {
            "project_path": str(self.project_path)
        })
        self.assertIn("openai", scan["results"]["detected_models"])

        # 3. Vérifier pour chaque catégorie
        for risk_category in ["minimal", "limited", "high"]:
            compliance = self.server.handle_request("check_compliance", {
                "project_path": str(self.project_path),
                "risk_category": risk_category
            })
            self.assertEqual(compliance["results"]["risk_category"], risk_category)
            self.assertIn("compliance_percentage", compliance["results"])

        # 4. Rapport final
        report = self.server.handle_request("generate_report", {
            "project_path": str(self.project_path),
            "risk_category": "limited"
        })

        # Vérifier structure complète du rapport
        self.assertIn("report_date", report["results"])
        self.assertIn("scan_summary", report["results"])
        self.assertIn("compliance_summary", report["results"])
        self.assertIn("detailed_findings", report["results"])
        self.assertIn("recommendations", report["results"])


    def test_scenario_unacceptable_risk_system(self):
        """
        Scénario: Système de notation sociale (risque inacceptable)
        - Devrait être interdit
        """
        (self.project_path / "social_scoring.py").write_text("""
from anthropic import Anthropic

def score_citizen(data):
    client = Anthropic()
    return client.messages.create(
        model="claude-3-opus-20240229",
        messages=[{"role": "user", "content": f"Score citizen: {data}"}]
    )
""")

        # Générer rapport pour risque inacceptable
        report = self.server.handle_request("generate_report", {
            "project_path": str(self.project_path),
            "risk_category": "unacceptable"
        })

        self.assertIn("anthropic", report["results"]["scan_summary"]["frameworks_detected"])
        # Pas de checks pour inacceptable (interdit)
        self.assertEqual(report["results"]["compliance_summary"]["compliance_score"], "0/0")

    def test_scenario_progressive_compliance(self):
        """
        Scénario: Amélioration progressive de la conformité
        - D'abord non-conforme, puis ajout documentation
        """
        # 1. Projet non-conforme
        (self.project_path / "app.py").write_text("import openai")

        result1 = self.server.handle_request("check_compliance", {
            "project_path": str(self.project_path),
            "risk_category": "limited"
        })
        pct1 = result1["results"]["compliance_percentage"]

        # 2. Ajouter README avec mention AI
        (self.project_path / "README.md").write_text("# AI-powered app using GPT")

        result2 = self.server.handle_request("check_compliance", {
            "project_path": str(self.project_path),
            "risk_category": "limited"
        })
        pct2 = result2["results"]["compliance_percentage"]

        # 3. Ajouter content marking
        (self.project_path / "app.py").write_text("""
import openai
# This content is generated by AI
def predict(): pass
""")

        result3 = self.server.handle_request("check_compliance", {
            "project_path": str(self.project_path),
            "risk_category": "limited"
        })
        pct3 = result3["results"]["compliance_percentage"]

        # Compliance should improve progressively
        self.assertGreater(pct2, pct1)
        self.assertGreater(pct3, pct2)
        self.assertEqual(pct3, 100.0)

    def test_scenario_real_world_project_scan(self):
        """
        Scénario: Scan du serveur MCP EU AI Act lui-même
        - Devrait détecter du code Python
        - Devrait être un projet réel
        """
        mcp_path = str(Path(__file__).parent.parent)
        scan_result = self.server.handle_request("scan_project", {
            "project_path": mcp_path
        })

        self.assertGreater(scan_result["results"]["files_scanned"], 0)


class TestErrorHandling(unittest.TestCase):
    """Tests de gestion des erreurs en conditions réelles"""

    def setUp(self):
        """Créer un environnement de test"""
        self.server = MCPServer()
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Nettoyer"""
        shutil.rmtree(self.test_dir)

    def test_invalid_project_path(self):
        """Test avec chemin de projet invalide"""
        result = self.server.handle_request("scan_project", {
            "project_path": "/this/path/does/not/exist"
        })

        self.assertIn("error", result["results"])

    def test_missing_parameters(self):
        """Test avec paramètres manquants"""
        result = self.server.handle_request("scan_project", {})

        self.assertIn("error", result)

    def test_invalid_risk_category(self):
        """Test avec catégorie de risque invalide"""
        project_path = Path(self.test_dir) / "test"
        project_path.mkdir()

        result = self.server.handle_request("check_compliance", {
            "project_path": str(project_path),
            "risk_category": "super_high"
        })

        self.assertIn("error", result["results"])

    def test_empty_params_dict(self):
        """Test avec dictionnaire de paramètres vide pour check_compliance"""
        result = self.server.handle_request("check_compliance", {})
        self.assertIn("error", result)

    def test_generate_report_nonexistent_path(self):
        """Test génération rapport avec chemin inexistant"""
        result = self.server.handle_request("generate_report", {
            "project_path": "/nonexistent/path",
            "risk_category": "limited"
        })
        # Should still return a report (with error in scan)
        self.assertEqual(result["tool"], "generate_report")


class TestReportGeneration(unittest.TestCase):
    """Tests de génération de rapports détaillés"""

    def setUp(self):
        """Créer un environnement de test"""
        self.test_dir = tempfile.mkdtemp()
        self.project_path = Path(self.test_dir) / "test_project"
        self.project_path.mkdir()

    def tearDown(self):
        """Nettoyer"""
        shutil.rmtree(self.test_dir)

    def test_report_structure_completeness(self):
        """Test de la structure complète du rapport"""
        (self.project_path / "main.py").write_text("import openai")
        (self.project_path / "README.md").write_text("# Test AI")

        checker = EUAIActChecker(str(self.project_path))
        scan_results = checker.scan_project()
        compliance_results = checker.check_compliance("limited")
        report = checker.generate_report(scan_results, compliance_results)

        # Vérifier tous les champs requis
        required_fields = [
            "report_date",
            "project_path",
            "scan_summary",
            "compliance_summary",
            "detailed_findings",
            "recommendations"
        ]

        for field in required_fields:
            self.assertIn(field, report)

        # Vérifier scan_summary
        self.assertIn("files_scanned", report["scan_summary"])
        self.assertIn("ai_files_detected", report["scan_summary"])
        self.assertIn("frameworks_detected", report["scan_summary"])

        # Vérifier compliance_summary
        self.assertIn("risk_category", report["compliance_summary"])
        self.assertIn("compliance_score", report["compliance_summary"])
        self.assertIn("compliance_percentage", report["compliance_summary"])

        # Vérifier detailed_findings
        self.assertIn("detected_models", report["detailed_findings"])
        self.assertIn("compliance_checks", report["detailed_findings"])
        self.assertIn("requirements", report["detailed_findings"])

        # Vérifier recommendations
        self.assertIsInstance(report["recommendations"], list)
        self.assertGreater(len(report["recommendations"]), 0)

    def test_report_json_serializable(self):
        """Test que le rapport est serializable en JSON"""
        (self.project_path / "test.py").write_text("from anthropic import Anthropic")

        checker = EUAIActChecker(str(self.project_path))
        scan_results = checker.scan_project()
        compliance_results = checker.check_compliance("high")
        report = checker.generate_report(scan_results, compliance_results)

        # Ne devrait pas lever d'exception
        json_str = json.dumps(report, indent=2)
        self.assertIsInstance(json_str, str)

        # Devrait être déserializable
        parsed = json.loads(json_str)
        self.assertEqual(parsed["project_path"], str(self.project_path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
