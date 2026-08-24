"""
Test des règles métier véhicules sur des trajectoires simulées.

On n'a pas besoin d'un vrai flux caméra pour vérifier que le comptage
directionnel et l'intrusion de zone sont corrects : on injecte des
trajectoires connues et on vérifie les événements produits. C'est ce qui
permet de garantir que le compteur du portail n'affichera pas 40 entrées
pour un seul camion.

  python scripts/test_regles_vehicules.py
"""
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
from src.utils.common import charger_config, configurer_console  # noqa: E402
from src.detect.vehicules import DetecteurVehicules, SuiviVehicule  # noqa: E402

configurer_console()
LARGEUR, HAUTEUR = 1280, 720


class DetecteurSansModele(DetecteurVehicules):
    """Sous-classe qui court-circuite YOLO : on ne teste que la logique."""

    def __init__(self, config, fps=25.0):
        self.cfg = config
        self.inf = config.get("inference", {})
        self.lignes = config.get("lignes_comptage", [])
        self.zones = config.get("zones", [])
        self.stationnement = config.get("stationnement", {})
        self.fps = fps
        self.suivis = {}
        from collections import defaultdict
        self.compteurs = defaultdict(lambda: {"entree": 0, "sortie": 0})
        self._horloge = 0.0

    def injecter(self, identifiant, classe, centre):
        """Simule une observation d'un véhicule à une position donnée."""
        self._horloge += 1.0 / self.fps
        suivi = self.suivis.get(identifiant)
        if suivi is None:
            suivi = SuiviVehicule(identifiant, classe, self.fps)
            suivi.premiere_vue = self._horloge
            self.suivis[identifiant] = suivi
        suivi.derniere_vue = self._horloge
        suivi.trajectoire.append(centre)
        evenements = self._regle_lignes(suivi, centre, LARGEUR, HAUTEUR)
        evenements += self._regle_zones(suivi, centre, LARGEUR, HAUTEUR, classe)
        return evenements


def executer():
    config = charger_config("configs/vehicules.yaml")
    echecs = []

    # --- Test 1 : un camion traverse la ligne du portail vers le bas ---
    d = DetecteurSansModele(config)
    franchissements = []
    for y in range(300, 600, 10):                      # ligne a y = 0.62 * 720 = 446
        franchissements += [e for e in d.injecter(1, "camion_benne", (640, y))
                            if e["type"] == "franchissement"]
    total = d.compteurs["portail_principal"]
    print(f"Test 1 - traversee simple : {len(franchissements)} evenement(s), "
          f"compteur = {total}")
    if len(franchissements) != 1:
        echecs.append(f"Test 1 : {len(franchissements)} franchissements au lieu de 1")
    if total["entree"] + total["sortie"] != 1:
        echecs.append(f"Test 1 : compteur incoherent {total}")

    # --- Test 2 : aller-retour = une entree et une sortie, pas quatre ---
    d = DetecteurSansModele(config)
    for y in list(range(300, 600, 10)) + list(range(600, 300, -10)):
        d.injecter(2, "voiture", (640, y))
    total = d.compteurs["portail_principal"]
    print(f"Test 2 - aller-retour     : compteur = {total}")
    if total["entree"] != 1 or total["sortie"] != 1:
        echecs.append(f"Test 2 : attendu 1 entree et 1 sortie, obtenu {total}")

    # --- Test 3 : vehicule longeant la ligne sans la franchir ---
    d = DetecteurSansModele(config)
    for x in range(200, 1000, 10):
        d.injecter(3, "voiture", (x, 300))             # reste au-dessus de la ligne
    total = d.compteurs["portail_principal"]
    print(f"Test 3 - longe sans passer: compteur = {total}")
    if total["entree"] + total["sortie"] != 0:
        echecs.append(f"Test 3 : franchissement fantome {total}")

    # --- Test 4 : chargeuse entrant en zone pietonne (classe interdite) ---
    d = DetecteurSansModele(config)
    intrusions = []
    for _ in range(100):                               # ~4 s a 25 fps
        intrusions += [e for e in d.injecter(4, "chargeuse", (200, 600))
                       if e["type"] == "intrusion_zone"]
    print(f"Test 4 - intrusion zone   : {len(intrusions)} evenement(s) "
          f"(alarme apres {config['zones'][0]['alarme_apres_s']} s)")
    if not intrusions:
        echecs.append("Test 4 : aucune intrusion detectee en zone pietonne")

    # --- Test 5 : voiture dans la meme zone (classe autorisee) ---
    d = DetecteurSansModele(config)
    intrusions = []
    for _ in range(100):
        intrusions += [e for e in d.injecter(5, "voiture", (200, 600))
                       if e["type"] == "intrusion_zone"]
    print(f"Test 5 - classe autorisee : {len(intrusions)} evenement(s)")
    if intrusions:
        echecs.append("Test 5 : fausse intrusion pour une classe autorisee")

    print()
    if echecs:
        print(f"{len(echecs)} ECHEC(S) :")
        for e in echecs:
            print(f"  - {e}")
        return 1
    print("Toutes les regles metier sont conformes.")
    return 0


if __name__ == "__main__":
    sys.exit(executer())
