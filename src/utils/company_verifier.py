import re
import requests

def extract_company_name(text):
    # Méthode 1: Recherche basée sur les formes juridiques
    forms = [
        "SARL", "SAS", "EURL", "SA", "SASU", "SCI", "SNC", "SCA", "SCS", "SCP", "SELARL", "SELAS", "Société", "Entreprise"
    ]
    pattern = r"([A-Z][A-Za-z0-9\-\' ]+?)\s+(" + "|".join(forms) + r")\b"
    match = re.search(pattern, text)
    if match:
        return match.group(0)
    
    # Méthode 2: Recherche de mots consécutifs en majuscules comme nom d'entreprise potentiel
    # Cela détecte des motifs comme "GEMYA AUTOMOBILES LAVAL"
    pattern_all_caps = r"([A-Z]{2,}(?:\s+[A-Z]{2,}){1,3})"
    match_all_caps = re.findall(pattern_all_caps, text)
    if match_all_caps:
        # Filtrer les candidats trop courts ou qui semblent être des titres/sections
        valid_candidates = [candidate for candidate in match_all_caps 
                           if len(candidate) > 10 and 
                           not any(common in candidate for common in ["TITRE", "SECTION", "TEXTE", "PRINCIPAL", "CARACTÈRES"])]
        if valid_candidates:
            return max(valid_candidates, key=len)  # Retourne le plus long candidat
    
    return None

def verify_company_name(name):
    url = f"https://entreprise.data.gouv.fr/api/sirene/v3/unites_legales/?denomination={name}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("unites_legales"):
                return True, data["unites_legales"][0]
            # Si aucun résultat exact, proposer des suggestions proches
            suggestions = []
            for unite in data.get("unites_legales", []):
                denomination = unite.get("denomination", "")
                if denomination:
                    suggestions.append(denomination)
            return False, {"suggestions": suggestions}
    except Exception as e:
        return False, {"error": str(e)}
    return False, None

def check_company_in_text(text):
    name = extract_company_name(text)
    if not name:
        return {"detected": False, "message": "Aucun nom d'entreprise détecté"}
    found, details = verify_company_name(name)
    if found:
        return {"detected": True, "valid": True, "name": name, "details": details}
    else:
        # Ajout d'une alerte explicite et suggestions si disponibles
        message = f"Nom d'entreprise non trouvé dans la base officielle pour '{name}'. Vérifiez l'orthographe ou la présence d'une erreur d'OCR."
        suggestions = details.get("suggestions") if details else None
        if suggestions:
            message += f" Suggestions de noms proches : {', '.join(suggestions[:3])}"
        return {"detected": True, "valid": False, "name": name, "message": message, "suggestions": suggestions} 