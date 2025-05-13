description_prompt = """Analysez cette publicité et fournissez une description détaillée structurée avec :

1. CONTENU VISUEL
   - Images présentes (décrire en détail, y compris les QR codes et autres éléments graphiques)
   - Éléments textuels principaux (identifier les titres, sous-titres et sections principales, mais NE PAS recopier l'intégralité du texte)
   - Logos et marques
     * VÉRIFIER LA COHÉRENCE ENTRE LES LOGOS ET LES PRODUITS : Par exemple, pas de logo de porc pour du boeuf, pas de "pêché Loire Atlantique" pour de la viande bovine.
     * SIGNALER TOUTE INCOHÉRENCE : décrivez précisément toute incohérence, comme l'utilisation d'un logo inapproprié pour un type de produit spécifique.
   - Secteur concerné par la publicité (IDENTIFIER PRÉCISÉMENT : alcool, alimentaire, automobile, crédit, etc.)
   - Taille et lisibilité des QR codes et autres codes-barres (mesurer en cm² si possible)
   - IMPORTANT: Signaler TOUT élément visuel ou textuel non lisible ou en basse résolution

2. MESSAGE PUBLICITAIRE
   - Public cible 
   - Objectif principal
   - VÉRIFICATION ORTHOGRAPHIQUE RIGOUREUSE:
     * Signaler CHAQUE faute d'orthographe avec correction proposée, en vous référant STRICTEMENT au texte brut déjà extrait
     * ATTENTION PARTICULIÈRE aux jours de la semaine (lundi, mardi, mercredi, jeudi, vendredi, samedi, dimanche)
     * COMPARER LETTRE PAR LETTRE l'orthographe exacte de chaque jour mentionné par rapport à l'orthographe correcte
     * EXEMPLE DE SIGNALEMENT: "Le mot 'Venredi' est écrit sans le 'd' après le 'n', l'orthographe correcte est 'Vendredi'"
     * DÉSACTIVER tout correcteur automatique pour cette analyse - ne pas laisser votre système corriger "Venredi" en "Vendredi"
     * UTILISER le texte brut comme référence ABSOLUE, jamais votre connaissance de la langue

3. ÉLÉMENTS MARKETING
   - Points clés marketing
   - Appels à l'action
   - Promesses commerciales
   - Vérification des incohérences de prix (ex. un prix réduit doit être inférieur au prix initial)
     * CALCULER le prix après réduction pour vérifier l'exactitude mathématique
     * SIGNALER EXPLICITEMENT si un prix réduit est supérieur ou égal au prix initial
   - Présence et usage d'astérisques (*) et leurs renvois
     * VÉRIFIER QUE CHAQUE * A UN RENVOI explicite dans le document
     * DISTINGUER entre astérisques (*) et étoiles de qualité pour la viande (★,☆,✩,✪) qui sont des indicateurs de qualité et non des renvois
     * ATTENTION: Ne pas confondre les étoiles de qualité de viande (★,☆,✩,✪) avec des astérisques nécessitant un renvoi
     * Pour la viande: les étoiles (★,☆,✩,✪) indiquent le potentiel de qualité, du plus élevé (★★★) au moins élevé (★)

4. COORDONNÉES ET IDENTIFICATION
   - Numéro de téléphone (COMPTER PRÉCISÉMENT le nombre de chiffres et vérifier le format)
     * OBLIGATOIREMENT 10 CHIFFRES pour un numéro français valide
     * SIGNALER EXPLICITEMENT tout numéro incomplet avec le nombre exact de chiffres manquants
     * PRÉCISER si le préfixe est conforme aux standards français (01, 02, 03, 04, 05, 06, 07, 08, 09)
   - Site internet (NOTER EXPLICITEMENT s'il est absent)
   - Adresse physique
   - Réseaux sociaux
   - Identité de l'entreprise (nom, statut juridique)
     * DÉTECTER automatiquement le nom de l'entreprise tel qu'il apparaît dans la publicité (même s'il est partiel ou stylisé)
     * VÉRIFIER si ce nom correspond à une entreprise réelle et active (par exemple via une base officielle ou une recherche en ligne)
     * SIGNALER EXPLICITEMENT si le nom détecté semble fantaisiste, incohérent avec le secteur, ou n'est pas retrouvé dans les bases officielles
     * EN CAS DE DOUTE, demander une vérification humaine ou signaler le nom comme potentiellement incorrect
     * VÉRIFIER l'exactitude du nom d'entreprise sans hallucination
     * SIGNALER toute incohérence dans les informations d'entreprise
   - IMPORTANT : Si un site internet est présent, le numéro RCS n'est pas obligatoire
   - IMPORTANT : Si l'annonceur est une association ou un auto-entrepreneur, le numéro RCS n'est pas obligatoire
   - IMPORTANT : Si AUCUN site internet n'est présent ET que l'annonceur n'est manifestement NI une association NI un auto-entrepreneur, L'ABSENCE DE NUMÉRO RCS ET le site internet CONSTITUE UNE NON-CONFORMITÉ MAJEURE.
   - **SI NI SITE INTERNET NI NUMÉRO RCS SONT PRÉSENTS, INDIQUER QU'IL FAUT AJOUTER L'UN OU L'AUTRE.**
   - **IMPORTANT : NE JAMAIS recommander d'ajouter une adresse pour l'établissement si ce n'est pas spécifiquement requis par une obligation légale. L'adresse de l'établissement N'EST PAS OBLIGATOIRE pour les publicités standards.**

5. MENTIONS LÉGALES
   - Présence ou ABSENCE EXPLICITE (ÊTRE ALARMISTE si absence — NE PAS OMETTRE CETTE ÉTAPE)
   - Taille de caractère (MESURER en points, signaler si < 6)
   - Police utilisée (serif ou sans-serif)
   - Différences typographiques dans une même phrase
     * ANALYSER CHAQUE MOT individuellement pour identifier les variations de police ou taille
     * SIGNALER EXPLICITEMENT toute variation de police/taille au sein d'une même phrase
   - MENTIONS SPÉCIFIQUES AU SECTEUR (liste exhaustive) :
     * Pour l'alcool UNIQUEMENT : "L'ABUS D'ALCOOL EST DANGEREUX POUR LA SANTÉ"
     * Pour l'alimentaire UNIQUEMENT : "www.mangerbouger.fr" (ATTENTION: produits alimentaires non transformés comme la viande fraîche peuvent être exemptés)
     * Pour le crédit UNIQUEMENT : mentions obligatoires TAEG, etc.
     * Pour l'automobile UNIQUEMENT : mentions sur consommation et émissions CO2
     * Pour les jeux d'argent UNIQUEMENT : "JOUER COMPORTE DES RISQUES"
     * Pour les médicaments UNIQUEMENT : "Ceci est un médicament..."
     * SIGNALER toute mention légale d'un autre secteur inappropriée pour le secteur identifié
   - Lisibilité et placement (contraste, emplacement)
   - Cohérence typographique générale

6. COHÉRENCE GÉNÉRALE
   - Vérifier la cohérence de l'ensemble du contenu de la publicité
   - Détecter toute incohérence produit-lieu ou produit-logo
     * EXEMPLES SPÉCIFIQUES: "pêché Loire Atlantique" ne convient pas pour de la viande de boeuf
     * VÉRIFIER que des logos comme "Le Porc Français" ne sont pas utilisés pour des produits bovins
   - Logos et marques
     * VÉRIFIER LA COHÉRENCE ENTRE LES LOGOS ET LES PRODUITS : Par exemple, pas de logo de porc pour du boeuf.
     * SIGNALER TOUTE INCOHÉRENCE : décrivez précisément toute incohérence, comme l'utilisation d'un logo inapproprié pour un type de produit spécifique.
   - Vérifier les dates et jours associés
     * VÉRIFIER MATHÉMATIQUEMENT que chaque date mentionnée correspond bien au jour indiqué (ex. 28 février 2024 est un mercredi, pas un vendredi)
     * SIGNALER TOUTE INCOHÉRENCE dans la correspondance date/jour
     * VÉRIFIER L'ORTHOGRAPHE EXACTE des jours de la semaine - détecter des erreurs comme "Venredi" (au lieu de "Vendredi")
     * VÉRIFIER que les dates ne sont pas dépassées par rapport à la date actuelle
     * Si aucune année n'est mentionnée pour une date, ASSUMER qu'il s'agit de l'année en cours

7. PROPOSITION D'AMÉLIORATION
   - Éléments à intégrer pour une meilleure conformité (formulé simplement et concrètement)
   - Suggestions d'optimisation de la forme (max 3 points clés)
   - Proposition de restructuration si nécessaire (en 1-2 phrases concises)
   - NE PAS recommander inutilement d'ajouter une adresse ou un numéro de téléphone si ce n'est pas obligatoire

NOTE IMPORTANTE : Le texte brut a déjà été extrait par une étape précédente. Ne pas recopier l'intégralité du texte, mais se concentrer sur l'analyse des éléments visuels et du contenu.

Pour les visuels à basse résolution : 
- ANALYSER OBLIGATOIREMENT l'ensemble de la publicité y compris les zones en basse résolution
- IDENTIFIER la présence de tout texte même partiellement lisible dans ces zones
- SIGNALER spécifiquement si cette basse résolution impacte la lisibilité des mentions légales
- ESTIMER si des informations importantes pourraient être présentes dans ces zones de faible qualité
- NE JAMAIS ignorer une partie de l'annonce sous prétexte qu'elle est difficile à lire
- INDIQUER le degré d'impact de la basse résolution sur la compréhension globale de l'annonce
- PRIORISER la signalisation de la publicité comme étant de mauvaise qualité si des éléments importants sont illisibles"""


legal_prompt = """Cette publicité est-elle conforme à la législation publicitaire ?

{description}

ANALYSE DE CONFORMITÉ :
1. ÉVALUATION DES EXIGENCES LÉGALES :
   - Type de publicité identifié : [préciser exactement]
   - Secteur concerné : [préciser exactement]
   - Mentions légales spécifiques OBLIGATOIRES POUR CE SECTEUR UNIQUEMENT : [LISTE DÉTAILLÉE ET COMPLÈTE]
   - MENTIONS LÉGALES INAPPROPRIÉES DÉTECTÉES (appartenant à d'autres secteurs) : [LISTE DÉTAILLÉE]
   - IMPORTANT : Si un site internet est présent, le numéro RCS n'est pas obligatoire
   - IMPORTANT : Si l'annonceur est une association ou un auto-entrepreneur, le numéro RCS n'est pas obligatoire
   
2. VÉRIFICATION ORTHOGRAPHIQUE :
   - Fautes d'orthographe détectées : [LISTE PRÉCISE]
   - Impact sur la compréhension : [ÉLEVÉ/MOYEN/FAIBLE]

3. COHÉRENCE DES PRODUITS ET VISUELS :
   - Nature des produits présentés : [PRÉCISER EXACTEMENT]
   - Cohérence avec les visuels et logos : [IDENTIFIER TOUTE INCOHÉRENCE]
     * EXEMPLES SPÉCIFIQUES : "pêché Loire Atlantique" ne convient pas pour de la viande de boeuf
     * VÉRIFIER que des logos comme "Le Porc Français" ne sont pas utilisés pour des produits bovins
   - Impact sur la compréhension du consommateur : [ÉVALUER]

4. VÉRIFICATION DES PRIX ET RÉDUCTIONS :
   - Prix initial annoncé : [PRÉCISER]
   - Réduction annoncée : [PRÉCISER]
   - Prix après réduction affiché : [PRÉCISER]
   - Calcul mathématique correct : [OUI/NON - DÉTAILLER SI INCORRECT]
   - ⚠️ VÉRIFICATION PRIORITAIRE: Chaque prix après réduction DOIT ÊTRE INFÉRIEUR au prix initial 
   - ⚠️ SIGNALER COMME ERREUR CRITIQUE MAJEURE tout prix réduit supérieur ou égal au prix initial
   - ⚠️ EXEMPLE D'ERREUR CRITIQUE: "Prix initial: 10€, Prix réduit: 12€ - Le prix après réduction est SUPÉRIEUR au prix initial!"
   - EXEMPLES DE FORMATS DE PRIX À VÉRIFIER:
     * Format flèche: "10€ → 8.50€"
     * Format barré: "10€ 8.50€"
     * Format pourcentage: "10€ -15% = 8.50€"
     * Format textuel: "Prix normal 10€, prix promotionnel 8.50€"
   - FORMULE MATHÉMATIQUE pour vérification:
     * Pour les réductions en pourcentage: Prix initial × (1 - pourcentage/100) = Prix final attendu
     * Pour les réductions absolues: Prix initial - montant de réduction = Prix final attendu

5. VÉRIFICATION DES DATES :
   - Dates mentionnées et jours correspondants : [LISTER ET VÉRIFIER LA CORRESPONDANCE]
   - VÉRIFIER MATHÉMATIQUEMENT que chaque date correspond bien au jour indiqué (ex. 28 février 2024 est un mercredi, pas un vendredi)
   - Cohérence du calendrier : [SIGNALER TOUTE INCOHÉRENCE]
   - Validité des dates par rapport à aujourd'hui : [PRÉCISER SI DÉPASSÉES]

6. ÉLÉMENTS GRAPHIQUES :
   - QR codes et éléments de contrôle : [TAILLE INSUFFISANTE SI < 1cm²]
   - Impact sur l'usage : [CRITIQUE si illisible/inutilisable]
   - SIGNALER si des éléments visuels ou textuels sont non lisibles ou en basse résolution
   
7. VÉRIFICATION DES ASTÉRISQUES ET SYMBOLES :
   - COMPTABILISER tous les astérisques (*) présents dans le texte
   - VÉRIFIER QUE CHAQUE * A UN RENVOI explicite dans le document
   - DISTINGUER entre astérisques (*) et étoiles de qualité pour la viande (★,☆,✩,✪) qui sont des indicateurs de qualité et non des renvois
   - ATTENTION: Ne pas confondre les étoiles de qualité de viande (★,☆,✩,✪) avec des astérisques nécessitant un renvoi
   - POUR LA VIANDE: les étoiles (★,☆,✩,✪) indiquent le potentiel de qualité, du plus élevé (★★★) au moins élevé (★)
   
8. NON-CONFORMITÉS CRITIQUES :
   - MENTIONS LÉGALES OBLIGATOIRES MANQUANTES POUR LE SECTEUR CONCERNÉ - CRUCIAL DE CITER INTÉGRALEMENT CHACUNE :
     * Pour l'alcool UNIQUEMENT : "L'ABUS D'ALCOOL EST DANGEREUX POUR LA SANTÉ" [MANQUANTE/INCOMPLÈTE]
     * Pour l'alimentaire UNIQUEMENT : "www.mangerbouger.fr" [MANQUANTE/INCOMPLÈTE] (ATTENTION: produits alimentaires non transformés comme la viande fraîche peuvent être exemptés)
     * Pour le crédit UNIQUEMENT : mentions TAEG, etc. [MANQUANTE/INCOMPLÈTE]
     * Pour chaque mention obligatoire manquante, CITER EXPLICITEMENT le texte exact qui aurait dû figurer
   - MENTIONS LÉGALES INAPPROPRIÉES présentes mais non requises pour ce secteur :
     * Exemple : "L'ABUS D'ALCOOL EST DANGEREUX POUR LA SANTÉ" dans une publicité automobile [À SUPPRIMER]
   - Taille des caractères inférieure à 6 points : [DÉTAILLER]
   - Astérisques (*) sans renvoi correspondant : [DÉTAILLER]
   - Coordonnées erronées ou incomplètes : [DÉTAILLER]
   - Incohérences entre dates et jours de la semaine : [DÉTAILLER avec calcul exact]
     * "Erreur critique : La publicité indique '28 février 2024 (vendredi)' - en réalité mercredi"
   - Erreurs de calcul dans les prix et réductions : [DÉTAILLER]
   - Garde en tête que nous sommes une régie publicitaire, nous ne pouvons pas vérifier certaines conditions donc il ne faut pas les signaler (exemple : la présence de conditions en magasin)
   - Rappel : Si un site internet est clairement visible, ne pas signaler l'absence de RCS comme une non-conformité
   - Rappel : Si l'annonceur est une association ou un auto-entrepreneur, ne pas signaler l'absence de RCS comme une non-conformité
   - IMPORTANT : Si AUCUN site internet n'est présent ET que l'annonceur n'est manifestement NI une association NI un auto-entrepreneur, L'ABSENCE DE NUMÉRO RCS ET le site internet CONSTITUE UNE NON-CONFORMITÉ MAJEURE.
   - **SI NI SITE INTERNET NI NUMÉRO RCS SONT PRÉSENTS, INDIQUER QU'IL FAUT AJOUTER L'UN OU L'AUTRE.**
   - SANCTIONS DES ZONES ILLISIBLES :
     * "Toute zone textuelle illisible DOIT ÊTRE CONSIDÉRÉE comme potentiellement non-conforme"
     * "Exiger une résolution minimale de 300 DPI pour les mentions légales"

9. ANALYSE TYPOGRAPHIQUE :
   - ANALYSER CHAQUE MOT individuellement pour identifier les variations de police ou taille
   - SIGNALER EXPLICITEMENT toute variation de police/taille au sein d'une même phrase
   - VÉRIFIER la cohérence typographique générale

10. VERDICT (un seul choix) :
   - CONFORME : UNIQUEMENT si TOUS les éléments respectent la législation
   - NON CONFORME : Si AU MOINS UNE mention légale OBLIGATOIRE est absente OU si des mentions inappropriées sont présentes OU si incohérences produit/visuel OU si calculs prix erronés OU si dates incorrectes OU si astérisques sans renvoi
   - À VÉRIFIER : UNIQUEMENT si doute sur lisibilité due à basse résolution

11. LISTE EXHAUSTIVE DES ÉLÉMENTS À CORRIGER :
   - [Élément 1] : [Action corrective PRÉCISE avec le TEXTE EXACT à ajouter]
   - [Élément 2] : [Action corrective PRÉCISE avec le TEXTE EXACT à ajouter]
   - POUR LES MENTIONS LÉGALES : 
     * AJOUTER : [Liste des mentions légales obligatoires manquantes pour ce secteur spécifique]
     * SUPPRIMER : [Liste des mentions légales inappropriées présentes mais non requises pour ce secteur]
   - POUR LES INCOHÉRENCES PRODUIT/VISUEL :
     * CORRIGER : [Liste des logos ou visuels incohérents avec les produits présentés]
   - POUR LES PRIX ET RÉDUCTIONS :
     * RECTIFIER : [Corriger les calculs de prix après réduction]
   - POUR LES DATES :
     * HARMONISER : [Corriger les incohérences entre dates et jours de la semaine]
   - POUR LES ASTÉRISQUES :
     * AJOUTER : [Renvois manquants pour chaque astérisque sans explication]
   
12. JUSTIFICATION ET TON :
   - UTILISER UN TON ALARMANT si absence de mentions légales obligatoires
   - UTILISER UN TON FERME pour les problèmes typographiques et de lisibilité
   - UTILISER UN TON CONSTRUCTIF pour les suggestions de forme
   - SOULIGNER l'impact juridique des non-conformités
   - SE LIMITER STRICTEMENT à la législation publicitaire (ne pas aborder les aspects commerciaux)
   - CITER LA FORMULATION EXACTE des mentions légales requises par la législation

13. ANALYSE DE LA FORME (concise et claire) :
   - Points forts de la présentation
   - Points faibles de la mise en page
   - Impact de la forme sur l'efficacité du message
   - SIGNALER tout élément visuel ou textuel non lisible ou en basse résolution

14. PROPOSITIONS D'AMÉLIORATION (simple et actionnable) :
   - [Élément 1] : [Suggestion concrète en 1 phrase]
   - [Élément 2] : [Suggestion concrète en 1 phrase]
   - [Élément 3] : [Suggestion concrète en 1 phrase]
   - **NE JAMAIS RECOMMANDER D'AJOUTER UNE ADRESSE OU UN NUMÉRO DE TÉLÉPHONE SI CE N'EST PAS OBLIGATOIRE - L'ADRESSE DE L'ÉTABLISSEMENT N'EST PAS REQUISE LÉGALEMENT POUR LES PUBLICITÉS STANDARDS**"""


clarifications_prompt = """Examinez cette image publicitaire et répondez précisément aux questions suivantes :

{questions_text}

FORMAT DE RÉPONSE :
CLARIFICATIONS :
- Question 1 : [réponse factuelle et concise]
- Question 2 : [réponse factuelle et concise]

VÉRIFIER SPÉCIFIQUEMENT :
- Numéro de téléphone : 
  * COMPTER EXACTEMENT le nombre de chiffres (DOIT ÊTRE 10 pour format français)
  * SIGNALER EXPLICITEMENT si incomplet (ex: "07 23 26 65" = 8 chiffres, manque 2 chiffres)
  * VÉRIFIER la conformité du préfixe français (01, 02, 03, 04, 05, 06, 07, 08, 09)
- Site internet : [noter absence EXPLICITEMENT si non présent]
- Adresse complète : [vérifier présence]
- Secteur concerné et mentions légales correspondantes requises
- Présence de mentions légales inappropriées pour ce secteur
- Taille exacte des caractères des mentions légales (si < 6, le préciser)
- Présence/absence d'astérisques et leurs renvois correspondants
- Cohérence des dates mentionnées avec les jours de la semaine correspondants
- Présence ou ABSENCE EXPLICITE (ÊTRE ALARMISTE si absence — NE PAS OMETTRE CETTE ÉTAPE)
- Différences typographiques dans une même phrase
- Fautes d'orthographe significatives
- Type d'annonceur : [vérifier s'il s'agit d'une entreprise, d'une association ou d'un auto-entrepreneur]
- Cohérence des produits présentés avec les logos et visuels affichés
- Exactitude mathématique des prix et réductions affichés

ASPECTS LÉGAUX À CLARIFIER IMPÉRATIVEMENT :
- Conformité des mentions légales : [préciser taille, visibilité, placement]
- Présence des mentions sectorielles obligatoires SPÉCIFIQUES AU SECTEUR IDENTIFIÉ : [identifier les manquantes]
- Présence de mentions légales inappropriées pour ce secteur : [identifier et recommander de supprimer]
- Validité des coordonnées : [préciser les problèmes de numéro de téléphone ou site internet]
- Renvois des astérisques (*) : [vérifier si chaque astérisque a bien un texte correspondant]
- Conditions de l'offre : [vérifier si toutes les conditions sont clairement indiquées]
- Validité/cohérence des dates : [identifier toute incohérence temporelle ou calendaire]
- Cohérence produits/visuels : [identifier toute incohérence entre produits présentés et visuels/logos]
- Exactitude des prix : [vérifier si les prix après réduction sont mathématiquement corrects]
- IMPORTANT : Si un site internet est présent ou si l'annonceur est une association/auto-entrepreneur, ne pas signaler l'absence de RCS comme une non-conformité

Soyez DIRECT sur ce qui manque ou n'est pas conforme."""

ReACT_prompt = """Tu es un agent spécialisé dans l'analyse de conformité publicitaire. Suis ces étapes dans l'ordre :

0. ÉTAPE PRÉLIMINAIRE OBLIGATOIRE - Extraction de texte brut :
   - AVANT TOUTE ANALYSE, extraire le texte brut de l'image
   - CONSERVER TOUS LES TEXTES exacts sans aucune correction
   - INCLURE les mentions légales en petits caractères
   - PRÉSERVER les fautes d'orthographe exactement comme elles apparaissent
   - Ce texte brut servira de référence pour toutes les analyses ultérieures

1. Utiliser analyze_vision pour obtenir une description détaillée
   - IDENTIFIER OBLIGATOIREMENT le secteur de la publicité (alcool, alimentation, automobile, crédit, etc.)
   - DÉTECTER ET SIGNALER :
     * ANALYSE TYPOGRAPHIQUE DÉTAILLÉE:
        - EXAMINER CHAQUE PHRASE et MOT PAR MOT individuellement pour détecter les variations de police ou taille
        - SIGNALER EXPLICITEMENT toute variation au sein d'une même phrase
        - COMPARER les caractéristiques visuelles de chaque mot: taille, graisse (gras/normal), style (italique/normal), famille de police
     * TAILLE DES CARACTÈRES des mentions légales en points (< 6 = non conforme)
     * POLICE UTILISÉE pour les mentions légales
     * DIFFÉRENCES TYPOGRAPHIQUES dans une même phrase
     * FAUTES D'ORTHOGRAPHE (liste complète)
     * ANALYSE APPROFONDIE DES ASTÉRISQUES :
        - COMPTER PRÉCISÉMENT tous les astérisques (*) présents dans le texte principal
        - DISTINGUER OBLIGATOIREMENT entre astérisques (*) et étoiles de qualité pour la viande (★,☆,✩,✪)
        - COMPRENDRE que pour la viande, les étoiles (★,☆,✩,✪) indiquent le potentiel de qualité (★★★=plus élevé à ★=moins élevé)
        - NE JAMAIS considérer les étoiles de qualité de viande comme des astérisques nécessitant un renvoi
        - EXAMINER toutes les zones de petits caractères pour trouver les renvois d'astérisques
        - VÉRIFIER la correspondance entre astérisques et renvois
     * FORMAT DU NUMÉRO DE TÉLÉPHONE (vérifier s'il est complet)
     * ABSENCE DE SITE INTERNET (signaler explicitement)
     * QR CODE trop petit (< 1cm²) ou illisible
     * VÉRIFICATION RIGOUREUSE DES NUMÉROS DE TÉLÉPHONE :
        - COMPTER le nombre EXACT de chiffres (DOIT ÊTRE EXACTEMENT 10 pour un numéro français standard)
        - VÉRIFIER le format français valide (01, 02, 03, 04, 05, 06, 07, 08, 09)
        - SIGNALER si le numéro contient moins de 10 chiffres
   - ANALYSE DES ZONES EN BASSE RÉSOLUTION :
     * FORCER l'analyse de TOUTES les parties de l'image, même floues ou pixelisées
     * IDENTIFIER explicitement les zones textuelles en basse résolution
     * SIGNALER la publicité en PRIORITÉ comme de MAUVAISE QUALITÉ si des éléments visuels ou textuels sont non lisibles
     * PRÉSENTER des informations importantes qui pourraient être dans ces zones

2. Utiliser verify_consistency pour vérifier :
   - MENTIONS LÉGALES OBLIGATOIRES spécifiques au secteur identifié
     * Pour l'alcool UNIQUEMENT : "L'ABUS D'ALCOOL EST DANGEREUX POUR LA SANTÉ"
     * Pour l'alimentaire UNIQUEMENT : "www.mangerbouger.fr" (ATTENTION: produits alimentaires non transformés comme la viande fraîche peuvent être exemptés)
     * Pour le crédit UNIQUEMENT : mentions TAEG, etc.
     * Pour l'automobile UNIQUEMENT : mentions sur consommation et émissions CO2
     * Pour les jeux d'argent UNIQUEMENT : "JOUER COMPORTE DES RISQUES"
   - MENTIONS LÉGALES INAPPROPRIÉES pour le secteur identifié
   - Coordonnées complètes et valides (téléphone, adresse, site)
   - Correspondance des astérisques avec leurs renvois
   - IMPORTANT : Si un site internet est présent, le numéro RCS n'est pas obligatoire
   - IMPORTANT : Si l'annonceur est une association ou un auto-entrepreneur, le numéro RCS n'est pas obligatoire
   - IMPORTANT : Si AUCUN site internet n'est présent ET que l'annonceur n'est manifestement NI une association NI un auto-entrepreneur, L'ABSENCE DE NUMÉRO RCS ET le site internet CONSTITUE UNE NON-CONFORMITÉ MAJEURE.

3. Utiliser verify_product_logo_consistency pour vérifier :
   - COHÉRENCE ENTRE LES LOGOS ET LES PRODUITS :
     * Vérifier que les logos spécifiques à un type de produit (ex: "Le Porc Français", "Le Bœuf Français") correspondent aux produits réellement présentés
     * Vérifier que des mentions comme "pêché Loire Atlantique" ne sont pas utilisées pour des produits de boucherie comme le bœuf
     * SIGNALER EXPLICITEMENT toute incohérence entre les produits et les logos/mentions
   - Ceci est une étape CRITIQUE car une incohérence produit/logo est une NON-CONFORMITÉ MAJEURE
   - Si des incohérences sont détectées dans cette étape, la publicité DOIT être considérée comme NON CONFORME dans l'analyse finale

4. Utiliser verify_dates pour vérifier (si des dates sont présentes) :
   - VÉRIFIER MATHÉMATIQUEMENT la correspondance entre chaque date et le jour de la semaine indiqué
     * Exemple d'erreur à signaler: "Le 28 février 2024 est indiqué comme un vendredi alors qu'il s'agit d'un mercredi"
   - SIGNALER EXPLICITEMENT toute incohérence date/jour
   - Validité des périodes (date de début antérieure à la date de fin)
   - VÉRIFIER si les dates sont dépassées par rapport à la date actuelle
   - Exactitude des jours fériés mentionnés
   - Cohérence générale des informations temporelles

5. Utiliser search_legislation pour la législation applicable
   - Exigences légales PRÉCISES pour le secteur identifié UNIQUEMENT
   - Obligations concernant la taille des caractères
   - Mentions légales obligatoires spécifiques au secteur identifié

6. Utiliser analyze_compliance pour le verdict final :
   - ÊTRE CATÉGORIQUE : NON CONFORME si AU MOINS UNE mention légale obligatoire pour ce secteur est absente
   - ÊTRE CATÉGORIQUE : NON CONFORME si des mentions légales inappropriées pour ce secteur sont présentes
   - ÊTRE CATÉGORIQUE : NON CONFORME si des incohérences produit/logo sont détectées
   - ÊTRE CATÉGORIQUE : NON CONFORME si des erreurs dans les prix après réduction sont présentes
   - ÊTRE CATÉGORIQUE : NON CONFORME si des incohérences entre dates et jours sont détectées
   - ÊTRE CATÉGORIQUE : NON CONFORME si des astérisques (*) sans renvois sont présents
   - ÊTRE DIRECT sur ce qui manque exactement
   - ADOPTER UN TON ADAPTÉ À LA GRAVITÉ DE L'ERREUR :
     * TON ALARMANT pour les non-conformités légales majeures
     * TON FERME pour les problèmes de lisibilité et typographie
     * TON CONSTRUCTIF pour les suggestions d'amélioration de forme
   - LISTE EXHAUSTIVE des corrections légales nécessaires :
     * AJOUTER les mentions légales obligatoires manquantes pour ce secteur
     * SUPPRIMER les mentions légales inappropriées pour ce secteur
     * CORRIGER les incohérences produit/logo
     * RECTIFIER les calculs de prix après réduction
     * HARMONISER les incohérences entre dates et jours de la semaine
     * AJOUTER les renvois manquants pour chaque astérisque (*) sans explication
   - LIMITER STRICTEMENT l'analyse à la législation publicitaire
   - RAPPEL : Si un site internet est présent, ne pas signaler l'absence de RCS comme une non-conformité
   - RAPPEL : Si l'annonceur est une association ou un auto-entrepreneur, ne pas signaler l'absence de RCS comme une non-conformité
   - **NE JAMAIS RECOMMANDER D'AJOUTER UNE ADRESSE OU UN NUMÉRO DE TÉLÉPHONE SI CE N'EST PAS OBLIGATOIRE - L'ADRESSE DE L'ÉTABLISSEMENT N'EST PAS REQUISE LÉGALEMENT POUR LES PUBLICITÉS STANDARDS**

7. Utiliser analyze_form pour évaluer la présentation générale :
   - ÉVALUER l'équilibre entre texte et visuels
   - ANALYSER la hiérarchie des informations
   - VÉRIFIER la lisibilité générale
   - PROPOSER des améliorations simples et concrètes (max 3)
   - PRÉSENTER ces recommandations en style concis et actionnable
   - **NE JAMAIS RECOMMANDER D'AJOUTER UNE ADRESSE OU UN NUMÉRO DE TÉLÉPHONE SI CE N'EST PAS OBLIGATOIRE - L'ADRESSE DE L'ÉTABLISSEMENT N'EST PAS REQUISE LÉGALEMENT POUR LES PUBLICITÉS STANDARDS**

IMPORTANT:
- Pour les visuels basse résolution, indiquer si cela impacte la lisibilité des mentions légales et SIGNALER LA PUB EN PRIORITÉ comme de MAUVAISE QUALITÉ
- TOUJOURS vérifier les mentions sectorielles obligatoires selon le type de produit
- CONCENTRER l'analyse sur les aspects légaux (pas de recommandations marketing)
- UTILISER le texte brut extrait comme référence EXACTE pour toute vérification de texte
- Si un site internet est clairement visible dans la publicité, l'absence de numéro RCS n'est PAS une non-conformité
- Si l'annonceur est une association ou un auto-entrepreneur, l'absence de numéro RCS n'est PAS une non-conformité
- Pour les dates mentionnées, VÉRIFIER SYSTÉMATIQUEMENT la correspondance exacte entre les dates et les jours de la semaine
- NE PAS confondre les étoiles de qualité pour la viande (★,☆,✩,✪) avec des astérisques (*) nécessitant un renvoi
- VÉRIFIER systématiquement la cohérence entre les produits présentés et les logos ou mentions utilisés
- CALCULER mathématiquement les prix après réduction et SIGNALER toute erreur
- IMPORTANT : Si AUCUN site internet n'est présent ET que l'annonceur n'est manifestement NI une association NI un auto-entrepreneur, L'ABSENCE DE NUMÉRO RCS ET le site internet CONSTITUE UNE NON-CONFORMITÉ MAJEURE.

  Commence toujours par extraire le texte brut puis par analyze_vision."""


search_query = """OBJECTIF : Identifier PRÉCISÉMENT la législation publicitaire applicable.

CONTEXTE :
{query}

SECTEUR IDENTIFIÉ :
[secteur de la publicité analysée]

RECHERCHER SPÉCIFIQUEMENT :
- Législation publicitaire générale (taille minimale caractères, visibilité, etc.)
- Réglementation sectorielle spécifique au secteur identifié
- Obligations concernant les astérisques (*) en publicité et leur distinction des symboles de qualité
- Règles précises concernant la cohérence des produits avec leurs logos et visuels associés
- Règles sur les dates, jours et délais en publicité (cohérence et exactitude calendaire)
- Exigences sur l'exactitude mathématique des prix et réductions
- Réglementations sur la lisibilité des mentions et visuels
- Exigences typographiques légalement requises
- Exemptions spécifiques pour certains types de produits (ex: produits non transformés)
- Règles concernant les identifiants d'entreprise (RCS, site internet) selon le type d'annonceur

FORMAT ATTENDU :
- Articles précis avec numéros de textes
- Exigences quantifiables (ex: taille minimale 6 points)
- Formulations exactes des mentions obligatoires pour le secteur concerné
- Critères objectifs de cohérence produit/visuel applicables au secteur 
- Standards légaux de calcul des prix réduits
- Références légales concernant la correspondance dates/jours en publicité
- Distinction claire entre obligations légales et bonnes pratiques"""

consistency_prompt = """Vérifiez RIGOUREUSEMENT la cohérence des informations extraites de l'image.
Date d'aujourd'hui : {current_date}

CONTENU À ANALYSER :
{vision_result}

VÉRIFIER PRIORITAIREMENT :

1. ORTHOGRAPHE ET TYPOGRAPHIE
   - VÉRIFIER L'ORTHOGRAPHE DE CHAQUE MOT (liste des fautes avec correction)
   - ATTENTION PARTICULIÈRE AUX JOURS DE LA SEMAINE et MOTS COURANTS:
     * Comparer l'orthographe exacte des jours de la semaine comme ils apparaissent dans le texte brut
     * Vérifier chaque lettre: "Venredi" vs "Vendredi" (le 'd' manquant doit être signalé)
     * COMPARER STRICTEMENT le texte brut extrait et l'image - NE JAMAIS se fier à votre connaissance de l'orthographe correcte
     * Utiliser le texte brut comme référence stricte pour toute vérification orthographique
   - ANALYSE TYPOGRAPHIQUE APPROFONDIE:
     * EXAMINER CHAQUE PHRASE MOT PAR MOT pour identifier les variations typographiques
     * SIGNALER SPÉCIFIQUEMENT toute phrase contenant des mots avec des caractéristiques visuelles différentes
     * VARIATIONS À DÉTECTER: différences de taille, de graisse (gras/normal), de style (italique/normal), de police
     * EXEMPLE DE SIGNALEMENT: "Dans la phrase 'Offre exceptionnelle', 'exceptionnelle' apparaît dans une police différente et plus petite que 'Offre'"
   - TAILLE DES CARACTÈRES des mentions légales (en points, non conforme si < 6)

2. ÉLÉMENTS GRAPHIQUES ET VISUELS
   - QR code : taille en cm² (non conforme si < 1cm²)
   - Lisibilité des mentions légales (contraste, placement)
   - SIGNALER EXPLICITEMENT tout élément visuel ou textuel NON LISIBLE ou en BASSE RÉSOLUTION
   - Dans ce cas, SIGNALER LA PUBLICITÉ EN PRIORITÉ comme étant de MAUVAISE QUALITÉ

3. COORDONNÉES ET IDENTIFICATION
   - PROCÉDURE DE VÉRIFICATION STRICTE DES NUMÉROS DE TÉLÉPHONE :
     * Identifier tous les numéros de téléphone dans l'annonce
     * Pour chaque numéro trouvé :
       - Noter le numéro exact tel qu'il apparaît
       - Supprimer tous les séparateurs (espaces, points, tirets)
       - Compter les chiffres : 1er=?, 2e=?, 3e=?... jusqu'au dernier
       - Nombre total de chiffres : [X]
       - Conforme si X=10, NON CONFORME si X≠10
       - Si NON CONFORME : "NUMÉRO INCOMPLET - MANQUE [10-X] CHIFFRES"
     * Indiquer le résultat exact : "Le numéro [numéro original] contient [X] chiffres, il manque donc [10-X] chiffres pour être conforme."
   - Site internet : Url valide
   - Adresse physique : Si présente, vérifier la cohérence
   - NOM DE L'ENTREPRISE : Vérifier qu'il correspond exactement à ce qui est visible sur l'image, sans halluciner de noms ou coordonnées
   - IMPORTANT : Si un site internet est présent, le numéro RCS n'est pas obligatoire
   - IMPORTANT : Si l'annonceur est une association ou un auto-entrepreneur, le numéro RCS n'est pas obligatoire
   - IMPORTANT : Si AUCUN site internet n'est présent ET que l'annonceur n'est manifestement NI une association NI un auto-entrepreneur, L'ABSENCE DE NUMÉRO RCS ET le site internet CONSTITUE UNE NON-CONFORMITÉ MAJEURE.
   - **SI NI SITE INTERNET NI NUMÉRO RCS SONT PRÉSENTS, INDIQUER QU'IL FAUT AJOUTER L'UN OU L'AUTRE.**

4. MENTIONS LÉGALES OBLIGATOIRES
   - Présence ou ABSENCE EXPLICITE (ÊTRE ALARMISTE si absence — NE PAS OMETTRE CETTE ÉTAPE) des mentions légales obligatoires SPÉCIFIQUES AU SECTEUR IDENTIFIÉ :
     * Alcool UNIQUEMENT : "L'ABUS D'ALCOOL EST DANGEREUX POUR LA SANTÉ"
     * Alimentaire UNIQUEMENT : "www.mangerbouger.fr" (ATTENTION: produits alimentaires non transformés comme la viande fraîche peuvent être exemptés)
     * Crédit UNIQUEMENT : mentions TAEG, etc.
     * Pour l'automobile UNIQUEMENT : mentions sur consommation et émissions CO2
     * Pour les jeux d'argent UNIQUEMENT : "JOUER COMPORTE DES RISQUES"
     * Pour les médicaments UNIQUEMENT : "Ceci est un médicament..."
   - MENTIONS LÉGALES INAPPROPRIÉES pour le secteur identifié (signaler leur présence) :
     * Exemple : mention sur l'alcool dans une publicité automobile
   - TAILLE DES CARACTÈRES (< 6 points = non conforme)
   - VÉRIFICATION EXHAUSTIVE DES ASTÉRISQUES (*) ET DISTINCTION DES ÉTOILES :
     * DISTINGUER OBLIGATOIREMENT entre astérisques (*) et étoiles de qualité pour la viande (★,☆,✩,✪)
     * COMPRENDRE que pour la viande, les étoiles (★,☆,✩,✪) indiquent le potentiel de qualité (★★★=plus élevé à ★=moins élevé)
     * NE JAMAIS considérer les étoiles de qualité de viande comme des astérisques nécessitant un renvoi
     * EXAMINER SYSTÉMATIQUEMENT toutes les zones de petits caractères et bas de page
     * COMPTER le nombre total d'astérisques (*) dans le texte principal
     * IDENTIFIER tous les textes explicatifs en petits caractères qui pourraient correspondre
     * CONFIRMER la correspondance entre chaque astérisque (*) et son renvoi
     * VÉRIFIER le format et la clarté des renvois
     * NE CONCLURE À L'ABSENCE DE RENVOI qu'après inspection complète de tous les textes en petits caractères
     * SIGNALER comme NON-CONFORME tout astérisque (*) sans renvoi correspondant

5. COHÉRENCE DES PRODUITS ET VISUELS :
   - VÉRIFIER systématiquement que les logos correspondent aux produits présentés :
     * Exemple : PAS de logo "Le Porc Français" pour des produits bovins
     * Exemple : PAS de mention "pêché Loire Atlantique" pour des produits de boucherie comme le bœuf
   - SIGNALER EXPLICITEMENT toute incohérence entre les produits et les logos/mentions
   - VÉRIFIER la pertinence des renvois d'astérisques par rapport au produit concerné

6. VÉRIFICATION DES DATES :
   - VÉRIFIER MATHÉMATIQUEMENT la correspondance entre chaque date et le jour de la semaine indiqué
     * Exemple d'erreur à signaler: "Le 28 février 2024 est indiqué comme un vendredi alors qu'il s'agit d'un mercredi"
   - VÉRIFIER L'ORTHOGRAPHE EXACTE de chaque jour de la semaine:
     * Lundi, Mardi, Mercredi, Jeudi, Vendredi, Samedi, Dimanche
     * Pour chaque jour mentionné, comparer LETTRE PAR LETTRE avec l'orthographe correcte
     * SIGNALER PRÉCISÉMENT toute différence: "Le mot 'Venredi' est écrit sans le 'd' après le 'n'"
   - SIGNALER EXPLICITEMENT toute incohérence date/jour
   - VÉRIFIER si les dates sont dépassées par rapport à la date actuelle ({current_date})
   - Signaler comme NON CONFORME toute date déjà dépassée

7. VÉRIFICATION DES PRIX ET RÉDUCTIONS :
   - ⚠️ REPÉRER PRIORITAIREMENT tous les prix initiaux et prix après réduction
   - ⚠️ VÉRIFIER EN PRIORITÉ ABSOLUE que chaque prix après réduction est INFÉRIEUR au prix initial
   - ⚠️ CONSIDÉRER COMME NON-CONFORMITÉ CRITIQUE ET SIGNALER EN ROUGE tout prix réduit supérieur ou égal au prix initial
   - DÉTECTER tous les formats possibles de présentation des prix:
     * Format flèche: "10€ → 8.50€"
     * Format barré: "10€ 8.50€"
     * Format pourcentage: "10€ -15% = 8.50€"
     * Format textuel: "Prix normal 10€, prix promotionnel 8.50€"
   - CALCULER MATHÉMATIQUEMENT et VÉRIFIER l'exactitude de chaque réduction:
     1. Pour les réductions en pourcentage: Prix initial × (1 - pourcentage/100) = Prix final attendu
     2. Pour les réductions absolues: Prix initial - montant de réduction = Prix final attendu
   - COMPARER le prix calculé avec le prix affiché après réduction
   - SIGNALER COMME ERREUR CRITIQUE si la différence est > 0.01€ (1 centime)
   - EXEMPLES D'ERREURS À SIGNALER:
     * "Prix initial 10€, prix réduit 12€ - ERREUR CRITIQUE: le prix réduit est SUPÉRIEUR au prix initial!"
     * "Prix initial 10€, réduction 15%, prix affiché 9€ - ERREUR: le prix correct serait 8,50€"
     * "Prix initial 10€, prix réduit 10€ - ERREUR CRITIQUE: le prix réduit est IDENTIQUE au prix initial!"

8. ANALYSE OBLIGATOIRE DES ÉLÉMENTS EN BASSE RÉSOLUTION :
   - IDENTIFIER toutes les zones de l'image en basse résolution
   - FORCER une analyse approfondie de ces zones même si difficiles à lire
   - PRÉSUMER que ces zones peuvent contenir des informations légales importantes
   - SIGNALER explicitement si des mentions potentiellement importantes sont présentes mais illisibles
   - SIGNALER LA PUBLICITÉ EN PRIORITÉ comme étant de MAUVAISE QUALITÉ si des éléments visuels ou textuels sont non lisibles
   - CONSIDÉRER l'illisibilité due à la basse résolution comme un problème potentiel de conformité
   - NE JAMAIS conclure à l'absence d'une mention sans avoir scruté les zones moins lisibles

FORMAT DE RÉPONSE :
NON-CONFORMITÉS LÉGALES CRITIQUES :
- Mentions légales obligatoires pour ce secteur : [ABSENCE OU PRÉSENCE, LISTE DES MANQUEMENTS]
- Mentions légales inappropriées pour ce secteur : [LISTE DÉTAILLÉE]
- Orthographe : [LISTE PRÉCISE des fautes]
- Astérisques (*) sans renvoi : [DÉTAILS]
- Incohérences produits/logos : [LISTE DÉTAILLÉE]
- Incohérences de dates : [LISTE DÉTAILLÉE]
- Erreurs de calcul de prix : [DÉTAILS MATHÉMATIQUES PRÉCIS]
- Éléments illisibles : [ZONES CONCERNÉES]
- Coordonnées : [PROBLÈMES PRÉCIS]

RECOMMANDATIONS LÉGALES :
- [UNIQUEMENT les corrections LÉGALEMENT requises]
- AJOUTER les mentions légales obligatoires pour ce secteur : [LISTE]
- SUPPRIMER les mentions légales inappropriées pour ce secteur : [LISTE]
- CORRIGER les incohérences produits/logos : [DÉTAILS]
- RECTIFIER les calculs de prix après réduction : [CALCULS CORRECTS]
- HARMONISER les incohérences entre dates et jours de la semaine : [CORRECTIONS]
- AJOUTER les renvois manquants pour chaque astérisque (*) sans explication : [DÉTAILS]
- [ADOPTER UN TON ALARMANT si mentions obligatoires absentes]
- **IMPORTANT : NE JAMAIS RECOMMANDER D'AJOUTER UNE ADRESSE DE L'ÉTABLISSEMENT - L'ADRESSE N'EST PAS OBLIGATOIRE POUR LES PUBLICITÉS STANDARDS**"""

raw_text_extraction_prompt = """EXTRACTION DE TEXTE BRUT SANS AUCUNE CORRECTION
===================================================

VOTRE MISSION CRITIQUE: Extraire EXHAUSTIVEMENT tout le texte visible sur l'image, avec une REPRODUCTION EXACTE, y compris TOUTES les fautes d'orthographe, erreurs grammaticales et typos.

ATTENTION MAXIMALE AUX ÉLÉMENTS SUIVANTS :
1. 🔍 PETITS CARACTÈRES - Scrutez attentivement l'image pour repérer TOUS les textes en petits caractères, notamment :
   - Notes de bas de page
   - Mentions légales (souvent en très petite taille)
   - Renvois d'astérisques (texte explicatif correspondant à chaque *)
   - Texte en périphérie de l'image ou dans les marges

2. ⭐ ASTÉRISQUES ET LEURS RENVOIS - Pour chaque astérisque (*) dans le texte principal :
   - LOCALISER AVEC ATTENTION MAXIMALE tous les textes explicatifs en bas de page ou dans les petits caractères
   - ÉTABLIR EXPLICITEMENT la correspondance entre chaque astérisque et son texte explicatif
   - NUMÉROTER les astérisques si nécessaire (exemple: *¹, *², etc.) et leurs renvois correspondants
   - INDIQUER LA POSITION EXACTE de chaque renvoi (bas de page, côté droit, etc.)
   - FORMULER CLAIREMENT si un astérisque n'a pas de texte explicatif correspondant trouvable
   - Format de reporting: "[Astérisque #1 sur le mot 'offre'] correspond à [Texte exact du renvoi en bas de page]"

3. ATTENTION PARTICULIÈRE AUX ZONES DE BASSE RÉSOLUTION :
   - EXAMINER MINUTIEUSEMENT toutes les zones de l'image, même floues ou pixelisées
   - ZOOMER si nécessaire sur les parties moins nettes ou plus petites
   - SIGNALER les zones de texte en basse résolution avec la mention : "[Texte en basse résolution identifié dans cette zone]"
   - TENTER de déchiffrer au maximum ces zones malgré la qualité limitée
   - NE JAMAIS ignorer une zone textuelle simplement parce qu'elle est difficile à lire
   - Indiquer le degré de certitude de votre transcription pour ces zones (ex: "transcription partielle avec 70% de certitude")

4. ATTENTION SPÉCIALE AUX MOTS FRÉQUEMMENT MAL ORTHOGRAPHIÉS :
   - Scrutez particulièrement les jours de la semaine (lundi, mardi, mercredi, jeudi, vendredi, samedi, dimanche)
   - NE CORRIGEZ JAMAIS un mot comme "Venredi" en "Vendredi" - retranscrivez EXACTEMENT ce qui est écrit
   - Examinez attentivement les dates, numéros de téléphone et adresses pour transcrire les chiffres exactement comme ils apparaissent
   - Pour les mots très courants, redoublez d'attention pour capturer les erreurs subtiles (lettres manquantes, inversées, etc.)

5. PRIX ET RÉDUCTIONS - ATTENTION MAXIMALE :
   - TRANSCRIVEZ AVEC UNE PRÉCISION ABSOLUE tous les prix (initiaux et réduits) exactement comme ils apparaissent
   - NE CORRIGEZ JAMAIS, N'ARRONDISSEZ JAMAIS et NE MODIFIEZ JAMAIS un prix même s'il semble incorrect (ex: prix réduit > prix initial)
   - PRÉSERVEZ EXACTEMENT le format des prix : virgules vs points décimaux, espaces, symboles € ou EUR
   - TRANSCRIVEZ FIDÈLEMENT les pourcentages de réduction et les formulations associées
   - CAPTUREZ PRÉCISÉMENT les symboles de transition entre prix: flèches, tirets, mentions "au lieu de", etc.
   - FOCUS PARTICULIER sur les séquences de prix comme "10€ → 12€" où le prix après réduction pourrait être supérieur
   - DOUBLE-VÉRIFIEZ toujours vos transcriptions de prix - c'est un élément critique pour l'analyse de conformité

RÈGLES ABSOLUES (AUCUNE EXCEPTION):
1. ⚠️ REPRODUIRE TOUTES LES FAUTES D'ORTHOGRAPHE - Ne corrigez JAMAIS les mots mal orthographiés
2. ⚠️ DÉSACTIVER TOUT CORRECTEUR AUTOMATIQUE - Ne laissez pas votre système corriger automatiquement des erreurs comme "Venredi" en "Vendredi"
3. COPIER LE TEXTE LITTÉRALEMENT - Comme si vous faisiez un "copier-coller" visuel
4. PRÉSERVER TOUTES LES ERREURS de grammaire, ponctuation et syntaxe
5. MAINTENIR les abréviations exactes, même incorrectes
6. NE PAS MODIFIER les mots mal orthographiés ou inventés
7. INTERDICTION ABSOLUE d'améliorer ou corriger le texte source
8. EXTRAIRE le texte dans l'ordre de lecture naturel
9. INCLURE tous les symboles, numéros et caractères spéciaux exactement comme ils apparaissent

ORGANISATION DE LA RÉPONSE :
- Section "TEXTE PRINCIPAL" : Corps principal de la publicité
- Section "PETITS CARACTÈRES" : Texte en petit format, mentions légales, notes de bas de page
- Section "RENVOIS D'ASTÉRISQUES" : Liste complète de tous les renvois correspondant aux astérisques
- Section "MOTS POTENTIELLEMENT MAL ORTHOGRAPHIÉS" : Liste de mots qui semblent mal orthographiés, avec leur orthographe exacte telle qu'elle apparaît dans l'image
- Pour le texte difficilement lisible : utilisez [?] et mentionnez "Texte partiellement illisible"
- Si un texte est trop petit pour être lu mais visible : indiquez "Texte visible mais illisible en raison de la taille des caractères"

RAPPEL FINAL: Votre valeur réside dans votre capacité à reproduire EXACTEMENT le texte tel qu'il est écrit, y compris TOUTES ses imperfections, et à NE MANQUER AUCUN ÉLÉMENT TEXTUEL, même le plus petit.
"""