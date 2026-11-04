"""
glace/data.py — Chargement des données observationnelles

Charge les profils de gaz mesurés dans les carottes de glace à partir du
fichier Excel Table_S1_122025_for manuscript.xlsx.

Colonnes utilisées (feuille CampCentury, 4 lignes d'en-tête à ignorer) :
    B  → hauteur au-dessus du bed          [m]
    D  → contenu total en gaz (TAC)        [mL_STP / kg_ice]
    L  → fraction molaire Ar dans le gaz   [%]
    W  → concentration Ar                  [mol / kg_ice]

Référence des paramètres de perte de gaz calibrés à partir de ces données :
    gas_loss   = 1.8e-6 yr⁻¹   (taux de disparition max des clathrates)
    z_char     = 8.0 m          (hauteur caractéristique de la zone de perte)
    z_gl_limit = 20.0 m         (limite supérieure de la zone de perte)
"""

import os
import numpy as np

# Chemin par défaut vers le fichier Excel (relatif au répertoire GLACE/)
_DEFAULT_EXCEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "Table_S1_122025_for manuscript.xlsx",
)


def load_campcentury(path=None, rho_ice=917.0):
    """Charge le profil Camp Century depuis le fichier Excel de laboratoire.

    Les paramètres de perte de gaz dans params.py (gas_loss, z_char, z_gl_limit)
    sont calibrés à partir de la colonne 'Gas Content' de ce fichier : on observe
    une chute brutale du contenu en gaz total (et de la fraction Ar) dans les
    ~2 premiers mètres au-dessus du bed, ce qui motive le profil exponentiel
    λ(z) = gas_loss · exp(−z / z_char).

    Parameters
    ----------
    path     : str, optional — chemin vers l'Excel. Par défaut : data/Table_S1_122025_for manuscript.xlsx
    rho_ice  : float — densité de la glace [kg m⁻³] pour convertir mol/kg → mol/m³

    Returns
    -------
    data : dict avec les clés
        'z'           : (n,) float — hauteur au-dessus du bed [m]
        'gas_content' : (n,) float — contenu total en gaz     [mL_STP / kg_ice]
        'f_Ar'        : (n,) float — fraction molaire Ar      [mol/mol]  (col L / 100)
        'C_Ar_molkg'  : (n,) float — concentration Ar         [mol / kg_ice]
        'C_Ar'        : (n,) float — concentration Ar         [mol / m³_ice]

    Raises
    ------
    ImportError  — si openpyxl n'est pas installé
    FileNotFoundError — si le fichier Excel est introuvable
    """
    try:
        import openpyxl
    except ImportError as e:
        raise ImportError(
            "openpyxl est nécessaire pour lire le fichier Excel.\n"
            "Installe-le avec :  pip install openpyxl"
        ) from e

    if path is None:
        path = _DEFAULT_EXCEL

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Fichier de données introuvable : {path}\n"
            "Place le fichier Excel dans le répertoire data/ ou passe son chemin en argument."
        )

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb["CampCentury"]

    rows = list(ws.iter_rows(min_row=5, values_only=True))  # 4 lignes d'en-tête

    z_list            = []
    gas_content_list  = []
    f_Ar_list         = []
    C_Ar_molkg_list   = []

    for row in rows:
        try:
            z_val  = float(row[1])   # col B (index 1)
            gc_val = float(row[3])   # col D (index 3)
            ar_pct = float(row[11])  # col L (index 11)
            ar_mol = float(row[22])  # col W (index 22)
        except (TypeError, ValueError, IndexError):
            continue   # ignore lignes incomplètes / vides

        z_list.append(z_val)
        gas_content_list.append(gc_val)
        f_Ar_list.append(ar_pct / 100.0)   # % → fraction
        C_Ar_molkg_list.append(ar_mol)

    wb.close()

    z            = np.array(z_list,           dtype=float)
    gas_content  = np.array(gas_content_list, dtype=float)
    f_Ar         = np.array(f_Ar_list,        dtype=float)
    C_Ar_molkg   = np.array(C_Ar_molkg_list,  dtype=float)
    C_Ar         = C_Ar_molkg * rho_ice        # mol/kg → mol/m³

    # Tri par hauteur croissante
    order = np.argsort(z)
    return {
        'z'          : z[order],
        'gas_content': gas_content[order],
        'f_Ar'       : f_Ar[order],
        'C_Ar_molkg' : C_Ar_molkg[order],
        'C_Ar'       : C_Ar[order],
    }


def load_grip(path=None, rho_ice=917.0):
    """Charge le profil GRIP depuis la feuille 'GRIP' du même fichier Excel.

    Même structure de colonnes que CampCentury.

    Parameters
    ----------
    path    : str, optional
    rho_ice : float

    Returns
    -------
    dict — mêmes clés que load_campcentury()
    """
    try:
        import openpyxl
    except ImportError as e:
        raise ImportError(
            "openpyxl est nécessaire.  pip install openpyxl"
        ) from e

    if path is None:
        path = _DEFAULT_EXCEL

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb["GRIP"]

    rows = list(ws.iter_rows(min_row=5, values_only=True))

    z_list = []; gas_content_list = []; f_Ar_list = []; C_Ar_molkg_list = []

    for row in rows:
        try:
            z_val  = float(row[1])
            gc_val = float(row[3])
            ar_pct = float(row[11])
            ar_mol = float(row[22])
        except (TypeError, ValueError, IndexError):
            continue

        z_list.append(z_val)
        gas_content_list.append(gc_val)
        f_Ar_list.append(ar_pct / 100.0)
        C_Ar_molkg_list.append(ar_mol)

    wb.close()

    z           = np.array(z_list,           dtype=float)
    gas_content = np.array(gas_content_list, dtype=float)
    f_Ar        = np.array(f_Ar_list,        dtype=float)
    C_Ar_molkg  = np.array(C_Ar_molkg_list,  dtype=float)
    C_Ar        = C_Ar_molkg * rho_ice

    order = np.argsort(z)
    return {
        'z'          : z[order],
        'gas_content': gas_content[order],
        'f_Ar'       : f_Ar[order],
        'C_Ar_molkg' : C_Ar_molkg[order],
        'C_Ar'       : C_Ar[order],
    }
