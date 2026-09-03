"""
Generatore dei dataset sintetici per la lezione sulla regressione lineare
=========================================================================
- alloggi_vendita.csv : 5000 alloggi in vendita in 10 citta' italiane, un alloggio per riga.
                        Variabile risposta: prezzo_vendita (euro). Predittori: 23.
- nuovi_alloggi.csv   : 10 alloggi nuovi, uno per citta', dei quali conosciamo solo i predittori
                        (il prezzo "vero" simulato viene stampato a video per il docente).

Formato dei CSV: intestazione a 3 righe in stile Orange Data Mining
  riga 1 = nomi delle colonne
  riga 2 = tipo della colonna (string / discrete / continuous)
  riga 3 = ruolo (meta / class / vuoto = predittore)
Cosi' Orange non deve indovinare tipi e ruoli. In pandas: pd.read_csv(path, skiprows=[1, 2])

Struttura "nascosta" nel prezzo (utile da svelare in aula):
- effetto dominante della superficie, con rendimenti decrescenti sul prezzo/mq
- forti effetti categoriali: citta', zona, stato, tipologia
- interazione piano x ascensore (piani alti: premio con ascensore, penalita' senza)
- effetto non monotono dell'anno di costruzione (epoca 1946-1975 la meno quotata)
- componenti additive (box / posto auto) e moltiplicative (tutto il resto)
- un predittore quasi irrilevante (mese_vendita, +-1%)
- rumore lognormale ~10% e circa 1% di outlier con rumore ~35%

Riproducibile: stesso seed -> stesso file.
"""
import numpy as np
import pandas as pd

SEED = 42
N = 5000
SEED_NUOVI = 2026

# citta: (peso nel campione, prezzo base euro/mq in semicentro stato "buono", valore di un box)
CITTA = {
    "Milano":  (0.20, 5000, 35000),
    "Roma":    (0.20, 3300, 30000),
    "Torino":  (0.12, 2000, 18000),
    "Napoli":  (0.10, 2600, 25000),
    "Bologna": (0.08, 3300, 25000),
    "Firenze": (0.07, 4000, 28000),
    "Genova":  (0.06, 1800, 18000),
    "Bari":    (0.06, 2000, 18000),
    "Padova":  (0.06, 2300, 20000),
    "Palermo": (0.05, 1400, 15000),
}
ZONE = ["centro", "semicentro", "periferia"]
ZONA_P = [0.25, 0.40, 0.35]
ZONA_MULT = {"centro": 1.45, "semicentro": 1.00, "periferia": 0.72}

TIPOLOGIE = ["appartamento", "attico", "villetta"]
TIPOLOGIA_P = {"centro": [0.90, 0.09, 0.01],
               "semicentro": [0.84, 0.08, 0.08],
               "periferia": [0.72, 0.06, 0.22]}
TIPOLOGIA_MULT = {"appartamento": 1.00, "attico": 1.15, "villetta": 1.05}

EPOCHE = [(1900, 1945), (1946, 1975), (1976, 1999), (2000, 2015), (2016, 2025)]
EPOCA_P = {"centro":     [0.45, 0.32, 0.13, 0.06, 0.04],
           "semicentro": [0.15, 0.40, 0.25, 0.12, 0.08],
           "periferia":  [0.05, 0.30, 0.30, 0.20, 0.15]}
EPOCA_MULT = [1.00, 0.92, 0.96, 1.02, 1.06]   # effetto non monotono dell'anno

STATO_MULT = {"da_ristrutturare": 0.80, "buono": 1.00, "ristrutturato": 1.12, "nuovo": 1.22}
CLASSE_MULT = {"A": 1.08, "B": 1.05, "C": 1.02, "D": 1.00, "E": 0.98, "F": 0.96, "G": 0.93}
ESPOSIZIONE_MULT = {"nord": 0.97, "sud": 1.03, "est": 1.00, "ovest": 1.00, "doppia": 1.03}


def scegli(rng, opzioni, p):
    return str(rng.choice(opzioni, p=p))


def fattore_piano(piano, ascensore):
    """Interazione piano x ascensore."""
    if piano == 0:
        return 0.94
    if piano == 1:
        return 0.98
    if ascensore == "si":
        return min(1.10, 1.0 + 0.012 * (piano - 2))
    return max(0.75, 1.0 - 0.05 * (piano - 1))


def genera_alloggio(i, rng, prefisso="IMM"):
    citta = scegli(rng, list(CITTA), [v[0] for v in CITTA.values()])
    _, base_mq, valore_box = CITTA[citta]
    zona = scegli(rng, ZONE, ZONA_P)
    tipologia = scegli(rng, TIPOLOGIE, TIPOLOGIA_P[zona])

    # --- localizzazione ---
    if zona == "centro":
        dist_centro = rng.uniform(0.2, 2.0)
        dist_trasp = rng.lognormal(np.log(300), 0.5)
    elif zona == "semicentro":
        dist_centro = rng.uniform(2.0, 6.0)
        dist_trasp = rng.lognormal(np.log(600), 0.6)
    else:
        dist_centro = rng.uniform(6.0, 15.0)
        dist_trasp = rng.lognormal(np.log(1200), 0.7)
    dist_centro = round(float(dist_centro), 1)
    dist_trasp = int(np.clip(dist_trasp, 50, 5000))

    # --- dimensioni ---
    if tipologia == "appartamento":
        sup = np.clip(rng.lognormal(np.log(80), 0.35), 30, 220)
    elif tipologia == "attico":
        sup = np.clip(rng.lognormal(np.log(120), 0.30), 60, 300)
    else:
        sup = np.clip(rng.lognormal(np.log(150), 0.25), 90, 350)
    sup = int(round(sup))
    locali = int(np.clip(round(sup / 28 + rng.normal(0, 0.6)), 1, 8))
    bagni = 1 + (sup > 70) + (sup > 140) + (sup > 220)
    if rng.random() < 0.15:
        bagni += int(rng.choice([-1, 1]))
    bagni = int(np.clip(bagni, 1, 4))

    # --- epoca e stato ---
    k = int(rng.choice(5, p=EPOCA_P[zona]))
    anno = int(rng.integers(EPOCHE[k][0], EPOCHE[k][1] + 1))
    if anno >= 2016:
        stato = scegli(rng, ["nuovo", "buono"], [0.85, 0.15])
    else:
        stato = scegli(rng, ["da_ristrutturare", "buono", "ristrutturato"], [0.22, 0.50, 0.28])

    punteggio = [0.5, 1.0, 2.0, 4.0, 6.0][k]          # base per epoca
    if stato == "ristrutturato":
        punteggio += 1.5
    if stato == "nuovo":
        punteggio = 6.0 + rng.random()
    punteggio += rng.normal(0, 0.8)
    classe = "GFEDCBA"[int(np.clip(round(punteggio), 0, 6))]

    # --- edificio ---
    if tipologia == "villetta":
        piani_edificio = int(rng.choice([2, 3], p=[0.7, 0.3]))
        piano, ascensore = 0, "no"
    else:
        if zona == "centro":
            piani_edificio = int(rng.integers(3, 8))
        elif zona == "semicentro":
            piani_edificio = int(rng.integers(3, 10))
        else:
            piani_edificio = int(rng.integers(2, 12))
        piano = piani_edificio if tipologia == "attico" else int(rng.integers(0, piani_edificio + 1))
        if piani_edificio <= 3:
            p_asc = 0.25
        elif anno >= 1976:
            p_asc = 0.95
        else:
            p_asc = 0.55
        ascensore = "si" if rng.random() < p_asc else "no"

    if tipologia == "villetta":
        riscaldamento = "autonomo"
    else:
        p_centr = 0.60 if (anno < 1990 and piani_edificio >= 4) else 0.25
        riscaldamento = "centralizzato" if rng.random() < p_centr else "autonomo"

    esposizione = scegli(rng, ["nord", "sud", "est", "ovest", "doppia"], [0.18, 0.25, 0.15, 0.15, 0.27])

    # --- pertinenze ---
    if tipologia == "villetta":
        balconi = int(rng.choice([0, 1, 2], p=[0.50, 0.35, 0.15]))
    elif anno < 1946:
        balconi = int(rng.choice([0, 1, 2, 3], p=[0.45, 0.40, 0.12, 0.03]))
    else:
        balconi = int(rng.choice([0, 1, 2, 3], p=[0.15, 0.45, 0.32, 0.08]))
    p_terr = {"attico": 0.90, "villetta": 0.40, "appartamento": 0.07}[tipologia]
    terrazzo = "si" if rng.random() < p_terr else "no"
    p_giard = 0.92 if tipologia == "villetta" else (0.35 if piano == 0 else 0.02)
    giardino = "si" if rng.random() < p_giard else "no"
    if tipologia == "villetta":
        p_auto = [0.10, 0.20, 0.70]
    elif zona == "centro":
        p_auto = [0.82, 0.10, 0.08]
    elif zona == "semicentro":
        p_auto = [0.55, 0.20, 0.25]
    else:
        p_auto = [0.35, 0.25, 0.40]
    posto_auto = scegli(rng, ["nessuno", "posto_auto", "box"], p_auto)
    p_cant = 0.50 if tipologia == "villetta" else (0.65 if anno < 1976 else 0.45)
    cantina = "si" if rng.random() < p_cant else "no"

    if tipologia == "villetta":
        spese = int(max(0, rng.normal(15, 12)))
    else:
        spese = (40 + 1.1 * sup + 35 * (ascensore == "si") + 25 * (riscaldamento == "centralizzato")
                 + 15 * (zona == "centro") + rng.normal(0, 25))
        spese = int(max(20, round(spese)))

    mese = int(rng.integers(1, 13))

    # --- prezzo: modello moltiplicativo in scala log + componente additiva ---
    log_p = np.log(base_mq) + np.log(sup) - 0.10 * np.log(sup / 80)   # rendimenti decrescenti sul prezzo/mq
    log_p += np.log(ZONA_MULT[zona]) - 0.02 * dist_centro - 0.00004 * dist_trasp
    log_p += np.log(TIPOLOGIA_MULT[tipologia])
    log_p += np.log(STATO_MULT[stato]) + np.log(EPOCA_MULT[k]) + np.log(CLASSE_MULT[classe])
    if tipologia != "villetta":
        log_p += np.log(fattore_piano(piano, ascensore))
        log_p += -0.0003 * (spese - 150)
    log_p += np.log(ESPOSIZIONE_MULT[esposizione])
    log_p += np.log(1.02 if riscaldamento == "autonomo" else 1.00)
    log_p += np.log(1.06 if terrazzo == "si" else 1.00) + np.log(1 + 0.015 * balconi)
    log_p += np.log(1.07 if giardino == "si" else 1.00) + np.log(1.01 if cantina == "si" else 1.00)
    log_p += 0.01 * np.sin(2 * np.pi * (mese - 3) / 12)              # stagionalita' trascurabile
    sigma = 0.35 if rng.random() < 0.01 else 0.10                    # ~1% di outlier
    log_p += rng.normal(0, sigma)

    prezzo = np.exp(log_p) + {"nessuno": 0, "posto_auto": 0.5 * valore_box, "box": valore_box}[posto_auto]
    prezzo = int(round(prezzo / 1000) * 1000)

    return {
        "id_immobile": f"{prefisso}-{i:05d}" if prefisso == "IMM" else f"{prefisso}-{i:02d}",
        "citta": citta,
        "zona": zona,
        "distanza_centro_km": dist_centro,
        "distanza_trasporti_m": dist_trasp,
        "tipologia": tipologia,
        "superficie_mq": sup,
        "locali": locali,
        "bagni": bagni,
        "piano": piano,
        "piani_edificio": piani_edificio,
        "ascensore": ascensore,
        "anno_costruzione": anno,
        "stato": stato,
        "classe_energetica": classe,
        "riscaldamento": riscaldamento,
        "esposizione": esposizione,
        "balconi": balconi,
        "terrazzo": terrazzo,
        "giardino": giardino,
        "posto_auto": posto_auto,
        "cantina": cantina,
        "spese_condominio_mese": spese,
        "mese_vendita": mese,
        "prezzo_vendita": prezzo,
    }



TIPI = {"id_immobile": "string", "prezzo_vendita": "continuous",
        **{c: "discrete" for c in ["citta", "zona", "tipologia", "ascensore", "stato", "classe_energetica",
                                   "riscaldamento", "esposizione", "terrazzo", "giardino", "posto_auto", "cantina"]},
        **{c: "continuous" for c in ["distanza_centro_km", "distanza_trasporti_m", "superficie_mq", "locali", "bagni",
                                     "piano", "piani_edificio", "anno_costruzione", "balconi",
                                     "spese_condominio_mese", "mese_vendita"]}}
RUOLI = {"id_immobile": "meta", "prezzo_vendita": "class"}


def scrivi_csv_orange(df, path):
    """CSV con intestazione a 3 righe (nomi / tipi / ruoli), leggibile da Orange senza configurazione."""
    cols = list(df.columns)
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        f.write(",".join(TIPI[c] for c in cols) + "\n")
        f.write(",".join(RUOLI.get(c, "") for c in cols) + "\n")
        df.to_csv(f, index=False, header=False, lineterminator="\n")


# dieci alloggi nuovi, uno per citta', con tipologia e zona prefissate per avere casi variati
NUOVI_SPEC = [("Milano", "attico", "centro"), ("Roma", "appartamento", "semicentro"),
              ("Torino", "villetta", "periferia"), ("Napoli", "appartamento", "centro"),
              ("Bologna", "appartamento", "semicentro"), ("Firenze", "appartamento", "centro"),
              ("Genova", "appartamento", "periferia"), ("Bari", "villetta", "periferia"),
              ("Padova", "appartamento", "semicentro"), ("Palermo", "appartamento", "periferia")]


def genera_nuovi():
    rng = np.random.default_rng(SEED_NUOVI)
    trovati = {}
    for i in range(1, 20001):
        a = genera_alloggio(i, rng)
        chiave = (a["citta"], a["tipologia"], a["zona"])
        if chiave in NUOVI_SPEC and chiave not in trovati:
            trovati[chiave] = a
        if len(trovati) == len(NUOVI_SPEC):
            break
    righe = []
    for k, chiave in enumerate(NUOVI_SPEC, start=1):
        a = dict(trovati[chiave])
        a["id_immobile"] = f"NUOVO-{k:02d}"
        righe.append(a)
    return pd.DataFrame(righe)


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    df = pd.DataFrame(genera_alloggio(i, rng) for i in range(1, N + 1))
    scrivi_csv_orange(df, "alloggi_vendita.csv")
    print("alloggi_vendita.csv", df.shape)
    print(df["prezzo_vendita"].describe().round(0))

    nuovi = genera_nuovi()
    scrivi_csv_orange(nuovi.drop(columns=["prezzo_vendita"]), "nuovi_alloggi.csv")
    print("\nnuovi_alloggi.csv", nuovi.shape[0], "righe - prezzi simulati (solo per il docente):")
    print(nuovi[["id_immobile", "citta", "zona", "tipologia", "superficie_mq", "stato", "prezzo_vendita"]].to_string(index=False))
