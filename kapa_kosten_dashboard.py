import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.optimize import linprog
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD

# Fixkosten und variable Produktionskosten. Produkte A, B und C 
fixkosten = {"Aachen": 5000, "Berlin": 7000, "Coesfeld": 4000}
produktionskosten = {
    "Aachen": {"A": 100, "B": 80, "C": 120},
    "Berlin": {"A": 110, "B": None, "C": 130},
    "Coesfeld": {"A": None, "B": 90, "C": 140},
}

# Startkapazitäten
original_kapa = {
    "Aachen": {"A": 3, "B": 10, "C": 2},
    "Berlin": {"A": 4, "B": 0, "C": 5},
    "Coesfeld": {"A": 0, "B": 2, "C": 1},
}

# Berechnung der Gesamtkosten je Werk
def berechne_gesamtkosten_je_werk(original_kapa):
    gesamtkosten = {}
    for werk in original_kapa:
        fixkosten_werk = fixkosten[werk]
        produktionskosten_werk = sum(
            produktionskosten[werk][prod] * menge
            for prod, menge in original_kapa[werk].items()
            if produktionskosten[werk][prod] is not None
        )
        gesamtkosten[werk] = fixkosten_werk + produktionskosten_werk
    return gesamtkosten
    
# Berechnung der Gesamtkosten insgesamt
def berechne_gesamtkosten(original_kapa):
    fixkosten_summe = sum(fixkosten[werk] for werk in fixkosten)
    produktionskosten_summe = sum(
        produktionskosten[werk][prod] * menge
        for werk, produkte in original_kapa.items()
        for prod, menge in produkte.items()
        if produktionskosten[werk][prod] is not None
    )
    return fixkosten_summe + produktionskosten_summe
    
# Streamlit
st.title("Interaktives Produktionskosten-Dashboard")

# Initialisieren der Session-State Variablen
if "original_kapa" not in st.session_state:
    st.session_state.original_kapa = original_kapa.copy()

# Header
st.sidebar.header("Kapazitätssteuerung")
aktuelle_kapa = {}

# Slider für jedes Werk und Produkt
for werk, produkte in original_kapa.items():
    st.sidebar.subheader(f"Kapazitäten in {werk}")
    aktuelle_kapa[werk] = {}
    for produkt, menge in produkte.items():
        if menge is not None:
            neue_menge = st.sidebar.slider(
                f"{produkt} in {werk}", 0, 20, menge, step=1
            )
            aktuelle_kapa[werk][produkt] = neue_menge

# Berechnung der Gesamtkosten
gesamt_kosten_je_werk = berechne_gesamtkosten_je_werk(aktuelle_kapa)
gesamt_kosten = berechne_gesamtkosten(aktuelle_kapa)

# Anzeige der Ergebnisse
st.header("Aktuelle Produktionsverteilung")
st.write(pd.DataFrame(aktuelle_kapa))

st.header("Gesamtkosten")
st.metric(label="Gesamtkosten (€)", value=f"{gesamt_kosten:,.2f}")
st.write(pd.DataFrame.from_dict(gesamt_kosten_je_werk, orient='index', columns=["Gesamtkosten (€)"]))


# Visualisierung der Produktionsverteilung
st.header("Produktionsverteilung je Werk")
kapa_df = pd.DataFrame(aktuelle_kapa).T
st.bar_chart(kapa_df)

# Button zum Zurücksetzen der Werte
if st.sidebar.button("🔄 Zurücksetzen auf ursprüngliche Werte"):
    st.session_state.original_kapa = original_kapa.copy()
    st.experimental_rerun()  # Streamlit neu laden, um die Werte zurückzusetzen

# Streamlit UI
st.title("Optimierung der Produktionskapazitäten")

st.sidebar.header("Nachfrage pro Produkt")
nachfrage = {
    "A": st.sidebar.slider("Nachfrage für Produkt A", 0, 20, 5, step=1),
    "B": st.sidebar.slider("Nachfrage für Produkt B", 0, 20, 8, step=1),
    "C": st.sidebar.slider("Nachfrage für Produkt C", 0, 20, 4, step=1),
}

# Funktion zur Berechnung der optimalen Produktionsverteilung mit PuLP
def berechne_optimal_produktionsverteilung_pulp(nachfrage):
    werke = list(original_kapa.keys())
    produkte = list(nachfrage.keys())

    # Optimierungsproblem definieren
    prob = LpProblem("Minimiere_Produktionskosten", LpMinimize)

    # Entscheidungsvariablen (Produktionsmengen)
    x = {(p, w): LpVariable(f"x_{p}_{w}", lowBound=0, cat="Continuous") for p in produkte for w in werke}

    # Zielfunktion: Minimierung der Produktionskosten
    prob += lpSum(produktionskosten[w][p] * x[p, w] for p in produkte for w in werke if produktionskosten[w][p] is not None)

    # Nachfragebedingungen (Gesamtproduktion pro Produkt muss Nachfrage entsprechen)
    for p in produkte:
        prob += lpSum(x[p, w] for w in werke) == nachfrage[p]

    # Kapazitätsbeschränkungen
    for w in werke:
        for p in produkte:
            if original_kapa[w][p] > 0:
                prob += x[p, w] <= original_kapa[w][p]

    # Optimierung lösen
    prob.solve(PULP_CBC_CMD(msg=False))

    # Ergebnisse speichern
    if prob.status == 1:
        opt_df = pd.DataFrame({w: [x[p, w].varValue for p in produkte] for w in werke}, index=produkte)
        return opt_df
    else:
        return "Keine optimale Lösung gefunden."

# Streamlit UI
st.title("Optimale Produktionsverteilung mit PuLP berechnen")

if st.button("🔍 Berechne optimale Produktionsverteilung"):
    optimales_ergebnis = berechne_optimal_produktionsverteilung_pulp(nachfrage)
    st.header("Optimale Produktionsverteilung")
    st.write(optimales_ergebnis)
