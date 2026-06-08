"""
Jednofazowa optymalizacja metaheurystyczna dla DIgSILENT PowerFactory.
Zredukowano do elementów symetrycznych (bez rozróżnienia na fazy).
Autor: ChatGPT (Copilot Space) dla dprzepiorka
Repo: dprzepiorka/ELVTF
"""

import sys
import os
import time
import math
import random
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from CEO import CEO
from PSO import PSO
from PO import PO
from ECO import ECO
from KEO import KEO
from AIG_ACNC import AIG_ACNC     #Pawła 1


# Scieżka do PowerFactory (dopasuj do instalacji)
sys.path.append(r"C:\Program Files\DIgSILENT\PowerFactory 2024 SP5A\Python\3.12")
try:
    import powerfactory
except Exception:
    powerfactory = None
    print("Warning: powerfactory module not available (script still importable for offline tests).")

from scipy.optimize import differential_evolution

# -------------------------
# KONFIGURACJA
# -------------------------
EXCEL_FILE = r"N:\ksiz\STUDIA_DOKTORANCKIE\PRZEPIORKA\Modele sieci\Srednie napiecie\APS\Optymalizacja\dane.xlsx"
PROJECT_NAME = "IEEE69-SN"
OUT_FILE = r"N:\ksiz\STUDIA_DOKTORANCKIE\PRZEPIORKA\Modele sieci\Srednie napiecie\APS\Optymalizacja\wyniki_Opt.xlsx"
START_FILE = r"N:\ksiz\STUDIA_DOKTORANCKIE\PRZEPIORKA\Modele sieci\Srednie napiecie\APS\Optymalizacja\wyniki_Pocz.xlsx"
USER = "minik"

METHOD = "PSO"          #CEO     PSO     PO      ECO     KEO 
OBJECTIVE = "VoltageTarget"

N_ITER = 1000
N_PARTICLES = 100
N = 1
W = 0.7
C1 = 1.5
C2 = 1.5
PENALTY = 1e6
LARGE_PENALTY_MULTIPLIER = 1e4
RANDOM_SEED = 42

EVAL_DELAY = 0.01

# Limity magazynów energii (jedna faza)
STORAGE_P_MIN = -7.0
STORAGE_P_MAX = 7.0
STORAGE_Q_MIN = -0.001
STORAGE_Q_MAX = 0.001

VOLTAGE_MIN = 0.9
VOLTAGE_MAX = 1.1
LOAD_MAX = 100.0  # [%]

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

WEIGHTS = np.array([ 0.817, 0.0871, 0.096,], dtype=float)  # np. [napięcie, strata, magazyn]

COMPONENT_HISTORY = []
STORAGE_CANDIDATES = []

# -------------------------
# Helper functions
# -------------------------
def find_element(app, name, pf_class):
    try:
        objs = app.GetCalcRelevantObjects(f"{name}.{pf_class}")
        if objs:
            return objs[0]
    except Exception:
        pass
    try:
        for o in app.GetCalcRelevantObjects(f"*.{pf_class}"):
            if getattr(o, "loc_name", None) == name:
                return o
    except Exception:
        pass
    return None

def load_vars(file_path, sheet="Vars"):
    df = pd.read_excel(file_path, sheet_name=sheet)
    df = df.rename(columns={c.strip().lower(): c.strip() for c in df.columns})
    vars_list = []
    for _, row in df.iterrows():
        vars_list.append({
            "name": str(row["name"]).strip(),
            "pf_class": str(row["pf_class"]).strip(),
            "attr": str(row["attr"]).strip(),
            "min": float(row["min"]),
            "max": float(row["max"])
        })
    return vars_list

def set_single_attribute(app, var, value):
    elm = find_element(app, var["name"], var["pf_class"])
    if elm is None:
        return False
    try:
        elm.SetAttribute(var["attr"], float(value))
        return True
    except Exception:
        return False

def apply_solution(app, vars_def, x):
    for idx, var in enumerate(vars_def):
        try:
            if var["name"].startswith("storage_"):
                continue
            set_single_attribute(app, var, float(x[idx]))
        except Exception:
            continue

def load_storage_candidates(file_path, sheet="StorageCandidates"):
    try:
        df = pd.read_excel(file_path, sheet_name=sheet)
        df = df.fillna("")
        candidates = []
        for _, r in df.iterrows():
            candidates.append({
                "node": str(r.get("node", "")).strip(),
                "elem": str(r.get("elem", "")).strip()  # zakładamy jeden element
            })
        return candidates
    except Exception as e:
        print(f"Nie udało się wczytać {sheet}: {e}")
        return []

def _set_element_attr_safe(elm, attr, val):
    try:
        elm.SetAttribute(attr, float(val))
        return True
    except Exception:
        return False

def apply_storage_on_node(app, candidate, P, Q):
    prev = {}
    ename = candidate.get("elem")
    elm = find_element(app, ename, "ElmGenstat") or find_element(app, ename, "ElmSym") \
        or find_element(app, ename, "ElmPvsys") or find_element(app, ename, "ElmLod")
    if not elm:
        prev[ename] = None
    else:
        got = {
            "pgini": elm.GetAttribute("pgini"),
            "qgini": elm.GetAttribute("qgini"),
        }
        _set_element_attr_safe(elm, "pgini", P)
        _set_element_attr_safe(elm, "qgini", Q)
        prev[ename] = got
    return prev

def reset_storage_on_node(app, candidate, prev_values):
    ename = candidate.get("elem")
    prev = prev_values.get(ename, None)
    elm = find_element(app, ename, "ElmGenstat") or find_element(app, ename, "ElmSym") \
        or find_element(app, ename, "ElmPvsys") or find_element(app, ename, "ElmLod")
    if not elm:
        return
    if prev is None:
        _set_element_attr_safe(elm, "pgini", 0.0)
        _set_element_attr_safe(elm, "qgini", 0.0)
    else:
        _set_element_attr_safe(elm, "pgini", prev.get("pgini", 0.0))
        _set_element_attr_safe(elm, "qgini", prev.get("qgini", 0.0))

def apply_solution_with_storage(app, vars_def, x):
    for idx, var in enumerate(vars_def):
        name = var["name"]
        if name.startswith("storage_"):
            continue
        val = float(x[idx])
        val = max(var["min"], min(var["max"], val))
        set_single_attribute(app, var, val)

    global STORAGE_CANDIDATES
    if not STORAGE_CANDIDATES:
        return None, None

    try:
        idx_node = next(i for i, v in enumerate(vars_def) if v["name"] == "storage_node_index")
    except StopIteration:
        return None, None
    node_val = int(round(x[idx_node]))
    node_val = max(0, min(len(STORAGE_CANDIDATES) - 1, node_val))
    candidate = STORAGE_CANDIDATES[node_val]
    try:
        idx_p = next(i for i, v in enumerate(vars_def) if v["name"] == "storage_P")
        P = float(x[idx_p])
        Q = float(x[idx_p+1])
    except Exception:
        return candidate, None
    prev = apply_storage_on_node(app, candidate, P, Q)
    return candidate, prev

# Zapis i ustawianie parametrów z Excela
def load_and_set_elements_from_excel(app, file_path):
    try:
        loads_df = pd.read_excel(file_path, sheet_name="Loads")
        gens_df = pd.read_excel(file_path, sheet_name="Generators")
        pv_df = pd.read_excel(file_path, sheet_name="PV")
        ES_df = pd.read_excel(file_path, sheet_name="StatGen")
    except Exception as e:
        print(f"Błąd przy wczytywaniu pliku {file_path}: {e}")
        return
    for _, row in loads_df.iterrows():
        name = str(row["name"]).strip()
        elm = find_element(app, name, "ElmLod")
        if not elm:
            continue
        _set_element_attr_safe(elm, "plini", float(row["P"]))
        _set_element_attr_safe(elm, "qlini", float(row["Q"]))
    for _, row in gens_df.iterrows():
        name = str(row["name"]).strip()
        elm = find_element(app, name, "ElmSym")
        if not elm:
            continue
        _set_element_attr_safe(elm, "pgini", float(row["P"]))
        _set_element_attr_safe(elm, "qgini", float(row["Q"]))
    for _, row in pv_df.iterrows():
        name = str(row["name"]).strip()
        elm = find_element(app, name, "ElmPvsys")
        if not elm:
            continue
        _set_element_attr_safe(elm, "pgini", float(row["P"]))
        _set_element_attr_safe(elm, "qgini", float(row["Q"]))
    for _, row in ES_df.iterrows():
        name = str(row["name"]).strip()
        elm = find_element(app, name, "ElmGenstat")
        if not elm:
            continue
        _set_element_attr_safe(elm, "pgini", float(row["P"]))
        _set_element_attr_safe(elm, "qgini", float(row["Q"]))
    print("Parametry (jednofazowe) zostały wczytane i ustawione w PowerFactory.")

def collect_results_snapshot(app):
    results_buses, results_lines, results_trafos, results_sys = [], [], [], []
    try:
        for bus in app.GetCalcRelevantObjects("*.ElmTerm"):
            try:
                results_buses.append({
                    "Bus": bus.loc_name,
                    "U [p.u.]": bus.GetAttribute("m:u"),
                })
            except Exception:
                continue
    except Exception:
        pass

    try:
        for line in app.GetCalcRelevantObjects("*.ElmLne"):
            try:
                results_lines.append({
                    "Line": line.loc_name,
                    "Loading [%]": line.GetAttribute("c:loading"),
                })
            except Exception:
                continue
    except Exception:
        pass

    try:
        for tr in app.GetCalcRelevantObjects("*.ElmTr2"):
            try:
                results_trafos.append({
                    "Trafo": tr.loc_name,
                    "Loading [%]": tr.GetAttribute("c:loading"),
                    "Tap": tr.GetAttribute("e:nntap"),
                })
            except Exception:
                continue
    except Exception:
        pass

    try:
        for net in app.GetCalcRelevantObjects("*.ElmNet"):
            try:
                results_sys.append({
                    "System": net.loc_name,
                    "Ploss [MW]": net.GetAttribute("c:LossP"),
                    "Pload [MW]": net.GetAttribute("c:LoadP"),
                })
            except Exception:
                continue
    except Exception:
        pass

    return results_buses, results_lines, results_trafos, results_sys

# Funkcja celu — jednofazowa, tylko npięcie i straty
def objective_function(app, vars_def, x, ldf):
    global COMPONENT_HISTORY, LARGE_PENALTY_MULTIPLIER
    LARGE_PENALTY_CAP = 1e300

    candidate = None
    prev_storage = None
    try:
        candidate, prev_storage = apply_solution_with_storage(app, vars_def, x)
        code = None
        try:
            code = ldf.Execute()
        except Exception as e:
            big_pen = min(PENALTY * float(LARGE_PENALTY_MULTIPLIER), LARGE_PENALTY_CAP)
            COMPONENT_HISTORY.append([np.nan, np.nan, big_pen, big_pen])
            if EVAL_DELAY: time.sleep(EVAL_DELAY)
            return big_pen

        if code is not None:
            try:
                code_num = int(code)
                if code_num != 0:
                    big_pen = min(PENALTY * float(LARGE_PENALTY_MULTIPLIER), LARGE_PENALTY_CAP)
                    COMPONENT_HISTORY.append([np.nan, np.nan, big_pen, big_pen])
                    if EVAL_DELAY: time.sleep(EVAL_DELAY)
                    return big_pen
            except Exception:
                pass

        if EVAL_DELAY:
            time.sleep(EVAL_DELAY)

        nets = app.GetCalcRelevantObjects("*.ElmNet")
        if not nets:
            big_pen = min(PENALTY * float(LARGE_PENALTY_MULTIPLIER), LARGE_PENALTY_CAP)
            COMPONENT_HISTORY.append([np.nan, np.nan, big_pen, big_pen])
            return big_pen
        net = nets[0]

        # --- Odchylenie napięcia od celu (np. 1.05 p.u.)
        buses = app.GetCalcRelevantObjects("*.ElmTerm")
        node_count = 0
        sum_sq = 0.0
        for b in buses:
            try:
                u = b.GetAttribute("m:u")
                vf = float(u)
                sum_sq += ((vf - 1.05)/1.05)**2
                node_count += 1
            except Exception:
                continue
        denom = max(1, node_count)
        C1 = math.sqrt(sum_sq / denom)

        # --- Straty aktywne / obciążenie
        try:
            Ploss = float(net.GetAttribute("c:LossP"))
        except Exception:
            Ploss = 0.0
        try:
            TotalLoad = float(net.GetAttribute("c:LoadP"))
        except Exception:
            TotalLoad = 0.0
        C2 = float(Ploss / max(1e-9, abs(TotalLoad)))

            # --- Składnik 3: stosunek mocy magazynu do maksymalnej
        try:
            idx_p = next(i for i, v in enumerate(vars_def) if v["name"] == "storage_P")
            P_storage = float(x[idx_p])
            C3 = abs(P_storage) / abs(STORAGE_P_MAX)
        except Exception:
            C3 = 0.0
        
        # --- Kary za przekroczenia napięcia i obciążenia
        penalty = 0.0
        buses_snap, lines_snap, trafos_snap, _ = collect_results_snapshot(app)
        for b in buses_snap:
            try:
                u = b["U [p.u.]"]
                if u < VOLTAGE_MIN or u > VOLTAGE_MAX:
                    penalty += PENALTY
            except Exception:
                continue
        for l in lines_snap:
            try:
                if l["Loading [%]"] is not None and float(l["Loading [%]"]) > LOAD_MAX:
                    penalty += PENALTY
            except Exception:
                continue
        for t in trafos_snap:
            try:
                if t["Loading [%]"] is not None and float(t["Loading [%]"]) > LOAD_MAX:
                    penalty += PENALTY
            except Exception:
                continue

        comps = [C1, C2, C3]
        w = WEIGHTS
        total_obj = float(np.dot(w, np.array(comps, dtype=float)) + penalty)
        COMPONENT_HISTORY.append([C1, C2, C3, penalty, total_obj])
        return total_obj
    finally:
        if prev_storage is not None and candidate is not None:
            try:
                reset_storage_on_node(app, candidate, prev_storage)
            except Exception:
                pass

def extract_node_num(bus_name):
    # Przykład: bus_name = 'Bus10' lub '10'
    try:
        # Czy jest format "BusN"?
        if isinstance(bus_name, str) and bus_name.startswith("Bus"):
            return int(bus_name.replace("Bus", '').strip())
        # Czy sama liczba jako string?
        return int(bus_name.strip())
    except Exception:
        # Jeśli nie da się sparsować, wyśrodkuj taki węzeł w końcu
        return 9999


def main():
    global STORAGE_CANDIDATES, COMPONENT_HISTORY
    if powerfactory is None:
        print("powerfactory package not available.")
        return

    app = powerfactory.GetApplicationExt(USER)
    app.ActivateProject(PROJECT_NAME)
    ldf = app.GetFromStudyCase("ComLdf")

    vars_def = load_vars(EXCEL_FILE)
    print(f"Wczytano {len(vars_def)} zmiennych do optymalizacji.")

    STORAGE_CANDIDATES = load_storage_candidates(EXCEL_FILE)
    K = len(STORAGE_CANDIDATES)
    if K > 0:
        vars_def.append({
            "name": "storage_node_index",
            "pf_class": "choice",
            "attr": "",
            "min": 0.0,
            "max": float(max(0, K - 1))
        })
        vars_def += [
            {"name": "storage_P", "pf_class": "storage", "attr": "", "min": STORAGE_P_MIN, "max": STORAGE_P_MAX},
            {"name": "storage_Q", "pf_class": "storage", "attr": "", "min": STORAGE_Q_MIN, "max": STORAGE_Q_MAX},
        ]
        print(f"Dodano zmienne magazynu (kandydatów: {K}) do vars_def.")

    load_and_set_elements_from_excel(app, EXCEL_FILE)

    try: ldf.Execute()
    except Exception: pass
    buses_before, lines, trafos, sys_before = collect_results_snapshot(app)
    df_buses_before = pd.DataFrame(buses_before)
    df_lines = pd.DataFrame(lines)
    df_trafos = pd.DataFrame(trafos)
    df_sys = pd.DataFrame(sys_before)
    with pd.ExcelWriter(START_FILE, engine="openpyxl") as writer:
        df_buses_before.to_excel(writer, sheet_name="Buses_Before", index=False)
        df_lines.to_excel(writer, sheet_name="Lines", index=False)
        df_trafos.to_excel(writer, sheet_name="Transformers", index=False)
        df_sys.to_excel(writer, sheet_name="System_Before", index=False)
    print(f"Wyniki początkowe zapisane do {START_FILE}")

    Dim = len(vars_def)
    Lb = np.array([v["min"] for v in vars_def])
    Ub = np.array([v["max"] for v in vars_def])

    def obj(x): return objective_function(app, vars_def, x, ldf)

    time_start = time.time()
    METHOD_U = METHOD.upper()
    print(f"Uruchamiam metodę: {METHOD_U}")
    res = None
    if METHOD_U == "PSO":
        pso = PSO(obj, N_PARTICLES, Dim, Lb, Ub, N_ITER, W, C1, C2,
                  autosave_every_iters=5, autosave_path="pso_checkpoint.npz", eval_delay=EVAL_DELAY)
        res = pso.optimize()
    elif METHOD_U == "CEO":
        max_fes = N_ITER * N * N_PARTICLES
        ceo = CEO(obj, N_PARTICLES, Dim, Lb, Ub, N, max_fes)
        res = ceo.optimize()
    elif METHOD_U == "PO":
        po = PO(obj, N_PARTICLES, Dim, Lb, Ub, N_ITER,
                autosave_every_iters=5, autosave_path="po_checkpoint.npz", eval_delay=EVAL_DELAY)
        res = po.optimize()
    elif METHOD_U == "KEO":
        keo = KEO(obj, N_PARTICLES, Dim, Lb, Ub, N_ITER,
                  autosave_every_iters=5, autosave_path="keo_checkpoint.npz", eval_delay=EVAL_DELAY)
        res = keo.optimize()
    elif METHOD_U == "ECO":
        eco = ECO(obj, N_PARTICLES, Dim, Lb, Ub, N_ITER,
                  autosave_every_iters=5, autosave_path="eco_checkpoint.npz", eval_delay=EVAL_DELAY)
        res = eco.optimize()
        
        
        
        
        
    elif METHOD_U == "AIG_ACNC":
        aig_acnc = AIG_ACNC(obj, N_PARTICLES, Dim, Lb, Ub, N_ITER,
                  autosave_every_iters=5, autosave_path="aig_acnc_checkpoint.npz", eval_delay=EVAL_DELAY)
        res = aig_acnc.optimize()    

    time_end = time.time()
    print("running time:{:.5f}".format(time_end - time_start))
    try: app.PrintError("running time:{:.5f}".format(time_end - time_start))
    except Exception: pass

    if res is None:
        print("No result returned by optimizer.")
        return

    lista = res.get("best_per_iter", [])
    with open("lista.txt", "w", encoding="utf-8") as f:
        for el in lista:
            f.write(f"{el}\n")

    try:
        candidate, prev = apply_solution_with_storage(app, vars_def, res.get("gbest"))
    except Exception:
        candidate = None
        prev = None
        try:
            apply_solution(app, vars_def, res.get("gbest"))
        except Exception:
            pass

    try:
        ldf.Execute()
    except Exception:
        pass
    buses_after, lines, trafos, sys_after = collect_results_snapshot(app)

    df_vars = pd.DataFrame(columns=["variable", "value", "pf_class", "attr"])
    try:
        if res.get("gbest") is not None:
            rows = []
            gbest = res["gbest"]
            for idx, var in enumerate(vars_def):
                name = var.get("name", "")
                try: val = float(gbest[idx])
                except Exception: val = gbest[idx] if idx < len(gbest) else None
                rows.append({
                    "variable": name, "value": val,
                    "pf_class": var.get("pf_class", ""), "attr": var.get("attr", "")
                })
            df_vars = pd.DataFrame(rows)
    except Exception:
        df_vars = pd.DataFrame(columns=["variable", "value", "pf_class", "attr"])

    df_storage_elems = pd.DataFrame(columns=["elem_name", "pgini", "qgini"])
    try:
        if candidate is not None:
            ename = candidate.get("elem")
            elm = find_element(app, ename, "ElmGenstat") or find_element(app, ename, "ElmSym") \
                or find_element(app, ename, "ElmPvsys") or find_element(app, ename, "ElmLod")
            if elm:
                df_storage_elems = pd.DataFrame([{
                    "elem_name": ename,
                    "pgini": elm.GetAttribute("pgini"),
                    "qgini": elm.GetAttribute("qgini"),
                }])
    except Exception:
        df_storage_elems = pd.DataFrame(columns=df_storage_elems.columns)

    try:
        if candidate is not None:
            reset_storage_on_node(app, candidate, prev if prev is not None else {})
    except Exception:
        pass

    df_buses_after = pd.DataFrame(buses_after)
    df_lines = pd.DataFrame(lines)
    df_trafos = pd.DataFrame(trafos)
    df_sys = pd.DataFrame(sys_after)
    df_eval = pd.DataFrame(res.get("best_per_iter", []), columns=["Best_per_iter"]) if res.get("best_per_iter") else pd.DataFrame()
    try:
        if COMPONENT_HISTORY:
            cols = ["C1_Udev", "C2_Ploss_norm", "C3_Pstor_norm", "penalty", "total_obj"]
            df_comp_hist = pd.DataFrame(COMPONENT_HISTORY, columns=cols)
        else:
            df_comp_hist = pd.DataFrame(columns=cols)
    except Exception:
        cols = ["C1_Udev", "C2_Ploss_norm", "C3_Pstor_norm", "penalty", "total_obj"]
        df_comp_hist = pd.DataFrame(columns=cols)

    best_components = None
    try:
        if res.get("gbest") is not None:
            _ = objective_function(app, vars_def, res["gbest"], ldf)
            if COMPONENT_HISTORY:
                last = COMPONENT_HISTORY[-1]
                best_components = dict(zip(["C1", "C2", "C3","penalty", "total"], last))
    except Exception:
        best_components = None

    with pd.ExcelWriter(OUT_FILE, engine="openpyxl") as writer:
        df_buses_after.to_excel(writer, sheet_name="Buses_After", index=False)
        df_lines.to_excel(writer, sheet_name="Lines", index=False)
        df_trafos.to_excel(writer, sheet_name="Transformers", index=False)
        df_sys.to_excel(writer, sheet_name="System_After", index=False)
        df_eval.to_excel(writer, sheet_name="ObjectiveHistory", index=False)
        if not df_vars.empty:
            df_vars.to_excel(writer, sheet_name="BestSolutionVars", index=False)
        if not df_storage_elems.empty:
            df_storage_elems.to_excel(writer, sheet_name="BestStorageElements", index=False)
        try:
            df_comp_hist.to_excel(writer, sheet_name="ComponentsHistory", index=False)
        except Exception:
            pass
        if best_components is not None:
            try:
                df_best_comp = pd.DataFrame([best_components])
                df_best_comp.to_excel(writer, sheet_name="BestComponents", index=False)
            except Exception:
                pass

    print(f"Wyniki zapisane do {OUT_FILE}")

    try:
        if res.get("best_per_iter"):
            plt.figure(figsize=(8, 5))
            plt.plot(res["best_per_iter"], marker="o", linewidth=2)
            plt.title("Przebieg optymalizacji (funkcja celu)")
            plt.xlabel("Iteracja")
            plt.ylabel("Funkcja celu")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(os.path.dirname(OUT_FILE), "Opt_Funkcja_celu.png"), dpi=300)
            plt.show()
    except Exception:
        pass
    
    
    
    
    # Przygotuj [nr_wezla, napiecie] dla obu przypadków
    u_before_pairs = [
        (extract_node_num(b["Bus"]), b["U [p.u.]"])
        for b in buses_before if b.get("U [p.u.]", 0) > 0
    ]
    u_after_pairs = [
        (extract_node_num(b["Bus"]), b["U [p.u.]"])
        for b in buses_after if b.get("U [p.u.]", 0) > 0
    ]

    # Posortuj po numerze węzła
    u_before_pairs.sort()
    u_after_pairs.sort()

    node_numbers_before, u_before_sorted = zip(*u_before_pairs) if u_before_pairs else ([], [])
    node_numbers_after, u_after_sorted = zip(*u_after_pairs) if u_after_pairs else ([], [])

    try:
        plt.figure(figsize=(10, 5))
        if u_before_pairs:
            plt.plot(node_numbers_before, u_before_sorted, marker='o', label="Napięcie przed", linestyle="-")
        if u_after_pairs:
            plt.plot(node_numbers_after, u_after_sorted, marker='x', label="Napięcie po", linestyle="--")
        plt.xlabel("Numer węzła")
        plt.ylabel("Napięcie [p.u.]")
        plt.title("Porównanie napięć węzłów przed i po optymalizacji (uporządkowane)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(OUT_FILE), "Voltage_Phases.png"), dpi=300)
        plt.show()
    except Exception:
        pass

    try:
        app = None
    except Exception:
        pass

if __name__ == "__main__":
    main()