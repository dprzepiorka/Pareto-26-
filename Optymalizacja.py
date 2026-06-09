"""
Skrypt wykonuje pojedyncze uruchomienie optymalizacji PSO dla zadanego zestawu wag
(w1, w2, w3). Funkcja celu obejmuje:
C1 - odchylenie napięć od U_REF,
C2 - względne straty mocy czynnej,
C3 - niewykorzystanie dostępnej mocy PV.

Skrypt nie wykonuje pętli po wielu zestawach wag.
Pętla po wielu kombinacjach wag powinna być realizowana w osobnym skrypcie nadrzędnym.
"""

import sys
import os
import time
import math
import random
import argparse
import traceback

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    plt = None
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Plot generation disabled.")

# Scieżka do PowerFactory (dopasuj do instalacji)
sys.path.append(r"C:\Program Files\DIgSILENT\PowerFactory 2024 SP5A\Python\3.12")
try:
    import powerfactory
except Exception:
    powerfactory = None
    print("Warning: powerfactory module not available (script still importable for offline tests).")


# -------------------------
# Wbudowany PSO
# -------------------------
class PSO:
    """
    PSO optimizer.
    Returns dict: {"gbest": ..., "gbest_val": ..., "best_per_iter": [...]}
    """
    def __init__(
        self,
        func,
        n_particles,
        dim,
        lb,
        ub,
        max_iter,
        w=0.7,
        c1=1.5,
        c2=1.5,
        autosave_every_iters=0,
        autosave_path="pso_checkpoint.npz",
        eval_delay=0.0
    ):
        self.func = func
        self.n_particles = int(n_particles)
        self.dim = int(dim)
        self.lb = np.array(lb, dtype=float)
        self.ub = np.array(ub, dtype=float)
        self.max_iter = int(max_iter)
        self.w = float(w)
        self.c1 = float(c1)
        self.c2 = float(c2)

        self.autosave_every_iters = int(autosave_every_iters)
        self.autosave_path = autosave_path
        self.eval_delay = float(eval_delay)

        self.X = np.random.uniform(self.lb, self.ub, (self.n_particles, self.dim))
        self.V = np.zeros_like(self.X)
        self.pbest = self.X.copy()
        self.pbest_val = np.full(self.n_particles, np.inf)
        self.gbest = None
        self.gbest_val = np.inf
        self.best_per_iter = []
        self.iter = 0

    def _eval_particle(self, x, p_idx=None):
        try:
            val = float(self.func(x))
        except Exception as e:
            try:
                with open("failed_evals_pso.csv", "a", encoding="utf-8") as f:
                    f.write(f"{time.time()},{','.join(map(str, x))},exception,{str(e)}\n")
            except Exception:
                pass
            val = np.inf
        if self.eval_delay:
            time.sleep(self.eval_delay)
        return val

    def save_checkpoint(self, path=None):
        if path is None:
            path = self.autosave_path
        try:
            np.savez(
                path,
                X=self.X,
                V=self.V,
                pbest=self.pbest,
                pbest_val=self.pbest_val,
                gbest=self.gbest,
                gbest_val=self.gbest_val,
                best_per_iter=np.array(self.best_per_iter),
                iter=self.iter,
                lb=self.lb,
                ub=self.ub,
            )
            with open(os.path.splitext(path)[0] + "_history.txt", "w", encoding="utf-8") as f:
                for el in self.best_per_iter:
                    f.write(f"{el}\n")
        except Exception as e:
            try:
                with open("pso_save_error.log", "a", encoding="utf-8") as f:
                    f.write(f"{time.time()} save error: {e}\n")
                    f.write(traceback.format_exc())
            except Exception:
                pass

    def optimize(self):
        try:
            for p in range(self.n_particles):
                val = self._eval_particle(self.X[p], p)
                self.pbest_val[p] = val
                if val < self.gbest_val:
                    self.gbest_val = val
                    self.gbest = self.X[p].copy()

            self.best_per_iter.append(self.gbest_val)

            for it in range(1, self.max_iter + 1):
                self.iter = it
                for p in range(self.n_particles):
                    r1 = np.random.rand(self.dim)
                    r2 = np.random.rand(self.dim)

                    self.V[p] = (
                        self.w * self.V[p]
                        + self.c1 * r1 * (self.pbest[p] - self.X[p])
                        + self.c2 * r2 * (self.gbest - self.X[p])
                    )

                    vmax = (self.ub - self.lb) * 0.5
                    self.V[p] = np.clip(self.V[p], -vmax, vmax)
                    self.X[p] = np.clip(self.X[p] + self.V[p], self.lb, self.ub)

                    val = self._eval_particle(self.X[p], p)

                    if val < self.pbest_val[p]:
                        self.pbest_val[p] = val
                        self.pbest[p] = self.X[p].copy()

                    if val < self.gbest_val:
                        self.gbest_val = val
                        self.gbest = self.X[p].copy()

                self.best_per_iter.append(self.gbest_val)

                if self.autosave_every_iters and (it % self.autosave_every_iters == 0):
                    self.save_checkpoint()

            if self.autosave_every_iters:
                self.save_checkpoint()

            return {
                "gbest": self.gbest,
                "gbest_val": self.gbest_val,
                "best_per_iter": self.best_per_iter,
            }

        except Exception:
            try:
                with open("pso_exception.log", "a", encoding="utf-8") as f:
                    f.write(traceback.format_exc())
            except Exception:
                pass
            try:
                self.save_checkpoint(self.autosave_path.replace(".npz", "_onexception.npz"))
            except Exception:
                pass
            raise


# -------------------------
# KONFIGURACJA DOMYŚLNA
# -------------------------
EXCEL_FILE = r"N:\ksiz\STUDIA_DOKTORANCKIE\PRZEPIORKA\Artykuły\Pareto\dane.xlsx"
PROJECT_NAME = "IEEE69-SN"
OUT_DIR_DEFAULT = r"N:\ksiz\STUDIA_DOKTORANCKIE\PRZEPIORKA\Artykuły\Pareto"
USER = "minik"

METHOD = "PSO"
OBJECTIVE = "VoltageTarget"

# Parametry robocze na pierwszy etap
N_ITER_DEFAULT = 10
N_PARTICLES_DEFAULT = 3
W = 0.7
C1_PSO = 1.5
C2_PSO = 1.5

PENALTY = 1e6
LARGE_PENALTY_MULTIPLIER = 1e4
RANDOM_SEED_DEFAULT = 42

EVAL_DELAY = 0.01

# Limity magazynów energii
STORAGE_P_MIN = -7.0
STORAGE_P_MAX = 7.0
STORAGE_Q_MIN = -0.001
STORAGE_Q_MAX = 0.001

VOLTAGE_MIN = 0.9
VOLTAGE_MAX = 1.1
LOAD_MAX = 100.0  # [%]
U_REF = 1.0

# Domyślne pojedyncze wagi
W1_DEFAULT = 0.4
W2_DEFAULT = 0.3
W3_DEFAULT = 0.3

WEIGHTS = np.array([W1_DEFAULT, W2_DEFAULT, W3_DEFAULT], dtype=float)

COMPONENT_HISTORY = []
STORAGE_CANDIDATES = []


# -------------------------
# Argumenty
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Single-run PSO worker for PowerFactory optimization.")
    parser.add_argument("--w1", type=float, default=W1_DEFAULT, help="Waga składnika napięciowego")
    parser.add_argument("--w2", type=float, default=W2_DEFAULT, help="Waga składnika strat")
    parser.add_argument("--w3", type=float, default=W3_DEFAULT, help="Waga składnika niewykorzystania PV")
    parser.add_argument("--case_id", type=int, default=1, help="Numer przypadku wag")
    parser.add_argument("--run_id", type=int, default=1, help="Numer uruchomienia")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED_DEFAULT, help="Ziarno losowości")
    parser.add_argument("--n_iter", type=int, default=N_ITER_DEFAULT, help="Liczba iteracji PSO")
    parser.add_argument("--n_particles", type=int, default=N_PARTICLES_DEFAULT, help="Liczba cząstek PSO")
    parser.add_argument("--out_file", type=str, default="", help="Opcjonalna pełna ścieżka pliku wynikowego")
    parser.add_argument("--excel_file", type=str, default=EXCEL_FILE, help="Ścieżka do pliku dane.xlsx")
    parser.add_argument("--project_name", type=str, default=PROJECT_NAME, help="Nazwa projektu PowerFactory")
    parser.add_argument("--user", type=str, default=USER, help="Użytkownik PowerFactory")
    return parser.parse_args()


def validate_weights(w1, w2, w3):
    s = float(w1 + w2 + w3)
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"Suma wag musi wynosić 1.0, a wynosi {s:.12f}")


def format_weight_for_filename(value):
    s = f"{value:.6f}".rstrip("0").rstrip(".")
    return s.replace("-", "m").replace(".", "p")


def build_output_file(out_file, case_id, run_id, w1, w2, w3):
    if out_file:
        return out_file

    name = (
        f"wyniki_case{int(case_id):03d}_run{int(run_id):02d}"
        f"_w1_{format_weight_for_filename(w1)}"
        f"_w2_{format_weight_for_filename(w2)}"
        f"_w3_{format_weight_for_filename(w3)}.xlsx"
    )
    return os.path.join(OUT_DIR_DEFAULT, name)


def ensure_parent_dir(path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


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
                "elem": str(r.get("elem", "")).strip()
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
    elm = (
        find_element(app, ename, "ElmGenstat")
        or find_element(app, ename, "ElmSym")
        or find_element(app, ename, "ElmPvsys")
        or find_element(app, ename, "ElmLod")
    )
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
    elm = (
        find_element(app, ename, "ElmGenstat")
        or find_element(app, ename, "ElmSym")
        or find_element(app, ename, "ElmPvsys")
        or find_element(app, ename, "ElmLod")
    )
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
        Q = float(x[idx_p + 1])
    except Exception:
        return candidate, None

    prev = apply_storage_on_node(app, candidate, P, Q)
    return candidate, prev


def load_and_set_elements_from_excel(app, file_path):
    try:
        loads_df = pd.read_excel(file_path, sheet_name="Loads")
        gens_df = pd.read_excel(file_path, sheet_name="Generators")
        pv_df = pd.read_excel(file_path, sheet_name="PV")
        es_df = pd.read_excel(file_path, sheet_name="StatGen")
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

    for _, row in es_df.iterrows():
        name = str(row["name"]).strip()
        elm = find_element(app, name, "ElmGenstat")
        if not elm:
            continue
        _set_element_attr_safe(elm, "pgini", float(row["P"]))
        _set_element_attr_safe(elm, "qgini", float(row["Q"]))

    print("Parametry (jednofazowe) zostały wczytane i ustawione w PowerFactory.")


def collect_results_snapshot(app):
    results_buses, results_lines, results_trafos, results_sys = [], [], [], []

    # BUSY
    try:
        for bus in app.GetCalcRelevantObjects("*.ElmTerm"):
            try:
                u_val = None
                ang_val = None

                try:
                    u_val = bus.GetAttribute("m:u")
                except Exception:
                    u_val = None

                try:
                    ang_val = bus.GetAttribute("m:phiu")
                except Exception:
                    try:
                        ang_val = bus.GetAttribute("m:phi")
                    except Exception:
                        ang_val = None

                results_buses.append({
                    "Bus": bus.loc_name,
                    "U [p.u.]": u_val,
                    "Angle [deg]": ang_val,
                })
            except Exception:
                continue
    except Exception:
        pass

    # LINIE
    try:
        for line in app.GetCalcRelevantObjects("*.ElmLne"):
            try:
                i_rated = None

                try:
                    i_rated = line.GetAttribute("Inom")
                except Exception:
                    try:
                        typ = line.GetAttribute("typ_id")
                        if typ is not None:
                            try:
                                i_rated = typ.GetAttribute("sline")
                            except Exception:
                                i_rated = None
                    except Exception:
                        i_rated = None

                results_lines.append({
                    "Line": line.loc_name,
                    "I_rated": i_rated,
                    "Loading [%]": line.GetAttribute("c:loading"),
                })
            except Exception:
                continue
    except Exception:
        pass

    # TRANSFORMATORY
    try:
        for tr in app.GetCalcRelevantObjects("*.ElmTr2"):
            try:
                results_trafos.append({
                    "Trafo": tr.loc_name,
                    "Loading [%]": tr.GetAttribute("c:loading"),
                })
            except Exception:
                continue
    except Exception:
        pass

    # SYSTEM
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


def merge_buses_before_after(buses_before, buses_after):
    before_map = {str(x.get("Bus", "")).strip(): x for x in buses_before}
    after_map = {str(x.get("Bus", "")).strip(): x for x in buses_after}

    all_names = sorted(set(before_map.keys()) | set(after_map.keys()), key=extract_node_num)
    rows = []

    for name in all_names:
        b0 = before_map.get(name, {})
        b1 = after_map.get(name, {})
        rows.append({
            "Bus number": name,
            "U before [p.u.]": b0.get("U [p.u.]"),
            "Angle before [deg]": b0.get("Angle [deg]"),
            "U after [p.u.]": b1.get("U [p.u.]"),
            "Angle after [deg]": b1.get("Angle [deg]"),
        })

    return pd.DataFrame(rows)


def merge_lines_before_after(lines_before, lines_after):
    before_map = {str(x.get("Line", "")).strip(): x for x in lines_before}
    after_map = {str(x.get("Line", "")).strip(): x for x in lines_after}

    all_names = sorted(set(before_map.keys()) | set(after_map.keys()))
    rows = []

    for name in all_names:
        l0 = before_map.get(name, {})
        l1 = after_map.get(name, {})
        i_rated = l0.get("I_rated")
        if i_rated is None:
            i_rated = l1.get("I_rated")

        rows.append({
            "Line name": name,
            "I rated": i_rated,
            "Loading before [%]": l0.get("Loading [%]"),
            "Loading after [%]": l1.get("Loading [%]"),
        })

    return pd.DataFrame(rows)


def merge_trafos_before_after(trafos_before, trafos_after):
    before_map = {str(x.get("Trafo", "")).strip(): x for x in trafos_before}
    after_map = {str(x.get("Trafo", "")).strip(): x for x in trafos_after}

    all_names = sorted(set(before_map.keys()) | set(after_map.keys()))
    rows = []

    for name in all_names:
        t0 = before_map.get(name, {})
        t1 = after_map.get(name, {})
        rows.append({
            "Trafo name": name,
            "Loading before [%]": t0.get("Loading [%]"),
            "Loading after [%]": t1.get("Loading [%]"),
        })

    return pd.DataFrame(rows)


def merge_system_before_after(sys_before, sys_after):
    before_map = {str(x.get("System", "")).strip(): x for x in sys_before}
    after_map = {str(x.get("System", "")).strip(): x for x in sys_after}

    all_names = sorted(set(before_map.keys()) | set(after_map.keys()))
    rows = []

    for name in all_names:
        s0 = before_map.get(name, {})
        s1 = after_map.get(name, {})
        rows.append({
            "System": name,
            "Ploss before [MW]": s0.get("Ploss [MW]"),
            "Pload before [MW]": s0.get("Pload [MW]"),
            "Ploss after [MW]": s1.get("Ploss [MW]"),
            "Pload after [MW]": s1.get("Pload [MW]"),
        })

    return pd.DataFrame(rows)


def calculate_dpv_from_vars(vars_def, x):
    pv_gen_sum = 0.0
    pv_av_sum = 0.0

    for idx, var in enumerate(vars_def):
        try:
            pf_class = str(var.get("pf_class", "")).strip()
            attr = str(var.get("attr", "")).strip()

            is_pv_p = (pf_class == "ElmPvsys" and attr == "pgini")
            if is_pv_p:
                p_av = float(var["max"])
                p_gen = float(x[idx])

                p_gen = max(0.0, min(p_gen, p_av))
                pv_gen_sum += p_gen
                pv_av_sum += max(0.0, p_av)
        except Exception:
            continue

    if pv_av_sum <= 1e-9:
        return 0.0, pv_gen_sum, pv_av_sum

    dpv = 1.0 - (pv_gen_sum / pv_av_sum)
    dpv = max(0.0, min(1.0, dpv))
    return dpv, pv_gen_sum, pv_av_sum


def objective_function(app, vars_def, x, ldf):
    global COMPONENT_HISTORY, LARGE_PENALTY_MULTIPLIER, WEIGHTS
    LARGE_PENALTY_CAP = 1e300

    candidate = None
    prev_storage = None
    try:
        candidate, prev_storage = apply_solution_with_storage(app, vars_def, x)

        code = None
        try:
            code = ldf.Execute()
        except Exception:
            big_pen = min(PENALTY * float(LARGE_PENALTY_MULTIPLIER), LARGE_PENALTY_CAP)
            COMPONENT_HISTORY.append([np.nan, np.nan, np.nan, big_pen, big_pen, np.nan, np.nan])
            if EVAL_DELAY:
                time.sleep(EVAL_DELAY)
            return big_pen

        if code is not None:
            try:
                code_num = int(code)
                if code_num != 0:
                    big_pen = min(PENALTY * float(LARGE_PENALTY_MULTIPLIER), LARGE_PENALTY_CAP)
                    COMPONENT_HISTORY.append([np.nan, np.nan, np.nan, big_pen, big_pen, np.nan, np.nan])
                    if EVAL_DELAY:
                        time.sleep(EVAL_DELAY)
                    return big_pen
            except Exception:
                pass

        if EVAL_DELAY:
            time.sleep(EVAL_DELAY)

        nets = app.GetCalcRelevantObjects("*.ElmNet")
        if not nets:
            big_pen = min(PENALTY * float(LARGE_PENALTY_MULTIPLIER), LARGE_PENALTY_CAP)
            COMPONENT_HISTORY.append([np.nan, np.nan, np.nan, big_pen, big_pen, np.nan, np.nan])
            return big_pen
        net = nets[0]

        # Udev
        buses = app.GetCalcRelevantObjects("*.ElmTerm")
        node_count = 0
        sum_sq = 0.0
        for b in buses:
            try:
                vf = float(b.GetAttribute("m:u"))
                sum_sq += ((vf - U_REF) / U_REF) ** 2
                node_count += 1
            except Exception:
                continue
        denom = max(1, node_count)
        udev = math.sqrt(sum_sq / denom)

        # dPloss
        try:
            ploss = float(net.GetAttribute("c:LossP"))
        except Exception:
            ploss = 0.0
        try:
            total_load = float(net.GetAttribute("c:LoadP"))
        except Exception:
            total_load = 0.0
        dploss = float(ploss / max(1e-9, abs(total_load)))

        # dPV
        try:
            dpv, pv_gen_sum, pv_av_sum = calculate_dpv_from_vars(vars_def, x)
        except Exception:
            dpv = 0.0
            pv_gen_sum = np.nan
            pv_av_sum = np.nan

        # Penalty
        penalty = 0.0
        buses_snap, lines_snap, trafos_snap, _ = collect_results_snapshot(app)

        for b in buses_snap:
            try:
                u = float(b["U [p.u.]"])
                if u < VOLTAGE_MIN or u > VOLTAGE_MAX:
                    penalty += PENALTY
            except Exception:
                continue

        for l in lines_snap:
            try:
                loading = l["Loading [%]"]
                if loading is not None and float(loading) > LOAD_MAX:
                    penalty += PENALTY
            except Exception:
                continue

        for t in trafos_snap:
            try:
                loading = t["Loading [%]"]
                if loading is not None and float(loading) > LOAD_MAX:
                    penalty += PENALTY
            except Exception:
                continue

        comps = np.array([udev, dploss, dpv], dtype=float)
        total_obj = float(np.dot(WEIGHTS, comps) + penalty)

        COMPONENT_HISTORY.append([udev, dploss, dpv, penalty, total_obj, pv_gen_sum, pv_av_sum])
        return total_obj

    finally:
        if prev_storage is not None and candidate is not None:
            try:
                reset_storage_on_node(app, candidate, prev_storage)
            except Exception:
                pass


def extract_node_num(bus_name):
    try:
        if isinstance(bus_name, str) and bus_name.startswith("Bus"):
            return int(bus_name.replace("Bus", "").strip())
        return int(bus_name.strip())
    except Exception:
        return 9999


def build_run_config_df(args, out_file, elapsed_time):
    global WEIGHTS
    return pd.DataFrame([{
        "project_name": args.project_name,
        "excel_file": args.excel_file,
        "out_file": out_file,
        "case_id": args.case_id,
        "run_id": args.run_id,
        "random_seed": args.seed,
        "method": METHOD,
        "objective": OBJECTIVE,
        "weights_w1": WEIGHTS[0],
        "weights_w2": WEIGHTS[1],
        "weights_w3": WEIGHTS[2],
        "u_ref": U_REF,
        "n_iter": args.n_iter,
        "n_particles": args.n_particles,
        "pso_w": W,
        "pso_c1": C1_PSO,
        "pso_c2": C2_PSO,
        "penalty_base": PENALTY,
        "large_penalty_multiplier": LARGE_PENALTY_MULTIPLIER,
        "eval_delay": EVAL_DELAY,
        "calc_time_sec": elapsed_time,
    }])


def main():
    global STORAGE_CANDIDATES, COMPONENT_HISTORY, WEIGHTS

    args = parse_args()
    validate_weights(args.w1, args.w2, args.w3)

    WEIGHTS = np.array([args.w1, args.w2, args.w3], dtype=float)

    random.seed(args.seed)
    np.random.seed(args.seed)
    COMPONENT_HISTORY = []

    out_file = build_output_file(
        out_file=args.out_file,
        case_id=args.case_id,
        run_id=args.run_id,
        w1=args.w1,
        w2=args.w2,
        w3=args.w3,
    )
    ensure_parent_dir(out_file)

    if powerfactory is None:
        print("powerfactory package not available.")
        return

    app = powerfactory.GetApplicationExt(args.user)
    app.ActivateProject(args.project_name)
    ldf = app.GetFromStudyCase("ComLdf")

    vars_def = load_vars(args.excel_file)
    print(f"Wczytano {len(vars_def)} zmiennych do optymalizacji.")

    STORAGE_CANDIDATES = load_storage_candidates(args.excel_file)
    k = len(STORAGE_CANDIDATES)
    if k > 0:
        vars_def.append({
            "name": "storage_node_index",
            "pf_class": "choice",
            "attr": "",
            "min": 0.0,
            "max": float(max(0, k - 1))
        })
        vars_def += [
            {"name": "storage_P", "pf_class": "storage", "attr": "", "min": STORAGE_P_MIN, "max": STORAGE_P_MAX},
            {"name": "storage_Q", "pf_class": "storage", "attr": "", "min": STORAGE_Q_MIN, "max": STORAGE_Q_MAX},
        ]
        print(f"Dodano zmienne magazynu (kandydatów: {k}) do vars_def.")

    load_and_set_elements_from_excel(app, args.excel_file)

    try:
        ldf.Execute()
    except Exception:
        pass

    buses_before, lines_before, trafos_before, sys_before = collect_results_snapshot(app)

    dim = len(vars_def)
    lb = np.array([v["min"] for v in vars_def], dtype=float)
    ub = np.array([v["max"] for v in vars_def], dtype=float)

    def obj(x):
        return objective_function(app, vars_def, x, ldf)

    time_start = time.time()
    print("Uruchamiam metodę: PSO")

    autosave_base = os.path.splitext(out_file)[0] + "_pso_checkpoint.npz"

    pso = PSO(
        obj,
        args.n_particles,
        dim,
        lb,
        ub,
        args.n_iter,
        W,
        C1_PSO,
        C2_PSO,
        autosave_every_iters=5,
        autosave_path=autosave_base,
        eval_delay=EVAL_DELAY
    )
    res = pso.optimize()

    time_end = time.time()
    elapsed = time_end - time_start

    print("running time:{:.5f}".format(elapsed))
    try:
        app.PrintError("running time:{:.5f}".format(elapsed))
    except Exception:
        pass

    if res is None:
        print("No result returned by optimizer.")
        return

    lista = res.get("best_per_iter", [])
    try:
        with open(os.path.splitext(out_file)[0] + "_lista.txt", "w", encoding="utf-8") as f:
            for el in lista:
                f.write(f"{el}\n")
    except Exception:
        pass

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

    buses_after, lines_after, trafos_after, sys_after = collect_results_snapshot(app)

    df_vars = pd.DataFrame(columns=["variable", "value", "pf_class", "attr"])
    try:
        if res.get("gbest") is not None:
            rows = []
            gbest = res["gbest"]
            for idx, var in enumerate(vars_def):
                name = var.get("name", "")
                try:
                    val = float(gbest[idx])
                except Exception:
                    val = gbest[idx] if idx < len(gbest) else None
                rows.append({
                    "variable": name,
                    "value": val,
                    "pf_class": var.get("pf_class", ""),
                    "attr": var.get("attr", "")
                })
            df_vars = pd.DataFrame(rows)
    except Exception:
        df_vars = pd.DataFrame(columns=["variable", "value", "pf_class", "attr"])

    df_storage_elems = pd.DataFrame(columns=["elem_name", "pgini", "qgini"])
    try:
        if candidate is not None:
            ename = candidate.get("elem")
            elm = (
                find_element(app, ename, "ElmGenstat")
                or find_element(app, ename, "ElmSym")
                or find_element(app, ename, "ElmPvsys")
                or find_element(app, ename, "ElmLod")
            )
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

    df_buses = merge_buses_before_after(buses_before, buses_after)
    df_lines = merge_lines_before_after(lines_before, lines_after)
    df_trafos = merge_trafos_before_after(trafos_before, trafos_after)
    df_system = merge_system_before_after(sys_before, sys_after)

    df_eval = (
        pd.DataFrame(res.get("best_per_iter", []), columns=["Best_per_iter"])
        if res.get("best_per_iter") else pd.DataFrame()
    )

    comp_cols = [
        "Udev",
        "dPloss",
        "dPV",
        "Penalty",
        "F_obj",
        "PV_gen_sum",
        "PV_av_sum",
    ]

    try:
        if COMPONENT_HISTORY:
            df_comp_hist = pd.DataFrame(COMPONENT_HISTORY, columns=comp_cols)
            df_comp_hist["w1"] = WEIGHTS[0]
            df_comp_hist["w2"] = WEIGHTS[1]
            df_comp_hist["w3"] = WEIGHTS[2]
            df_comp_hist["case_id"] = args.case_id
            df_comp_hist["run_id"] = args.run_id
            df_comp_hist["random_seed"] = args.seed
        else:
            df_comp_hist = pd.DataFrame(columns=comp_cols + [
                "w1", "w2", "w3", "case_id", "run_id", "random_seed"
            ])
    except Exception:
        df_comp_hist = pd.DataFrame(columns=comp_cols + [
            "w1", "w2", "w3", "case_id", "run_id", "random_seed"
        ])

    best_components = None
    try:
        if res.get("gbest") is not None:
            _ = objective_function(app, vars_def, res["gbest"], ldf)
            if COMPONENT_HISTORY:
                last = COMPONENT_HISTORY[-1]
                best_components = {
                    "w1": WEIGHTS[0],
                    "Udev": last[0],
                    "w2": WEIGHTS[1],
                    "dPloss": last[1],
                    "w3": WEIGHTS[2],
                    "dPV": last[2],
                    "Penalty": last[3],
                    "F_obj": last[4],
                    "PV_gen_sum": last[5],
                    "PV_av_sum": last[6],
                    "case_id": args.case_id,
                    "run_id": args.run_id,
                    "random_seed": args.seed,
                }
    except Exception:
        best_components = None

    df_run_config = build_run_config_df(args, out_file, elapsed)

    with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
        df_buses.to_excel(writer, sheet_name="Buses", index=False)
        df_lines.to_excel(writer, sheet_name="Lines", index=False)
        df_trafos.to_excel(writer, sheet_name="Transformers", index=False)
        df_system.to_excel(writer, sheet_name="System", index=False)
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

        try:
            df_run_config.to_excel(writer, sheet_name="RunConfig", index=False)
        except Exception:
            pass

    print(f"Wyniki zapisane do {out_file}")

    if HAS_MATPLOTLIB:
        try:
            if res.get("best_per_iter"):
                plt.figure(figsize=(8, 5))
                plt.plot(res["best_per_iter"], marker="o", linewidth=2)
                plt.title("Przebieg optymalizacji (funkcja celu)")
                plt.xlabel("Iteracja")
                plt.ylabel("Funkcja celu")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.splitext(out_file)[0] + "_Opt_Funkcja_celu.png", dpi=300)
                plt.close()
        except Exception:
            pass

    u_before_pairs = [
        (extract_node_num(b["Bus"]), b["U [p.u.]"])
        for b in buses_before if b.get("U [p.u.]", 0) not in [None, 0]
    ]
    u_after_pairs = [
        (extract_node_num(b["Bus"]), b["U [p.u.]"])
        for b in buses_after if b.get("U [p.u.]", 0) not in [None, 0]
    ]

    u_before_pairs.sort()
    u_after_pairs.sort()

    node_numbers_before, u_before_sorted = zip(*u_before_pairs) if u_before_pairs else ([], [])
    node_numbers_after, u_after_sorted = zip(*u_after_pairs) if u_after_pairs else ([], [])

    if HAS_MATPLOTLIB:
        try:
            plt.figure(figsize=(10, 5))
            if u_before_pairs:
                plt.plot(node_numbers_before, u_before_sorted, marker="o", label="Napięcie przed", linestyle="-")
            if u_after_pairs:
                plt.plot(node_numbers_after, u_after_sorted, marker="x", label="Napięcie po", linestyle="--")
            plt.xlabel("Numer węzła")
            plt.ylabel("Napięcie [p.u.]")
            plt.title("Porównanie napięć węzłów przed i po optymalizacji")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.splitext(out_file)[0] + "_Voltage_Profile.png", dpi=300)
            plt.close()
        except Exception:
            pass

    try:
        app = None
    except Exception:
        pass


if __name__ == "__main__":
    main()