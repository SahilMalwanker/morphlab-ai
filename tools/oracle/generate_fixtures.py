"""Generate ModernRobotics oracle fixtures for MorphLab kinematics tests.

Uses the published `modern_robotics` package (MIT, NxRLab) as an independent
reference: forward kinematics by the product of exponentials (FKinSpace) and
the space Jacobian (JacobianSpace). MorphLab's TypeScript implementation uses
a different formulation (a chain of local transforms), so agreement between
the two is an independent check.

Run:  tools/oracle/.venv/Scripts/python.exe tools/oracle/generate_fixtures.py
Output: tests/fixtures/mr_oracle.json
"""

import json
import pathlib

import modern_robotics as mr
import numpy as np

# Each chain: base pose, joints (type, origin xyz, origin rpy, axis, limits), tip offset.
CHAINS = {
    "arm3": {
        "base": {"xyz": [0.0, 0.0, 0.03], "rpy": [0.0, 0.0, 0.0]},
        "joints": [
            {"type": "revolute", "xyz": [0, 0, 0.03], "rpy": [0, 0, 0], "axis": [0, 0, 1], "lower": -2.9, "upper": 2.9},
            {"type": "revolute", "xyz": [0, 0, 0.12], "rpy": [0, 0, 0], "axis": [0, 1, 0], "lower": -1.75, "upper": 1.75},
            {"type": "revolute", "xyz": [0, 0, 0.35], "rpy": [0, 0, 0], "axis": [0, 1, 0], "lower": -2.5, "upper": 2.5},
        ],
        "tip": [0, 0, 0.3],
    },
    "skewed4": {
        "base": {"xyz": [0.1, -0.2, 0.05], "rpy": [0.0, 0.0, 0.3]},
        "joints": [
            {"type": "revolute", "xyz": [0, 0, 0.1], "rpy": [0, 0, 0], "axis": [0, 0, 1], "lower": -2.0, "upper": 2.0},
            {"type": "revolute", "xyz": [0.05, 0, 0.2], "rpy": [0.2, 0, 0], "axis": [0, 1, 0], "lower": -1.5, "upper": 1.5},
            {"type": "prismatic", "xyz": [0.3, 0, 0], "rpy": [0, 0, 0], "axis": [1, 0, 0], "lower": 0.0, "upper": 0.2},
            {"type": "revolute", "xyz": [0.1, 0.05, 0], "rpy": [0, 0.3, -0.4], "axis": [0, 0.6, 0.8], "lower": -2.2, "upper": 2.2},
        ],
        "tip": [0, 0, 0.15],
    },
    "chain6": {
        "base": {"xyz": [0.0, 0.0, 0.03], "rpy": [0.0, 0.0, 0.0]},
        "joints": [
            {"type": "revolute", "xyz": [0, 0, 0.03], "rpy": [0, 0, 0], "axis": [0, 0, 1], "lower": -2.6, "upper": 2.6},
            {"type": "revolute", "xyz": [0, 0, 0.2], "rpy": [0, 0, 0], "axis": [0, 1, 0], "lower": -2.6, "upper": 2.6},
            {"type": "revolute", "xyz": [0, 0, 0.3], "rpy": [0, 0, 0], "axis": [0, 1, 0], "lower": -2.6, "upper": 2.6},
            {"type": "revolute", "xyz": [0, 0, 0.25], "rpy": [0, 0, 0], "axis": [1, 0, 0], "lower": -2.6, "upper": 2.6},
            {"type": "revolute", "xyz": [0, 0, 0.1], "rpy": [0, 0, 0], "axis": [0, 1, 0], "lower": -2.6, "upper": 2.6},
            {"type": "revolute", "xyz": [0, 0, 0.08], "rpy": [0, 0, 0], "axis": [1, 0, 0], "lower": -2.6, "upper": 2.6},
        ],
        "tip": [0, 0, 0.06],
    },
}


def rot(axis, angle):
    """Rotation matrix about a unit axis using the library's matrix exponential."""
    return mr.MatrixExp3(mr.VecToso3(np.array(axis, dtype=float) * angle))


def pose(xyz, rpy):
    """URDF roll-pitch-yaw: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = rpy
    R = rot([0, 0, 1], y) @ rot([0, 1, 0], p) @ rot([1, 0, 0], r)
    return mr.RpToTrans(R, np.array(xyz, dtype=float))


def screw_list_and_home(chain):
    T = pose(chain["base"]["xyz"], chain["base"]["rpy"])
    screws = []
    for j in chain["joints"]:
        T = T @ pose(j["xyz"], j["rpy"])
        R, p = mr.TransToRp(T)
        a = np.array(j["axis"], dtype=float)
        a = a / np.linalg.norm(a)
        w = R @ a
        if j["type"] == "prismatic":
            screws.append(np.concatenate([np.zeros(3), w]))
        else:
            screws.append(np.concatenate([w, -np.cross(w, p)]))
    M = T @ mr.RpToTrans(np.eye(3), np.array(chain["tip"], dtype=float))
    return np.array(screws).T, M


def main():
    rng = np.random.default_rng(20261003)
    out = {
        "generator": "tools/oracle/generate_fixtures.py",
        "library": "modern_robotics 1.1.1 (NxRLab, MIT)",
        "conventions": "T is row-major 4x4 (T[row][col]); positionJacobian is 3 x n row-major, tip linear velocity in world",
        "chains": CHAINS,
        "cases": [],
    }
    for name, chain in CHAINS.items():
        Slist, M = screw_list_and_home(chain)
        n = len(chain["joints"])
        configs = [np.zeros(n)]
        for _ in range(12):
            configs.append(np.array([rng.uniform(j["lower"], j["upper"]) for j in chain["joints"]]))
        for theta in configs:
            T = mr.FKinSpace(M, Slist, theta)
            Js = mr.JacobianSpace(Slist, theta)
            p = T[:3, 3]
            Jpos = Js[3:6, :] + np.cross(Js[0:3, :].T, p).T
            out["cases"].append({
                "chain": name,
                "q": theta.tolist(),
                "T": T.tolist(),
                "positionJacobian": Jpos.tolist(),
            })
    path = pathlib.Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "mr_oracle.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {len(out['cases'])} cases to {path}")


if __name__ == "__main__":
    main()
