#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nea_w_dodecahedron_cascade_v2.py

Same method as nea_w_icosahedron_dirac_cascade.py, but on DODECAHEDRON.
Previous dodecahedron script gave d_spec ~ 0.85, but it squared the
eigenvalues (used D^2 spectrum). This script uses D directly, matching
the icosahedron cascade methodology.
"""

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def build_dodecahedron():
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    inv_phi = 1.0 / phi
    verts = np.array([
        [ 1,  1,  1], [ 1,  1, -1], [ 1, -1,  1], [ 1, -1, -1],
        [-1,  1,  1], [-1,  1, -1], [-1, -1,  1], [-1, -1, -1],
        [ 0,  inv_phi,  phi], [ 0,  inv_phi, -phi],
        [ 0, -inv_phi,  phi], [ 0, -inv_phi, -phi],
        [ inv_phi,  phi, 0], [ inv_phi, -phi, 0],
        [-inv_phi,  phi, 0], [-inv_phi, -phi, 0],
        [ phi, 0,  inv_phi], [ phi, 0, -inv_phi],
        [-phi, 0,  inv_phi], [-phi, 0, -inv_phi],
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])
    return verts


def build_pentagonal_faces(verts):
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    inv_phi = 1.0 / phi
    face_dirs = np.array([
        [0,  1,  phi], [0,  1, -phi], [0, -1,  phi], [0, -1, -phi],
        [1,  phi, 0], [1, -phi, 0], [-1,  phi, 0], [-1, -phi, 0],
        [phi, 0,  1], [phi, 0, -1], [-phi, 0,  1], [-phi, 0, -1],
    ], dtype=float)
    face_dirs /= np.linalg.norm(face_dirs[0])

    faces = []
    for n in face_dirs:
        dots = verts @ n
        idx = np.argsort(dots)[-5:]
        u = verts[idx[0]] - np.dot(verts[idx[0]], n) * n
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        angles = []
        for i in idx:
            w = verts[i] - np.dot(verts[i], n) * n
            angles.append(np.arctan2(np.dot(w, v), np.dot(w, u)))
        order = np.argsort(angles)
        faces.append([int(idx[i]) for i in order])
    return np.array(faces)


def triangulate_pentagons(verts, pentagons):
    new_verts = list(verts)
    triangles = []
    for face in pentagons:
        center = np.mean(verts[face], axis=0)
        center /= np.linalg.norm(center)
        c_idx = len(new_verts)
        new_verts.append(center)
        for i in range(5):
            triangles.append([c_idx, int(face[i]), int(face[(i + 1) % 5])])
    return np.array(new_verts), np.array(triangles)


def subdivide(verts, faces):
    edge_mid = {}
    new_verts = list(verts)
    new_faces = []

    def get_mid(i, j):
        key = (min(i, j), max(i, j))
        if key not in edge_mid:
            m = (new_verts[i] + new_verts[j]) / 2.0
            m /= np.linalg.norm(m)
            edge_mid[key] = len(new_verts)
            new_verts.append(m)
        return edge_mid[key]

    for f in faces:
        v0, v1, v2 = f
        m01 = get_mid(v0, v1)
        m12 = get_mid(v1, v2)
        m20 = get_mid(v2, v0)
        new_faces.extend([[v0, m01, m20], [v1, m12, m01],
                          [v2, m20, m12], [m01, m12, m20]])
    return np.array(new_verts), np.array(new_faces)


def face_data(verts, faces):
    Nf = len(faces)
    centers = np.zeros((Nf, 3))
    areas = np.zeros(Nf)
    normals = np.zeros((Nf, 3))
    u_hat = np.zeros((Nf, 3))
    v_hat = np.zeros((Nf, 3))

    for i, f in enumerate(faces):
        v0, v1, v2 = verts[f[0]], verts[f[1]], verts[f[2]]
        c = (v0 + v1 + v2) / 3.0
        centers[i] = c / np.linalg.norm(c)
        n = np.cross(v1 - v0, v2 - v0)
        areas[i] = 0.5 * np.linalg.norm(n)
        n = n / np.linalg.norm(n)
        normals[i] = n
        u = v1 - v0
        u = u / np.linalg.norm(u)
        vv = np.cross(n, u)
        u_hat[i] = u
        v_hat[i] = vv / np.linalg.norm(vv)

    return centers, areas, normals, u_hat, v_hat


def build_dirac_sparse(verts, faces):
    Nf = len(faces)
    centers, areas, normals, u_hat, v_hat = face_data(verts, faces)

    sig1 = np.array([[0, 1], [1, 0]], dtype=complex)
    sig2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sig3 = np.array([[1, 0], [0, -1]], dtype=complex)

    face_vsets = [set(f) for f in faces]
    rows, cols, data = [], [], []

    for i in range(Nf):
        for j in range(i + 1, Nf):
            common = list(face_vsets[i] & face_vsets[j])
            if len(common) != 2:
                continue
            edge_vec = verts[common[1]] - verts[common[0]]
            edge_len = np.linalg.norm(edge_vec)
            d_ij = centers[j] - centers[i]
            proj = d_ij - np.dot(d_ij, normals[i]) * normals[i]
            nn = np.linalg.norm(proj)
            if nn < 1e-12:
                continue
            n3 = proj / nn
            n2 = np.array([np.dot(n3, u_hat[i]), np.dot(n3, v_hat[i])])
            n2 /= np.linalg.norm(n2)
            e_i = np.array([np.dot(edge_vec, u_hat[i]), np.dot(edge_vec, v_hat[i])])
            e_j = np.array([np.dot(edge_vec, u_hat[j]), np.dot(edge_vec, v_hat[j])])
            theta = np.arctan2(e_i[1], e_i[0]) - np.arctan2(e_j[1], e_j[0])
            U = la.expm(1j * (theta / 2.0) * sig3)
            n_sig = n2[0] * sig1 + n2[1] * sig2
            coeff = -1j * edge_len / (2.0 * np.sqrt(areas[i] * areas[j]))
            blk = coeff * (n_sig @ U)
            for r in range(2):
                for c in range(2):
                    rows.append(2*i + r); cols.append(2*j + c); data.append(blk[r, c])
                    rows.append(2*j + c); cols.append(2*i + r); data.append(np.conj(blk[r, c]))

    dim = 2 * Nf
    D = sp.coo_matrix((data, (rows, cols)), shape=(dim, dim)).tocsr()
    D = 0.5 * (D + D.conj().T)
    return D


def weyl_slope_abs(evals, fit_lo=0.10, fit_hi=0.60):
    ev = np.abs(evals)
    ev = np.sort(ev)
    ev = ev[ev > 1e-10 * np.max(ev)]
    n = len(ev)
    if n < 40:
        return float("nan")
    i0, i1 = int(fit_lo * n), int(fit_hi * n)
    s, _ = np.polyfit(np.log(ev[i0:i1]), np.log(np.arange(1, n+1)[i0:i1]), 1)
    return s


def main():
    print("=" * 92)
    print("  E's face-centered Dirac on DODECAHEDRON subdivision")
    print("  Same method as icosahedron cascade")
    print("=" * 92)
    print()
    print(f"  {'n_sub':>6s}  {'N_faces':>8s}  {'dim':>6s}  "
          f"{'d_spec':>10s}  {'first nonzero |evals|':>30s}")
    print("-" * 92)

    verts = build_dodecahedron()
    pentagons = build_pentagonal_faces(verts)
    verts, faces = triangulate_pentagons(verts, pentagons)

    for n_sub in range(0, 4):
        if n_sub > 0:
            verts, faces = subdivide(verts, faces)
        Nf = len(faces)
        dim = 2 * Nf

        D = build_dirac_sparse(verts, faces)
        if dim > 4000:
            k = min(dim - 2, 2000)
            try:
                ev = spla.eigsh(D, k=k, sigma=-0.1, which='LM',
                                return_eigenvectors=False)
            except Exception as e:
                print(f"  {n_sub:6d}  {Nf:8d}  {dim:6d}  eigsh failed")
                continue
        else:
            ev = la.eigvalsh(D.toarray())

        s = weyl_slope_abs(ev)
        ds = 2.0 * s if not np.isnan(s) else float("nan")

        ev_abs = np.sort(np.abs(ev))
        ev_nz = ev_abs[ev_abs > 1e-10]
        first_evals_str = "  ".join(f"{x:.4f}" for x in ev_nz[:6])

        print(f"  {n_sub:6d}  {Nf:8d}  {dim:6d}  {ds:10.4f}  {first_evals_str:>30s}")

    print()
    print("=" * 92)
    print("  Reference:")
    print("    Icosahedron cascade: d_spec -> 2.08 (at N_faces = 1280)")
    print("    Old dodecahedron (D^2 method): d_spec -> 0.85")
    print("=" * 92)


if __name__ == "__main__":
    main()