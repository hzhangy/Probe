#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nea_w_dodecahedron_diagnose.py

Diagnose the dodecahedron cascade reversal at n_sub = 3.

Distinguish:
  A. eigsh numerical failure (shift-invert on near-singular matrix)
  B. mesh degeneracy (skinny triangles from pentagon triangulation)
  C. genuine physical near-zero modes

Method:
  1. Compute all eigenvalues up to n_sub = 2 with FULL diagonalization
     (small enough).
  2. At n_sub = 2, check the eigenspectrum for spurious structure.
  3. Check triangle quality: aspect ratio, min angle, edge ratio.
  4. Compare eigsh(sigma=0.0) vs eigsh(sigma=-0.1) vs dense.
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


def mesh_quality(verts, faces):
    """Report min angle, aspect ratio distribution."""
    min_angles = []
    aspect_ratios = []
    edge_ratios = []
    for f in faces:
        v0, v1, v2 = verts[f[0]], verts[f[1]], verts[f[2]]
        # edge lengths
        a = np.linalg.norm(v1 - v2)
        b = np.linalg.norm(v0 - v2)
        c = np.linalg.norm(v0 - v1)
        edges = np.array([a, b, c])
        edge_ratios.append(edges.max() / edges.min())
        # angles via law of cosines
        A = np.arccos(np.clip((b**2 + c**2 - a**2) / (2*b*c), -1, 1))
        B = np.arccos(np.clip((a**2 + c**2 - b**2) / (2*a*c), -1, 1))
        C = np.pi - A - B
        min_angles.append(min(A, B, C))
        # aspect ratio: area / (max edge)^2, small means skinny
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        aspect_ratios.append(area / edges.max()**2)
    return (np.min(min_angles), np.median(min_angles),
            np.max(edge_ratios), np.median(edge_ratios),
            np.min(aspect_ratios), np.median(aspect_ratios))


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


def main():
    print("=" * 96)
    print("  Diagnose dodecahedron cascade reversal at n_sub = 3")
    print("=" * 96)
    print()

    verts = build_dodecahedron()
    pentagons = build_pentagonal_faces(verts)
    verts, faces = triangulate_pentagons(verts, pentagons)

    for n_sub in range(0, 4):
        if n_sub > 0:
            verts, faces = subdivide(verts, faces)
        Nf = len(faces)
        dim = 2 * Nf

        # mesh quality
        min_ang, med_ang, max_er, med_er, min_ar, med_ar = mesh_quality(verts, faces)

        print("--- n_sub = {}, N_faces = {}, dim = {} ---".format(n_sub, Nf, dim))
        print("  min triangle angle: {:.4f} rad ({:.2f} deg)".format(
            min_ang, min_ang * 180 / np.pi))
        print("  median angle:       {:.4f} rad ({:.2f} deg)".format(
            med_ang, med_ang * 180 / np.pi))
        print("  max edge ratio:     {:.4f}".format(max_er))
        print("  median edge ratio:  {:.4f}".format(med_er))
        print("  min aspect (A/L^2): {:.6e}".format(min_ar))
        print("  median aspect:      {:.6f}".format(med_ar))
        print()

        # spectrum
        D = build_dirac_sparse(verts, faces)
        if dim <= 2000:
            ev = la.eigvalsh(D.toarray())
            ev = np.sort(np.abs(ev))
            print("  FULL diag. First 10 |evals|:")
            print("   ", " ".join("{:.4e}".format(x) for x in ev[:10]))
        else:
            for sigma in [0.0, -0.1, -0.5]:
                try:
                    ev = spla.eigsh(D, k=20, sigma=sigma, which='LM',
                                    return_eigenvectors=False)
                    ev = np.sort(np.abs(ev))
                    print("  eigsh(sigma={:.2f}). First 6 |evals|:".format(sigma))
                    print("   ", " ".join("{:.4e}".format(x) for x in ev[:6]))
                except Exception as e:
                    print("  eigsh(sigma={:.2f}) failed: {}".format(sigma, e))
        print()


if __name__ == "__main__":
    main()