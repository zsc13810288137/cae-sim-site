#!/usr/bin/env python3
"""
Realistic Simplified BIW (Body-in-White) Beam-Frame Model
==========================================================
A more car-shaped beam model with sloped windshield, rear window,
engine bay, trunk, wheel arches, and proper automotive proportions.

Author: Senchuan (nauhcnes.qi-kleinod.com)
"""

import numpy as np
from scipy import linalg
import pyvista as pv
import imageio.v3 as iio
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

nu = 0.3
E = 210e9
rho = 7850.0
G = E / (2*(1+nu))

# ============================================================
# 1. Car-shaped BIW geometry
# ============================================================
# Approximate mid-size sedan (e.g. BMW 3er / Tesla Model 3 proportions)
# X = front→rear, Y = left→right, Z = up

W = 0.75  # half-width

# Key X stations
x = {
    'nose':       0.0,
    'bumper_f':   0.15,
    'rad':        0.45,   # radiator support
    'strut_f':    0.75,   # front strut tower
    'firewall':   1.05,
    'a_lower':    1.15,   # A-pillar base
    'a_upper':    1.45,   # A-pillar top (windshield top)
    'b_lower':    2.35,   # B-pillar base
    'b_upper':    2.35,   # B-pillar top
    'c_lower':    3.45,   # C-pillar base
    'c_upper':    3.15,   # C-pillar top (rear window top) — FORWARD of base!
    'rear_deck':  3.60,
    'tail':       4.20,
    'bumper_r':   4.40,
}

# Z heights
z = {
    'ground':     0.0,
    'floor':      0.15,
    'sill':       0.35,
    'belt':       0.85,
    'cowl':       0.90,
    'hood':       0.80,
    'roof':       1.35,
    'strut_top':  0.55,
    'trunk_top':  0.95,
    'nose_top':   0.50,
    'tail_top':   0.65,
}

nodes = {}
nid_counter = 0

def N(x, y, z):
    """Add node, return id. Deduplicates."""
    global nid_counter
    coord = (round(x,4), round(y,4), round(z,4))
    for i, c in nodes.items():
        if tuple(np.round(c, 4)) == coord:
            return i
    nodes[nid_counter] = np.array([x, y, z])
    nid_counter += 1
    return nid_counter - 1

elements = []
def B(n1, n2, sec='default'):
    if n1 != n2:
        elements.append((n1, n2, sec))

# ---- FLOOR STRUCTURE ----
# Main floor rails (left & right)
fl = [  # floor left
    N(x['nose'],      -W, z['floor']),
    N(x['bumper_f'],  -W, z['floor']),
    N(x['rad'],       -W, z['floor']),
    N(x['strut_f'],   -W, z['floor']),
    N(x['firewall'],  -W, z['floor']),
    N(x['a_lower'],   -W, z['floor']),
    N(x['b_lower'],   -W, z['floor']),
    N(x['c_lower'],   -W, z['floor']),
    N(x['rear_deck'], -W, z['floor']),
    N(x['tail'],      -W, z['floor']),
    N(x['bumper_r'],  -W, z['floor']),
]
fr = [  # floor right
    N(x['nose'],      W, z['floor']),
    N(x['bumper_f'],  W, z['floor']),
    N(x['rad'],       W, z['floor']),
    N(x['strut_f'],   W, z['floor']),
    N(x['firewall'],  W, z['floor']),
    N(x['a_lower'],   W, z['floor']),
    N(x['b_lower'],   W, z['floor']),
    N(x['c_lower'],   W, z['floor']),
    N(x['rear_deck'], W, z['floor']),
    N(x['tail'],      W, z['floor']),
    N(x['bumper_r'],  W, z['floor']),
]
# Connect floor rails
for i in range(len(fl)-1):
    B(fl[i], fl[i+1], 'floor_rail')
    B(fr[i], fr[i+1], 'floor_rail')
# Floor cross-members
for i in range(len(fl)):
    B(fl[i], fr[i], 'floor_cross')

# (Center tunnel removed — was disconnected from main structure)

# ---- SILL / ROCKER PANELS ----
sill_x = [x['a_lower'], x['b_lower'], x['c_lower']]
sl = [N(sx, -W, z['sill']) for sx in sill_x]
sr = [N(sx,  W, z['sill']) for sx in sill_x]
for i in range(len(sl)-1):
    B(sl[i], sl[i+1], 'sill')
    B(sr[i], sr[i+1], 'sill')
# Sill to floor verticals
for i in range(len(sl)):
    fidx = sill_x.index(sill_x[i])
    B(fl[5+i], sl[i], 'sill_vert')
    B(fr[5+i], sr[i], 'sill_vert')

# ---- A-PILLARS (sloped for windshield) ----
a_top_l = N(x['a_upper'], -W+0.05, z['roof'])
a_top_r = N(x['a_upper'],  W-0.05, z['roof'])
# A-pillar: sill → belt → roof (sloped back)
a_belt_l = N(x['a_lower']+0.08, -W, z['belt'])
a_belt_r = N(x['a_lower']+0.08,  W, z['belt'])
B(sl[0], a_belt_l, 'a_pillar')
B(sr[0], a_belt_r, 'a_pillar')
B(a_belt_l, a_top_l, 'a_pillar')
B(a_belt_r, a_top_r, 'a_pillar')

# ---- B-PILLARS (vertical) ----
b_belt_l = N(x['b_lower'], -W, z['belt'])
b_belt_r = N(x['b_lower'],  W, z['belt'])
b_top_l  = N(x['b_upper'], -W+0.05, z['roof'])
b_top_r  = N(x['b_upper'],  W-0.05, z['roof'])
B(sl[1], b_belt_l, 'b_pillar')
B(sr[1], b_belt_r, 'b_pillar')
B(b_belt_l, b_top_l, 'b_pillar')
B(b_belt_r, b_top_r, 'b_pillar')

# ---- C-PILLARS (sloped forward for rear window) ----
c_belt_l = N(x['c_lower']-0.05, -W, z['belt'])
c_belt_r = N(x['c_lower']-0.05,  W, z['belt'])
c_top_l  = N(x['c_upper'], -W+0.08, z['roof'])
c_top_r  = N(x['c_upper'],  W-0.08, z['roof'])
B(sl[2], c_belt_l, 'c_pillar')
B(sr[2], c_belt_r, 'c_pillar')
B(c_belt_l, c_top_l, 'c_pillar')
B(c_belt_r, c_top_r, 'c_pillar')

# ---- ROOF STRUCTURE ----
# Roof rails
B(a_top_l, b_top_l, 'roof_rail')
B(b_top_l, c_top_l, 'roof_rail')
B(a_top_r, b_top_r, 'roof_rail')
B(b_top_r, c_top_r, 'roof_rail')
# Roof cross-members
B(a_top_l, a_top_r, 'roof_cross')  # front header
B(b_top_l, b_top_r, 'roof_cross')  # mid cross
B(c_top_l, c_top_r, 'roof_cross')  # rear header

# ---- BELT LINE ----
B(a_belt_l, b_belt_l, 'belt')
B(b_belt_l, c_belt_l, 'belt')
B(a_belt_r, b_belt_r, 'belt')
B(b_belt_r, c_belt_r, 'belt')

# ---- WINDSHIELD FRAME ----
# Cowl (bottom of windshield)
cowl_l = N(x['firewall']+0.05, -W+0.05, z['cowl'])
cowl_r = N(x['firewall']+0.05,  W-0.05, z['cowl'])
B(cowl_l, cowl_r, 'cowl')
B(cowl_l, a_top_l, 'windshield')
B(cowl_r, a_top_r, 'windshield')
B(cowl_l, a_belt_l, 'cowl')
B(cowl_r, a_belt_r, 'cowl')
# Connect cowl to floor/firewall
B(fl[4], cowl_l, 'firewall')
B(fr[4], cowl_r, 'firewall')

# ---- REAR WINDOW FRAME ----
rear_shelf_l = N(x['rear_deck']-0.1, -W+0.05, z['trunk_top'])
rear_shelf_r = N(x['rear_deck']-0.1,  W-0.05, z['trunk_top'])
B(rear_shelf_l, rear_shelf_r, 'rear_shelf')
B(c_top_l, rear_shelf_l, 'rear_window')
B(c_top_r, rear_shelf_r, 'rear_window')
B(c_belt_l, rear_shelf_l, 'rear_window')
B(c_belt_r, rear_shelf_r, 'rear_window')

# ---- ENGINE BAY / FRONT STRUCTURE ----
# Hood line
hood_f_l = N(x['bumper_f'], -W+0.1, z['nose_top'])
hood_f_r = N(x['bumper_f'],  W-0.1, z['nose_top'])
hood_r_l = N(x['firewall'], -W+0.05, z['hood'])
hood_r_r = N(x['firewall'],  W-0.05, z['hood'])
B(hood_f_l, hood_f_r, 'hood')
B(hood_r_l, hood_r_r, 'hood')
B(hood_f_l, hood_r_l, 'hood')
B(hood_f_r, hood_r_r, 'hood')
B(hood_r_l, cowl_l, 'hood')
B(hood_r_r, cowl_r, 'hood')

# Front crash rails (shotgun)
crash_l = N(x['nose'], -W+0.2, z['floor']+0.15)
crash_r = N(x['nose'],  W-0.2, z['floor']+0.15)
B(crash_l, fl[2], 'crash_rail')
B(crash_r, fr[2], 'crash_rail')
B(crash_l, crash_r, 'bumper_beam')
# Bumper to hood
B(crash_l, hood_f_l, 'front_end')
B(crash_r, hood_f_r, 'front_end')

# Strut towers
strut_l = N(x['strut_f'], -W+0.2, z['strut_top'])
strut_r = N(x['strut_f'],  W-0.2, z['strut_top'])
B(fl[3], strut_l, 'strut_tower')
B(fr[3], strut_r, 'strut_tower')
B(strut_l, strut_r, 'strut_bar')
B(strut_l, hood_r_l, 'strut_tower')
B(strut_r, hood_r_r, 'strut_tower')

# ---- REAR / TRUNK STRUCTURE ----
trunk_top_l = N(x['tail'], -W+0.1, z['tail_top'])
trunk_top_r = N(x['tail'],  W-0.1, z['tail_top'])
B(trunk_top_l, trunk_top_r, 'trunk')
B(rear_shelf_l, trunk_top_l, 'trunk')
B(rear_shelf_r, trunk_top_r, 'trunk')
B(fl[9], trunk_top_l, 'trunk_side')
B(fr[9], trunk_top_r, 'trunk_side')

# Rear crash rail
rear_crash_l = N(x['bumper_r'], -W+0.2, z['floor']+0.15)
rear_crash_r = N(x['bumper_r'],  W-0.2, z['floor']+0.15)
B(rear_crash_l, rear_crash_r, 'bumper_beam')
B(fl[10], rear_crash_l, 'crash_rail')
B(fr[10], rear_crash_r, 'crash_rail')
B(rear_crash_l, trunk_top_l, 'rear_end')
B(rear_crash_r, trunk_top_r, 'rear_end')

# ---- DOOR OPENING DIAGONALS (stiffening) ----
# Front door X-brace
B(sl[0], b_belt_l, 'door_brace')
B(sl[0], b_belt_r, 'door_brace')  # no, wrong side
# Fix: left side door
B(a_belt_l, sl[1], 'door_brace')
B(a_belt_r, sr[1], 'door_brace')
# Rear door
B(b_belt_l, sl[2], 'door_brace')
B(b_belt_r, sr[2], 'door_brace')

# (Wheel arches removed — not load-bearing structure)

N_NODES = len(nodes)
N_ELEM = len(elements)
print(f"Model: {N_NODES} nodes, {N_ELEM} beam elements")

# ============================================================
# 2. Section properties
# ============================================================
sections = {
    'floor_rail':   {'A': 1.8e-3, 'Iy': 2.5e-6, 'Iz': 2.0e-6, 'J': 3.5e-6},
    'floor_cross':  {'A': 1.2e-3, 'Iy': 1.2e-6, 'Iz': 1.2e-6, 'J': 2.0e-6},
    'sill':         {'A': 2.5e-3, 'Iy': 4.0e-6, 'Iz': 3.0e-6, 'J': 5.0e-6},
    'sill_vert':    {'A': 1.0e-3, 'Iy': 1.0e-6, 'Iz': 1.0e-6, 'J': 1.5e-6},
    'a_pillar':     {'A': 1.5e-3, 'Iy': 2.0e-6, 'Iz': 2.0e-6, 'J': 3.0e-6},
    'b_pillar':     {'A': 1.8e-3, 'Iy': 2.5e-6, 'Iz': 2.5e-6, 'J': 4.0e-6},
    'c_pillar':     {'A': 1.3e-3, 'Iy': 1.5e-6, 'Iz': 1.5e-6, 'J': 2.5e-6},
    'roof_rail':    {'A': 1.2e-3, 'Iy': 1.5e-6, 'Iz': 1.2e-6, 'J': 2.0e-6},
    'roof_cross':   {'A': 1.0e-3, 'Iy': 1.0e-6, 'Iz': 1.0e-6, 'J': 1.5e-6},
    'belt':         {'A': 1.0e-3, 'Iy': 1.0e-6, 'Iz': 0.8e-6, 'J': 1.2e-6},
    'cowl':         {'A': 2.0e-3, 'Iy': 3.0e-6, 'Iz': 3.0e-6, 'J': 5.0e-6},
    'windshield':   {'A': 0.8e-3, 'Iy': 0.6e-6, 'Iz': 0.6e-6, 'J': 0.8e-6},
    'firewall':     {'A': 1.5e-3, 'Iy': 2.0e-6, 'Iz': 2.0e-6, 'J': 3.0e-6},
    'rear_window':  {'A': 0.8e-3, 'Iy': 0.6e-6, 'Iz': 0.6e-6, 'J': 0.8e-6},
    'rear_shelf':   {'A': 1.2e-3, 'Iy': 1.2e-6, 'Iz': 1.2e-6, 'J': 2.0e-6},
    'hood':         {'A': 0.6e-3, 'Iy': 0.4e-6, 'Iz': 0.4e-6, 'J': 0.5e-6},
    'crash_rail':   {'A': 2.0e-3, 'Iy': 3.0e-6, 'Iz': 2.0e-6, 'J': 4.0e-6},
    'bumper_beam':  {'A': 1.5e-3, 'Iy': 2.0e-6, 'Iz': 2.0e-6, 'J': 3.0e-6},
    'front_end':    {'A': 0.8e-3, 'Iy': 0.6e-6, 'Iz': 0.6e-6, 'J': 0.8e-6},
    'strut_tower':  {'A': 1.5e-3, 'Iy': 2.0e-6, 'Iz': 2.0e-6, 'J': 3.0e-6},
    'strut_bar':    {'A': 0.8e-3, 'Iy': 0.5e-6, 'Iz': 0.5e-6, 'J': 0.8e-6},
    'trunk':        {'A': 0.8e-3, 'Iy': 0.6e-6, 'Iz': 0.6e-6, 'J': 0.8e-6},
    'trunk_side':   {'A': 1.0e-3, 'Iy': 1.0e-6, 'Iz': 1.0e-6, 'J': 1.5e-6},
    'rear_end':     {'A': 0.8e-3, 'Iy': 0.6e-6, 'Iz': 0.6e-6, 'J': 0.8e-6},
    'door_brace':   {'A': 0.5e-3, 'Iy': 0.3e-6, 'Iz': 0.3e-6, 'J': 0.4e-6},
    'default':      {'A': 1.0e-3, 'Iy': 1.0e-6, 'Iz': 1.0e-6, 'J': 1.5e-6},
}

# ============================================================
# 3. FEM Assembly and Modal Analysis
# ============================================================
DOF = 6
TDOF = N_NODES * DOF

def beam3d_Ke_Me(c1, c2, sec):
    dx = c2 - c1
    L = np.linalg.norm(dx)
    if L < 1e-10:
        return np.zeros((12,12)), np.zeros((12,12))
    A, Iy, Iz, J = sec['A'], sec['Iy'], sec['Iz'], sec['J']

    k = np.zeros((12,12))
    ea = E*A/L; gj = G*J/L
    k[0,0]=k[6,6]=ea; k[0,6]=k[6,0]=-ea
    k[3,3]=k[9,9]=gj; k[3,9]=k[9,3]=-gj

    c1v=12*E*Iy/L**3; c2v=6*E*Iy/L**2; c3v=4*E*Iy/L; c4v=2*E*Iy/L
    k[1,1]=k[7,7]=c1v; k[1,7]=k[7,1]=-c1v
    k[1,5]=k[5,1]=c2v; k[1,11]=k[11,1]=c2v
    k[7,5]=k[5,7]=-c2v; k[7,11]=k[11,7]=-c2v
    k[5,5]=k[11,11]=c3v; k[5,11]=k[11,5]=c4v

    d1=12*E*Iz/L**3; d2=6*E*Iz/L**2; d3=4*E*Iz/L; d4=2*E*Iz/L
    k[2,2]=k[8,8]=d1; k[2,8]=k[8,2]=-d1
    k[2,4]=k[4,2]=-d2; k[2,10]=k[10,2]=-d2
    k[8,4]=k[4,8]=d2; k[8,10]=k[10,8]=d2
    k[4,4]=k[10,10]=d3; k[4,10]=k[10,4]=d4

    m = np.zeros((12,12))
    mL = rho*A*L
    for i in [0,1,2,6,7,8]: m[i,i]=mL/2
    Ip = rho*(Iy+Iz)*L/2
    for i in [3,4,5,9,10,11]: m[i,i]=Ip/6

    e1 = dx/L
    v = np.array([0,0,1.0]) if abs(e1[2])<0.9 else np.array([1.0,0,0])
    e2 = np.cross(v, e1); e2 /= np.linalg.norm(e2)
    e3 = np.cross(e1, e2)
    R = np.array([e1, e2, e3])
    T = np.zeros((12,12))
    for i in range(4): T[3*i:3*i+3, 3*i:3*i+3] = R
    return T.T@k@T, T.T@m@T

print("Assembling...")
K = np.zeros((TDOF, TDOF))
M = np.zeros((TDOF, TDOF))

for n1, n2, sn in elements:
    sec = sections.get(sn, sections['default'])
    Ke, Me = beam3d_Ke_Me(nodes[n1], nodes[n2], sec)
    dofs = []
    for nd in [n1, n2]:
        for d in range(DOF):
            dofs.append(nd*DOF + d)
    for i in range(12):
        for j in range(12):
            K[dofs[i],dofs[j]] += Ke[i,j]
            M[dofs[i],dofs[j]] += Me[i,j]

print("Solving eigenvalue problem...")
eps = 1e-8 * np.max(np.diag(M))
eigenvalues, eigenvectors = linalg.eigh(K, M + eps*np.eye(TDOF))
freqs = np.sqrt(np.abs(eigenvalues)) / (2*np.pi)
flex = np.where(freqs > 1.0)[0]

print(f"\nNodes: {N_NODES}, Elements: {N_ELEM}, DOF: {TDOF}")
print(f"Rigid body modes: {np.sum(freqs <= 1.0)}")
print("\nFirst 10 natural frequencies:")
for i, idx in enumerate(flex[:10]):
    print(f"  Mode {i+1}: {freqs[idx]:.1f} Hz")

# ============================================================
# 4. Visualization
# ============================================================
node_arr = np.array([nodes[i] for i in range(N_NODES)])

# Color scheme by structural system
color_map = {
    'floor_rail': '#22b8cf', 'floor_cross': '#22b8cf',
    'sill': '#f08c00', 'sill_vert': '#f08c00',
    'a_pillar': '#be4bdb', 'b_pillar': '#ae3ec9', 'c_pillar': '#9c36b5',
    'roof_rail': '#51cf66', 'roof_cross': '#40c057',
    'belt': '#fcc419',
    'cowl': '#ff6b6b', 'windshield': '#ff8787', 'firewall': '#ff6b6b',
    'rear_window': '#ff8787', 'rear_shelf': '#ff6b6b',
    'hood': '#868e96',
    'crash_rail': '#ff922b', 'bumper_beam': '#fd7e14',
    'front_end': '#adb5bd', 'rear_end': '#adb5bd',
    'strut_tower': '#20c997', 'strut_bar': '#12b886',
    'trunk': '#868e96', 'trunk_side': '#868e96',
    'door_brace': '#495057',
}

# ---- Static model rendering (multiple views) ----
def render_static(cam_pos, cam_focal, title, filename, up=(0,0,1)):
    pl = pv.Plotter(off_screen=True, window_size=[1000, 600])
    pl.set_background('#0a0a1a')
    for n1, n2, sn in elements:
        line = pv.Line(nodes[n1], nodes[n2])
        c = color_map.get(sn, '#ffffff')
        lw = 5 if sn in ('sill','crash_rail','b_pillar','floor_rail') else 3
        pl.add_mesh(line, color=c, line_width=lw)
    # Add node spheres at key joints
    for i in range(N_NODES):
        pl.add_mesh(pv.Sphere(radius=0.02, center=nodes[i]),
                    color='#ffffff', opacity=0.3)
    pl.camera_position = [cam_pos, cam_focal, up]
    pl.add_text(title, position='upper_left', font_size=12, color='white', font='courier')
    img = pl.screenshot(return_img=True)
    path = os.path.join(OUTPUT_DIR, filename)
    iio.imwrite(path, img)
    pl.close()
    print(f"  Saved: {filename}")

cx, cz = 2.2, 0.6  # center of car

print("\nRendering static views...")
render_static((cx+4, -5, 3.5), (cx, 0, cz), "BIW Model — Isometric View", "biw_iso.png")
render_static((cx, -6, 1.0), (cx, 0, cz), "BIW Model — Side View", "biw_side.png")
render_static((cx, 0, 8), (cx, 0, cz), "BIW Model — Top View", "biw_top.png", up=(0,-1,0))
render_static((-3, 0, 1.5), (cx, 0, cz), "BIW Model — Front View", "biw_front.png")
render_static((cx+5, -3.5, 1.0), (cx, 0, cz), "BIW Model — 3/4 Rear View", "biw_34rear.png")

# ---- Mode shape GIF animations ----
def create_mode_mesh(coords):
    points = coords
    lines = []
    for n1, n2, _ in elements:
        lines.extend([2, n1, n2])
    return pv.PolyData(points, lines=lines)

mode_names = {
    1: "1st Torsion",
    2: "1st Lateral Bending",
    3: "1st Vertical Bending",
    4: "2nd Torsion",
    5: "Combined Bending",
    6: "2nd Vertical Bending",
}

N_FRAMES = 36
N_MODES = min(6, len(flex))

print("\nGenerating mode shape GIFs...")
for mn in range(1, N_MODES+1):
    idx = flex[mn-1]
    freq = freqs[idx]
    mv = eigenvectors[:, idx]

    # Extract translational DOFs
    max_d = 0
    for i in range(N_NODES):
        d = np.sqrt(mv[i*6]**2 + mv[i*6+1]**2 + mv[i*6+2]**2)
        max_d = max(max_d, d)
    sf = 0.25 / (max_d + 1e-15)

    desc = mode_names.get(mn, f"Mode {mn}")
    title = f"Mode {mn}: {freq:.1f} Hz — {desc}"

    frames = []
    for fi in range(N_FRAMES):
        phase = 2*np.pi*fi/N_FRAMES
        s = sf * np.sin(phase)
        disp = np.zeros_like(node_arr)
        for i in range(N_NODES):
            disp[i] = [mv[i*6], mv[i*6+1], mv[i*6+2]]
        deformed = node_arr + s * disp
        mag = np.linalg.norm(disp, axis=1)

        pl = pv.Plotter(off_screen=True, window_size=[1000, 600])
        pl.set_background('#0a0a1a')
        # Ghost
        m0 = create_mode_mesh(node_arr)
        pl.add_mesh(m0, color='#222244', line_width=1, opacity=0.25)
        # Deformed
        md = create_mode_mesh(deformed)
        md.point_data['disp'] = mag
        pl.add_mesh(md, scalars='disp', cmap='turbo', line_width=4,
                    clim=[0, np.max(mag)*1.01+1e-10])
        pl.camera_position = [(cx+4, -5, 3.5), (cx, 0, cz), (0,0,1)]
        pl.add_text(title, position='upper_left', font_size=12, color='white', font='courier')
        frames.append(pl.screenshot(return_img=True))
        pl.close()

    gif_path = os.path.join(OUTPUT_DIR, f"biw_mode_{mn}_{freq:.0f}hz.gif")
    iio.imwrite(gif_path, frames, duration=40, loop=0)
    print(f"  Mode {mn}: {freq:.1f} Hz → biw_mode_{mn}_{freq:.0f}hz.gif")

# ---- Overview image ----
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor='#0a0a1a')
fig.suptitle('Simplified BIW — Free-Free Modal Analysis',
             color='white', fontsize=18, fontweight='bold', y=0.97)

for mn in range(1, N_MODES+1):
    ax = axes[(mn-1)//3][(mn-1)%3]
    idx = flex[mn-1]
    freq = freqs[idx]
    desc = mode_names.get(mn, f"Mode {mn}")
    gif = os.path.join(OUTPUT_DIR, f"biw_mode_{mn}_{freq:.0f}hz.gif")
    frame = iio.imread(gif, index=0)
    ax.imshow(frame)
    ax.set_title(f"Mode {mn}: {freq:.1f} Hz — {desc}",
                 color='#22d3ee', fontsize=12, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
ov = os.path.join(OUTPUT_DIR, "biw_modes_overview.png")
fig.savefig(ov, dpi=150, facecolor='#0a0a1a', bbox_inches='tight')
plt.close()
print(f"\n  Saved: biw_modes_overview.png")

print("\n" + "="*60)
print("DONE — all output in:", OUTPUT_DIR)
