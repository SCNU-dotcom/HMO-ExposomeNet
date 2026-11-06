#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cardiovascular-Related Emerging Contaminants — Molecular Network Visualization
==============================================================================

Algorithm pipeline (Smer-Barreto et al., Nat. Commun. 2023):
RDKit ~200 physicochemical descriptors -> Tanimoto distance matrix
-> Louvain community detection (gamma scan) -> t-SNE embedding
-> MST + k-NN network -> D3.js force-directed interactive HTML

All categories are processed with the same unsupervised pipeline and the same
parameters; no cluster labels are assigned manually.

Outputs (per category):
- <Category>_network.html : interactive D3.js network (sphere nodes, curved
  similarity-weighted edges, cluster centers, functional-group pie charts,
  representative structures, 1000 DPI PNG / vector PDF export)
- compounds_with_clusters.csv : input table with an added Cluster column
  (cluster numbering starts from 1 within each category)

Usage:
    python Net.py --input compounds_list.xlsx --output results
"""

import os
import json
import base64
import argparse
import warnings
import traceback
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.ML.Descriptors import MoleculeDescriptors
import networkx as nx
from networkx.algorithms.community import louvain_communities

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# Global Parameters
# ═══════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_FILE = SCRIPT_DIR / "compounds_list.xlsx"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results"

RANDOM_SEED = 42
MIN_MOLECULES = 5
TANIMOTO_EDGE_THRESHOLD = 0.7
TSNE_PERPLEXITY = 50
TSNE_LEARNING_RATE = 200
TSNE_MAX_ITER = 1200
LOUVAIN_GAMMA_RANGE = (0.5, 3.0)
LOUVAIN_GAMMA_STEP = 0.1
LOUVAIN_N_RUNS = 30
KNN_EXTRA = 2
EDGE_DIST_CUTOFF = 0.7

NATURE_PALETTE = [
    "#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F",
    "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85",
    "#E4E1E3", "#F4A582", "#D6604D", "#1B7837", "#762A83",
    "#5AAE61", "#9970AB", "#E08214", "#542788", "#B35806",
]

PIE_COLORS = [
    "#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F",
    "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85",
    "#E08214", "#762A83",
]

# ═══════════════════════════════════════════════════════════════
# CVD Functional Groups (SMARTS + CVD relevance)
# ═══════════════════════════════════════════════════════════════

CVD_FUNCTIONAL_GROUPS = {
    "Quinone": {
        "smarts": "O=C1C=CC(=O)C=C1",
        "cvd_relevance": "Generates ROS via redox cycling, causing myocardial oxidative stress",
    },
    "Phenol": {
        "smarts": "c1ccc(O)cc1",
        "cvd_relevance": "Phenolic OH can undergo oxidative metabolism to form quinone intermediates",
    },
    "Aromatic amine": {
        "smarts": "c1ccc(N)cc1",
        "cvd_relevance": "Metabolic activation to reactive nitroso/hydroxylamine intermediates",
    },
    "Nitro": {
        "smarts": "[N+](=O)[O-]",
        "cvd_relevance": "In vivo reduction generates reactive nitrogen species",
    },
    "Sulfonamide": {
        "smarts": "S(=O)(=O)N",
        "cvd_relevance": "Some sulfonamides prolong QT interval",
    },
    "Tertiary amine": {
        "smarts": "[NX3;H0;!$([NX3](=O))]",
        "cvd_relevance": "Key hERG pharmacophore feature via cation-pi interaction",
    },
    "Ester": {
        "smarts": "C(=O)OC",
        "cvd_relevance": "Ester hydrolysis releases active metabolites affecting cardiovascular exposure",
    },
    "Amide": {
        "smarts": "C(=O)N",
        "cvd_relevance": "Amide bonds affect H-bond networks and target binding",
    },
    "Halide": {
        "smarts": "[CX4][F,Cl,Br,I]",
        "cvd_relevance": "Halogen substitution increases lipophilicity and potential hERG affinity",
    },
    "Organophosphate": {
        "smarts": "OP(=O)(O)O",
        "cvd_relevance": "AChE inhibition leading to cardiac arrhythmias",
    },
    "Epoxide": {
        "smarts": "C1OC1",
        "cvd_relevance": "Covalent binding to proteins/DNA triggering cardiomyocyte apoptosis",
    },
    "Carboxylic acid": {
        "smarts": "C(=O)[OH]",
        "cvd_relevance": "Affects plasma protein binding and free drug concentration",
    },
}

# ═══════════════════════════════════════════════════════════════
# RDKit ~200 physicochemical descriptors (Paper method)
# ═══════════════════════════════════════════════════════════════

def _get_rdkit_descriptor_names():
    names = [x[0] for x in Descriptors._descList]
    # Ipc spans many orders of magnitude and dominates min-max scaling
    exclude = {'Ipc'}
    names = [n for n in names if n not in exclude]
    return names

DESCRIPTOR_NAMES = _get_rdkit_descriptor_names()
_DESCRIPTOR_CALCULATOR = MoleculeDescriptors.MolecularDescriptorCalculator(DESCRIPTOR_NAMES)

def compute_rdkit_descriptors(mol):
    return list(_DESCRIPTOR_CALCULATOR.CalcDescriptors(mol))

# ═══════════════════════════════════════════════════════════════
# Tanimoto distance matrix from descriptors (Paper method)
# ═══════════════════════════════════════════════════════════════

def tanimoto_distance_matrix_from_descriptors(desc_array):
    """
    Tanimoto distance matrix on min-max scaled descriptor vectors
    (reproduces the paper method).

    For continuous variables the Tanimoto similarity is
        sim(A, B) = dot(A, B) / (|A|^2 + |B|^2 - dot(A, B))
    and distance = 1 - similarity.
    """
    scaler = MinMaxScaler()
    X = scaler.fit_transform(desc_array)
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)

    dots = X @ X.T
    norms_sq = np.diag(dots)
    denom = norms_sq[:, None] + norms_sq[None, :] - dots
    # Two all-zero vectors are treated as identical (similarity = 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(denom < 1e-12, 1.0, dots / denom)
    D = 1.0 - sim

    np.fill_diagonal(D, 0.0)
    D = np.clip(D, 0.0, None)

    return D

# ═══════════════════════════════════════════════════════════════
# Sanitize distance matrix
# ═══════════════════════════════════════════════════════════════

def sanitize_distance_matrix(D):
    """
    Ensure the distance matrix is valid for t-SNE / MST:
    1. zero diagonal
    2. all values >= 0
    3. symmetric
    4. no NaN / Inf
    """
    D = np.array(D, dtype=np.float64)
    D = np.nan_to_num(D, nan=0.0, posinf=1.0, neginf=0.0)
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    D = np.clip(D, 0.0, None)
    return D

# ═══════════════════════════════════════════════════════════════
# Tanimoto graph construction (Paper Fig.1d)
# ═══════════════════════════════════════════════════════════════

def build_tanimoto_graph(D, threshold=TANIMOTO_EDGE_THRESHOLD):
    n = D.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] < threshold:
                G.add_edge(i, j, weight=1.0 - D[i, j])
    return G

# ═══════════════════════════════════════════════════════════════
# Louvain plateau clustering (Paper Fig.1e)
# ═══════════════════════════════════════════════════════════════

def louvain_plateau_clustering(G, gamma_range, gamma_step, n_runs, seed=RANDOM_SEED):
    """
    Scan the Louvain resolution gamma, run Louvain n_runs times per gamma and
    pick the gamma in the most stable plateau of the mean cluster count K.
    Each run uses seed + run_index so runs differ but remain reproducible.
    """
    gamma_values = np.arange(gamma_range[0], gamma_range[1] + gamma_step / 2, gamma_step)
    gamma_values = np.round(gamma_values, 4)

    avg_k = []
    best_parts = []

    for gamma in gamma_values:
        k_list = []
        all_parts = []
        for run_idx in range(n_runs):
            run_seed = None if seed is None else seed + run_idx
            communities = louvain_communities(G, resolution=gamma, seed=run_seed)
            k_list.append(len(communities))
            all_parts.append(communities)
        avg_k.append(np.mean(k_list))
        # Representative partition: first run whose K equals the median K
        median_k = int(np.median(k_list))
        rep_part = None
        for p in all_parts:
            if len(p) == median_k:
                rep_part = p
                break
        if rep_part is None:
            rep_part = all_parts[0]
        best_parts.append(rep_part)

    avg_k = np.array(avg_k)
    dk = np.abs(np.diff(avg_k))

    win = 3
    if len(dk) >= win:
        smooth = np.convolve(dk, np.ones(win) / win, mode='valid')
        best_start = np.argmin(smooth)
        plateau_idx = best_start + win // 2
    else:
        plateau_idx = np.argmin(dk) if len(dk) > 0 else 0

    best_gamma = gamma_values[plateau_idx]
    communities = best_parts[plateau_idx]

    print("    Louvain gamma scan: best gamma={:.2f}, K={}".format(
        best_gamma, len(communities)))

    n = G.number_of_nodes()
    labels = np.zeros(n, dtype=int)
    # Clusters are numbered from 1, largest cluster first
    for c_idx, comm in enumerate(sorted(communities, key=len, reverse=True)):
        for node in comm:
            labels[node] = c_idx + 1

    return labels, len(communities), float(best_gamma)

def louvain_simple_fallback(D, n_nodes, seed=RANDOM_SEED):
    """Louvain on a 5-NN graph, used when the Tanimoto graph is too sparse."""
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    for i in range(n_nodes):
        dists = D[i].copy()
        dists[i] = np.inf
        knn = np.argsort(dists)[:5]
        for j in knn:
            if not G.has_edge(i, int(j)):
                G.add_edge(i, int(j), weight=max(0.01, 1.0 - D[i, int(j)]))
    communities = louvain_communities(G, resolution=1.0, seed=seed)
    labels = np.zeros(n_nodes, dtype=int)
    for c_idx, comm in enumerate(sorted(communities, key=len, reverse=True)):
        for node in comm:
            labels[node] = c_idx + 1
    return labels, len(communities)

# ═══════════════════════════════════════════════════════════════
# Find Cluster Centers
# ═══════════════════════════════════════════════════════════════

def find_cluster_centers(labels_array, dist_matrix):
    """The center of a cluster is the member with the lowest mean intra-cluster distance."""
    center_set = set()
    unique_clusters = sorted(set(labels_array))
    for cid in unique_clusters:
        members = [i for i in range(len(labels_array)) if labels_array[i] == cid]
        if len(members) == 0:
            continue
        if len(members) == 1:
            center_set.add(members[0])
            continue
        sub_dist = dist_matrix[np.ix_(members, members)]
        mean_dist = sub_dist.mean(axis=1)
        center_local_idx = np.argmin(mean_dist)
        center_global_idx = members[center_local_idx]
        center_set.add(center_global_idx)
    return center_set

# ═══════════════════════════════════════════════════════════════
# Helper: Title Case for compound names
# ═══════════════════════════════════════════════════════════════

def title_case_name(name):
    if not name:
        return name
    name = str(name).strip()
    if not name:
        return name
    words = name.split()
    result = []
    for w in words:
        if len(w) <= 1:
            result.append(w.upper())
        elif w.isupper() and len(w) <= 5:
            result.append(w)
        else:
            result.append(w[0].upper() + w[1:])
    return " ".join(result)

# ═══════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════

def build_network(dist_matrix, knn_extra=2, edge_cutoff=0.7):
    """Display network: minimum spanning tree plus up to knn_extra nearest neighbours per node."""
    n = dist_matrix.shape[0]
    csr = minimum_spanning_tree(dist_matrix)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    cx = csr.tocoo()
    for i, j, w in zip(cx.row, cx.col, cx.data):
        G.add_edge(int(i), int(j), weight=float(w))
    for i in range(n):
        dists = dist_matrix[i].copy()
        dists[i] = np.inf
        neighbors = np.argsort(dists)[:knn_extra]
        for j in neighbors:
            if dist_matrix[i][j] < edge_cutoff and not G.has_edge(i, int(j)):
                G.add_edge(i, int(j), weight=float(dist_matrix[i][j]))
    return G

def detect_functional_groups(mol):
    found = []
    for name, info in CVD_FUNCTIONAL_GROUPS.items():
        pattern = Chem.MolFromSmarts(info["smarts"])
        if pattern and mol.HasSubstructMatch(pattern):
            found.append(name)
    return found

def compute_cvd_properties(mol):
    props = {}
    for prop_name, func in [
        ("MolLogP", Descriptors.MolLogP),
        ("TPSA", Descriptors.TPSA),
        ("MolWt", Descriptors.MolWt),
        ("NumHDonors", Descriptors.NumHDonors),
        ("NumHAcceptors", Descriptors.NumHAcceptors),
        ("NumRotatableBonds", Descriptors.NumRotatableBonds),
        ("FractionCSP3", Descriptors.FractionCSP3),
        ("NumAromaticRings", Descriptors.NumAromaticRings),
        ("BertzCT", Descriptors.BertzCT),
    ]:
        try:
            val = func(mol)
            if prop_name in ("NumHDonors", "NumHAcceptors", "NumRotatableBonds", "NumAromaticRings"):
                props[prop_name] = int(val)
            elif prop_name == "FractionCSP3":
                props[prop_name] = round(val, 3)
            elif prop_name == "BertzCT":
                props[prop_name] = round(val, 1)
            else:
                props[prop_name] = round(val, 2)
        except Exception:
            props[prop_name] = None
    return props

def mol_to_base64_svg(mol, size=(220, 160)):
    try:
        drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        return base64.b64encode(svg.encode("utf-8")).decode("ascii")
    except Exception:
        return ""

def select_representative_mols(mols, valid_rows, labels, cluster_id, dist_matrix, max_rep=1):
    members = [i for i in range(len(labels)) if labels[i] == cluster_id]
    if len(members) == 0:
        return []
    sub_dist = dist_matrix[np.ix_(members, members)]
    mean_dist = sub_dist.mean(axis=1)
    sorted_idx = np.argsort(mean_dist)
    reps = []
    for rank in range(min(max_rep, len(sorted_idx))):
        i = members[sorted_idx[rank]]
        mol = mols[i]
        row = valid_rows[i]
        b64 = mol_to_base64_svg(mol, size=(220, 160))
        raw_name = str(row.get("Name", f"Compound_{i}"))
        reps.append({
            "name": title_case_name(raw_name),
            "smiles": str(row.get("SMILES", "")),
            "img": b64,
        })
    return reps

# ═══════════════════════════════════════════════════════════════
# Determine sidebar column count based on number of clusters
# ═══════════════════════════════════════════════════════════════

def get_sidebar_columns(n_clusters):
    if n_clusters <= 6:
        return 2, 560
    elif n_clusters <= 10:
        return 3, 780
    elif n_clusters <= 20:
        return 4, 960
    else:
        return 5, 1120

# ═══════════════════════════════════════════════════════════════
# D3.js HTML Generation
# ═══════════════════════════════════════════════════════════════

def _js():
    return r"""

<script src="https://d3js.org/d3.v7.min.js"></script>

<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>

<script>
(function(){
const nodes = window.__NODES__;
const links = window.__LINKS__;
const palette = window.__PALETTE__;
const clusterStats = window.__CLUSTER_STATS__;
const pieColors = window.__PIE_COLORS__;

const container = document.getElementById('network');
const width = container.clientWidth;
const height = container.clientHeight;

const svg = d3.select('#network').append('svg')
    .attr('id','main-svg')
    .attr('width', width).attr('height', height)
    .style('background','#ffffff');

const defs = svg.append('defs');

palette.forEach(function(color, idx){
    const grad = defs.append('radialGradient')
        .attr('id', 'sphere-grad-' + idx)
        .attr('cx', '30%').attr('cy', '25%').attr('r', '65%');
    grad.append('stop').attr('offset', '0%')
        .attr('stop-color', '#ffffff').attr('stop-opacity', 0.9);
    grad.append('stop').attr('offset', '25%')
        .attr('stop-color', d3.color(color).brighter(0.6));
    grad.append('stop').attr('offset', '60%')
        .attr('stop-color', color);
    grad.append('stop').attr('offset', '100%')
        .attr('stop-color', d3.color(color).darker(1.2));
});

const filter = defs.append('filter').attr('id', 'drop-shadow')
    .attr('x', '-50%').attr('y', '-50%').attr('width', '200%').attr('height', '200%');
filter.append('feGaussianBlur').attr('in', 'SourceAlpha').attr('stdDeviation', 2);
filter.append('feOffset').attr('dx', 1).attr('dy', 1).attr('result', 'offsetblur');
const fmerge = filter.append('feMerge');
fmerge.append('feMergeNode').attr('in', 'offsetblur');
fmerge.append('feMergeNode').attr('in', 'SourceGraphic');

const glowFilter = defs.append('filter').attr('id', 'center-glow')
    .attr('x', '-80%').attr('y', '-80%').attr('width', '260%').attr('height', '260%');
glowFilter.append('feGaussianBlur').attr('in', 'SourceGraphic').attr('stdDeviation', 2.5).attr('result', 'blur');
glowFilter.append('feColorMatrix').attr('in', 'blur').attr('type', 'matrix')
    .attr('values', '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.35 0').attr('result', 'glow');
const glowMerge = glowFilter.append('feMerge');
glowMerge.append('feMergeNode').attr('in', 'glow');
glowMerge.append('feMergeNode').attr('in', 'SourceGraphic');

const g = svg.append('g');
svg.call(d3.zoom().scaleExtent([0.1, 10]).on('zoom', function(event){
    g.attr('transform', event.transform);
}));

const linkG = g.append('g').attr('class', 'links');
const nodeG = g.append('g').attr('class', 'nodes');
const centerRingG = g.append('g').attr('class', 'center-rings');

function computeArcPath(d){
    const sx = d.source.x, sy = d.source.y;
    const tx = d.target.x, ty = d.target.y;
    const dx = tx - sx, dy = ty - sy;
    const dist = Math.sqrt(dx*dx + dy*dy);
    if(dist < 1) return 'M'+sx+','+sy+'L'+tx+','+ty;
    const curvature = Math.min(0.20, 0.06 + dist * 0.0002);
    const mx = (sx + tx) / 2, my = (sy + ty) / 2;
    const nx2 = -dy / dist * dist * curvature;
    const ny2 = dx / dist * dist * curvature;
    return 'M'+sx+','+sy+' Q'+(mx+nx2)+','+(my+ny2)+' '+tx+','+ty;
}

const link = linkG.selectAll('path').data(links).enter().append('path')
    .attr('fill', 'none')
    .attr('stroke', '#333333')
    .attr('stroke-opacity', function(d){ return d.sameCluster ? 0.45 : 0.2; })
    .attr('stroke-width', function(d){
        var sim = 1.0 - d.weight;
        return Math.max(0.4, 0.5 + sim * 1.0);
    });

const node = nodeG.selectAll('circle').data(nodes).enter().append('circle')
    .attr('r', function(d){ return 4 + (d.degree || 1) * 0.7; })
    .attr('fill', function(d){ return 'url(#sphere-grad-' + ((d.cluster - 1) % palette.length) + ')'; })
    .attr('stroke', function(d){ return d3.color(palette[(d.cluster - 1) % palette.length]).darker(0.5); })
    .attr('stroke-width', 0.6)
    .attr('filter', 'url(#drop-shadow)')
    .attr('cursor', 'pointer')
    .on('mouseover', function(event, d){
        d3.select(this).transition().duration(200)
            .attr('r', (4 + (d.degree||1)*0.7) * 1.6);
        if(d.isCenter){
            centerRingG.selectAll('circle').filter(function(rd){ return rd.id === d.id; })
                .transition().duration(200)
                .attr('r', ((4 + (d.degree||1)*0.7) * 1.6) + 3);
        }
        showTooltip(event, d);
    })
    .on('mouseout', function(event, d){
        d3.select(this).transition().duration(200)
            .attr('r', 4 + (d.degree||1)*0.7);
        if(d.isCenter){
            centerRingG.selectAll('circle').filter(function(rd){ return rd.id === d.id; })
                .transition().duration(200)
                .attr('r', (4 + (d.degree||1)*0.7) + 3);
        }
        hideTooltip();
    })
    .call(d3.drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended));

const centerNodes = nodes.filter(function(d){ return d.isCenter; });
const centerRing = centerRingG.selectAll('circle').data(centerNodes).enter().append('circle')
    .attr('r', function(d){ return (4 + (d.degree || 1) * 0.7) + 3; })
    .attr('fill', 'none')
    .attr('stroke', function(d){ return palette[(d.cluster - 1) % palette.length]; })
    .attr('stroke-width', 2.0)
    .attr('stroke-dasharray', '5,3')
    .attr('filter', 'url(#center-glow)')
    .attr('opacity', 0.85)
    .attr('pointer-events', 'none');

function animateDash(){
    centerRing
        .attr('stroke-dashoffset', 0)
        .transition()
        .duration(3000)
        .ease(d3.easeLinear)
        .attr('stroke-dashoffset', -16)
        .on('end', animateDash);
}
if(centerNodes.length > 0) animateDash();

const tooltip = d3.select('body').append('div')
    .style('position','absolute').style('padding','10px 14px')
    .style('background','rgba(255,255,255,0.97)').style('color','#222')
    .style('border','1px solid #ccc')
    .style('border-radius','8px').style('font-size','13px')
    .style('font-family','Arial, sans-serif')
    .style('pointer-events','none').style('opacity',0)
    .style('max-width','400px').style('line-height','1.6')
    .style('box-shadow','0 4px 16px rgba(0,0,0,0.12)')
    .style('z-index', 10000);

function showTooltip(event, d){
    let html = '<b style="font-size:15px;color:#3C5488;">' + d.name + '</b>';
    if(d.isCenter){
        html += ' <span style="color:#E64B35;font-size:12px;font-weight:bold;">★ Cluster Center</span>';
    }
    html += '<br>';
    html += '<span style="color:#888;">Cluster-' + d.cluster + '</span>';
    if(d.molImg){
        html += '<br><img src="data:image/svg+xml;base64,' + d.molImg + '" '
            + 'style="max-width:200px;margin:6px 0;border:1px solid #eee;border-radius:4px;">';
    }
    if(d.cvdProps){
        html += '<hr style="border-color:#ddd;margin:4px 0;">';
        for(let k in d.cvdProps){
            if(d.cvdProps[k] !== null){
                html += '<span style="color:#E64B35;">' + k + ':</span> ' + d.cvdProps[k] + '<br>';
            }
        }
    }
    if(d.funcGroups && d.funcGroups.length > 0){
        html += '<hr style="border-color:#ddd;margin:4px 0;">';
        html += '<span style="color:#00A087;">Functional Groups:</span> ' + d.funcGroups.join(', ') + '<br>';
    }
    if(d.simWeight !== undefined){
        html += '<hr style="border-color:#ddd;margin:4px 0;">';
        html += '<span style="color:#762A83;">Similarity Weight:</span> ' + d.simWeight.toFixed(3) + '<br>';
    }
    tooltip.html(html)
        .style('left', (event.pageX + 15) + 'px')
        .style('top', (event.pageY - 10) + 'px')
        .transition().duration(200).style('opacity', 1);
}
function hideTooltip(){
    tooltip.transition().duration(300).style('opacity', 0);
}

function clusterForce(alpha){
    const centers = {};
    const counts = {};
    nodes.forEach(function(n){
        if(!centers[n.cluster]) { centers[n.cluster] = {x:0, y:0}; counts[n.cluster] = 0; }
        centers[n.cluster].x += n.x;
        centers[n.cluster].y += n.y;
        counts[n.cluster] += 1;
    });
    for(let c in centers){
        centers[c].x /= counts[c];
        centers[c].y /= counts[c];
    }
    const strength = 0.12 * alpha;
    nodes.forEach(function(n){
        const center = centers[n.cluster];
        n.vx += (center.x - n.x) * strength;
        n.vy += (center.y - n.y) * strength;
    });
}

function radialPull(alpha){
    const cx = width / 2, cy = height / 2;
    const strength = 0.08 * alpha;
    nodes.forEach(function(n){
        n.vx += (cx - n.x) * strength;
        n.vy += (cy - n.y) * strength;
    });
}

const simulation = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(links).id(function(d){ return d.id; })
        .distance(function(d){ return d.sameCluster ? 55 : 75; })
        .strength(function(d){ return d.sameCluster ? 0.5 : 0.25; }))
    .force('charge', d3.forceManyBody().strength(-30))
    .force('center', d3.forceCenter(width / 2, height / 2))
    .force('collision', d3.forceCollide().radius(function(d){ return 8 + (d.degree||1)*0.7; }))
    .force('cluster', clusterForce)
    .force('radial', radialPull)
    .alphaDecay(0.005)
    .on('tick', ticked);

function ticked(){
    link.attr('d', computeArcPath);
    node.attr('cx', function(d){ return d.x; })
        .attr('cy', function(d){ return d.y; });
    centerRing.attr('cx', function(d){ return d.x; })
        .attr('cy', function(d){ return d.y; });
}

function dragstarted(event, d){
    if(!event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x; d.fy = d.y;
}
function dragged(event, d){ d.fx = event.x; d.fy = event.y; }
function dragended(event, d){
    if(!event.active) simulation.alphaTarget(0);
    d.fx = null; d.fy = null;
}

const sidebar = d3.select('#sidebar-content');

const clusterKeys = Object.keys(clusterStats).sort(function(a,b){ return +a - +b; });
clusterKeys.forEach(function(cid){
    const cs = clusterStats[cid];
    const card = sidebar.append('div')
        .attr('class', 'cluster-card')
        .style('background','#f8f8f8').style('border-radius','8px')
        .style('padding','10px 12px')
        .style('border-left','4px solid ' + palette[(+cid - 1) % palette.length])
        .style('box-shadow','0 1px 4px rgba(0,0,0,0.07)');

    card.append('div')
        .html('<b style="color:' + palette[(+cid - 1) % palette.length] + ';font-size:15px;">Cluster-' + cid + '</b>'
            + ' <span style="color:#888;font-size:13px;">(' + cs.size + ' compounds)</span>');

    if(cs.centerName){
        card.append('div')
            .html('<span style="color:#E64B35;font-size:11px;">★ Center: </span>'
                + '<span style="font-size:11px;color:#444;">' + cs.centerName + '</span>')
            .style('margin-top','3px');
    }

    if(cs.groupCounts && Object.keys(cs.groupCounts).length > 0){
        const pieDiv = card.append('div').style('margin','6px 0').style('display','flex')
            .style('align-items','center').style('gap','8px');
        const pieW = 80, pieH = 80, radius = 36;
        const pieSvg = pieDiv.append('svg').attr('width', pieW).attr('height', pieH)
            .style('flex-shrink','0');
        const pieG = pieSvg.append('g').attr('transform','translate(' + pieW/2 + ',' + pieH/2 + ')');

        const entries = Object.entries(cs.groupCounts).sort(function(a,b){ return b[1]-a[1]; });
        const total = entries.reduce(function(s,e){ return s+e[1]; }, 0);
        const pieData = entries.map(function(e,i){
            return {label: e[0], value: e[1], color: pieColors[i % pieColors.length]};
        });

        const pie = d3.pie().value(function(d){ return d.value; }).sort(null);
        const arc = d3.arc().innerRadius(0).outerRadius(radius);

        pieG.selectAll('path').data(pie(pieData)).enter().append('path')
            .attr('d', arc)
            .attr('fill', function(d){ return d.data.color; })
            .attr('stroke', '#fff').attr('stroke-width', 1);

        const legendDiv = pieDiv.append('div').style('font-size','11px').style('color','#555')
            .style('line-height','1.5').style('overflow','hidden');
        pieData.slice(0, 4).forEach(function(d){
            const pct = (d.value / total * 100).toFixed(0);
            legendDiv.append('span')
                .html('<span style="display:inline-block;width:9px;height:9px;'
                    + 'background:' + d.color + ';border-radius:2px;margin-right:3px;vertical-align:middle;"></span>'
                    + d.label + ' ' + pct + '%<br>');
        });
        if(pieData.length > 4){
            legendDiv.append('span')
                .html('<span style="color:#aaa;">+' + (pieData.length - 4) + ' more</span>');
        }
    }

    if(cs.repMols && cs.repMols.length > 0){
        const rep = cs.repMols[0];
        const repDiv = card.append('div').style('margin-top','6px')
            .style('display','flex').style('align-items','center').style('gap','8px');
        if(rep.img){
            repDiv.append('img')
                .attr('src','data:image/svg+xml;base64,' + rep.img)
                .style('width','110px').style('height','auto')
                .style('border','1px solid #ddd')
                .style('border-radius','4px').style('background','#fff')
                .style('flex-shrink','0');
        }
        repDiv.append('div').text(rep.name)
            .style('font-size','11px').style('color','#444')
            .style('word-wrap','break-word').style('overflow-wrap','break-word')
            .style('word-break','break-all')
            .style('white-space','normal')
            .style('line-height','1.4')
            .style('max-width','140px');
    }

    if(cs.avgProps){
        const propDiv = card.append('div').style('margin-top','4px')
            .style('font-size','11px').style('color','#666').style('line-height','1.5');
        let propHtml = '';
        var showKeys = ['MolLogP', 'TPSA', 'MolWt'];
        showKeys.forEach(function(pk, idx){
            if(cs.avgProps[pk] !== undefined){
                propHtml += '<span style="color:#E64B35;">' + pk + ':</span>' + cs.avgProps[pk];
                if(idx < showKeys.length - 1) propHtml += ' &middot; ';
            }
        });
        propDiv.html(propHtml);
    }
});

const legendSvg = d3.select('#sidebar').append('div')
    .style('margin-top','10px').style('padding','6px 10px')
    .style('background','#f0f0f0').style('border-radius','6px')
    .style('font-size','12px').style('color','#555')
    .style('display','flex').style('align-items','center').style('gap','8px');

legendSvg.append('div')
    .style('width','22px').style('height','22px')
    .style('border','2px dashed #E64B35')
    .style('border-radius','50%')
    .style('flex-shrink','0');
legendSvg.append('span').text('Cluster Center Node');

function crc32(bytes){
    var table = crc32.table;
    if(!table){
        table = crc32.table = [];
        for(var n = 0; n < 256; n++){
            var c = n;
            for(var k = 0; k < 8; k++){
                c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1);
            }
            table[n] = c >>> 0;
        }
    }
    var crc = 0xffffffff;
    for(var i = 0; i < bytes.length; i++){
        crc = table[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
    }
    return (crc ^ 0xffffffff) >>> 0;
}

function pngBlobWithDpi(blob, dpi){
    return blob.arrayBuffer().then(function(buffer){
        var source = new Uint8Array(buffer);
        var ppm = Math.round(dpi / 0.0254);
        var data = new Uint8Array(9);
        data[0] = (ppm >>> 24) & 255; data[1] = (ppm >>> 16) & 255;
        data[2] = (ppm >>> 8) & 255; data[3] = ppm & 255;
        data[4] = (ppm >>> 24) & 255; data[5] = (ppm >>> 16) & 255;
        data[6] = (ppm >>> 8) & 255; data[7] = ppm & 255;
        data[8] = 1;
        var type = new Uint8Array([112, 72, 89, 115]);
        var crcInput = new Uint8Array(type.length + data.length);
        crcInput.set(type, 0); crcInput.set(data, type.length);
        var chunk = new Uint8Array(4 + crcInput.length + 4);
        chunk[3] = data.length;
        chunk.set(crcInput, 4);
        var crc = crc32(crcInput);
        var crcAt = 4 + crcInput.length;
        chunk[crcAt] = (crc >>> 24) & 255; chunk[crcAt + 1] = (crc >>> 16) & 255;
        chunk[crcAt + 2] = (crc >>> 8) & 255; chunk[crcAt + 3] = crc & 255;
        var ihdrLength = ((source[8] << 24) | (source[9] << 16) |
                          (source[10] << 8) | source[11]) >>> 0;
        var insertAt = 8 + 12 + ihdrLength;
        var output = new Uint8Array(source.length + chunk.length);
        output.set(source.subarray(0, insertAt), 0);
        output.set(chunk, insertAt);
        output.set(source.subarray(insertAt), insertAt + chunk.length);
        return new Blob([output], {type:'image/png'});
    });
}

function downloadBlob(blob, filename){
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.download = filename;
    a.href = url;
    a.click();
    setTimeout(function(){ URL.revokeObjectURL(url); }, 1000);
}

var exportButtons = [document.getElementById('btn-export'), document.getElementById('btn-export-pdf')];

document.getElementById('btn-export').addEventListener('click', function(){
    var dpi = 1000;
    exportButtons.forEach(function(button){ button.style.display = 'none'; });
    html2canvas(document.body, {
        scale: dpi / 96,
        useCORS: true,
        backgroundColor: '#ffffff',
        width: document.body.scrollWidth,
        height: document.body.scrollHeight,
        windowWidth: document.body.scrollWidth,
        windowHeight: document.body.scrollHeight,
        logging: false
    }).then(function(canvas){
        return new Promise(function(resolve, reject){
            canvas.toBlob(function(blob){
                if(blob) resolve(blob); else reject(new Error('PNG encoding failed'));
            }, 'image/png');
        });
    }).then(function(blob){
        return pngBlobWithDpi(blob, dpi);
    }).then(function(blob){
        downloadBlob(blob, 'network_full_1000dpi.png');
    }).catch(function(err){
        console.error('Export failed:', err);
        alert('Export failed. The 1000 DPI image may exceed browser memory.');
    }).finally(function(){
        exportButtons.forEach(function(button){ button.style.display = ''; });
    });
});

document.getElementById('btn-export-pdf').addEventListener('click', function(){
    window.print();
});

})();
</script>

"""

def generate_html(category, nodes_json, links_json, cluster_stats_json, n_clusters):
    col_count, sidebar_width = get_sidebar_columns(n_clusters)
    html = f"""<!DOCTYPE html>

<html lang="en">
<head>
<meta charset="UTF-8">
<title>{category} - Molecular Network</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#ffffff; color:#333; font-family:Arial,sans-serif; display:flex; height:100vh; overflow:hidden; }}
#sidebar {{
    width:{sidebar_width}px; min-width:{sidebar_width}px; max-width:{sidebar_width}px;
    background:#fafafa;
    overflow:hidden;
    padding:12px;
    border-right:1px solid #e0e0e0;
}}
#sidebar h2 {{ color:#3C5488; margin-bottom:4px; font-size:18px; font-family:Arial,sans-serif; }}
#sidebar-content {{
    font-size:13px;
    line-height:1.5;
    font-family:Arial,sans-serif;
    column-count: {col_count};
    column-gap: 10px;
    -webkit-column-count: {col_count};
    -webkit-column-gap: 10px;
    -moz-column-count: {col_count};
    -moz-column-gap: 10px;
}}
.cluster-card {{
    break-inside: avoid;
    -webkit-column-break-inside: avoid;
    page-break-inside: avoid;
    margin-bottom: 10px;
    min-height: 180px;
}}
#network {{ flex:1; position:relative; background:#ffffff; }}
#btn-export {{
    position:fixed; top:12px; right:18px; z-index:9999;
    padding:8px 18px; background:#3C5488; color:#fff;
    border:none; border-radius:6px; font-size:14px;
    font-family:Arial,sans-serif; cursor:pointer;
    box-shadow:0 2px 8px rgba(0,0,0,0.15);
    transition: background 0.2s;
}}
#btn-export:hover {{ background:#2a3d6b; }}
#btn-export-pdf {{
    position:fixed; top:52px; right:18px; z-index:9999;
    padding:8px 18px; background:#5C8D89; color:#fff;
    border:none; border-radius:6px; font-size:14px;
    font-family:Arial,sans-serif; cursor:pointer;
    box-shadow:0 2px 8px rgba(0,0,0,0.15);
}}
#btn-export-pdf:hover {{ background:#3f6f6b; }}
a {{ color:#4DBBD5; }}
a:hover {{ color:#3C5488; }}
hr {{ border:none; border-top:1px solid #e0e0e0; margin:6px 0; }}
@media print {{
    @page {{ size: auto; margin: 0; }}
    body {{ height:auto; min-height:100vh; overflow:visible; }}
    #btn-export, #btn-export-pdf {{ display:none !important; }}
    #sidebar {{ overflow:visible; height:auto; }}
    #network {{ min-height:100vh; }}
    #main-svg *, .cluster-card, #sidebar * {{
        filter:none !important;
        box-shadow:none !important;
        text-shadow:none !important;
    }}
}}
</style>
</head>
<body>
<button id="btn-export">Export 1000 DPI PNG</button>
<button id="btn-export-pdf">Export Vector PDF</button>
<div id="sidebar">
    <h2>{category}</h2>
    <p style="color:#888;font-size:13px;margin-bottom:8px;">Molecular network &middot; {n_clusters} clusters &middot; Weighted similarity</p>
    <hr>
    <div id="sidebar-content"></div>
</div>
<div id="network"></div>
<script>
window.__NODES__ = {nodes_json};
window.__LINKS__ = {links_json};
window.__PALETTE__ = {json.dumps(NATURE_PALETTE)};
window.__PIE_COLORS__ = {json.dumps(PIE_COLORS)};
window.__CLUSTER_STATS__ = {cluster_stats_json};
</script>
{_js()}
</body>
</html>"""
    return html

# ═══════════════════════════════════════════════════════════════
# Process Single Category
# ═══════════════════════════════════════════════════════════════

def process_category(cat_name, df_cat, output_dir):
    """
    Run the full pipeline for one category.

    Returns a list of {"original_index": ..., "cluster": ...} dicts used to
    fill the Cluster column of the output CSV.
    """
    print(f"\n{'=' * 60}")
    print(f"  Processing: {cat_name} ({len(df_cat)} compounds)")
    print(f"{'=' * 60}")

    cluster_assignments = []

    # 1. Parse SMILES
    mols, valid_rows, valid_original_indices = [], [], []
    for idx, row in df_cat.iterrows():
        smi = str(row.get("SMILES", "")).strip()
        if not smi or smi == "nan":
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mols.append(mol)
            valid_rows.append(row)
            valid_original_indices.append(idx)
    if len(mols) < MIN_MOLECULES:
        print(f"  Warning: valid molecules < {MIN_MOLECULES}, skipped")
        return cluster_assignments
    n = len(mols)
    print(f"  Valid molecules: {n}")

    # 2. Compute RDKit ~200 physicochemical descriptors
    print("  -> Computing ~200 RDKit physicochemical descriptors...")
    desc_array = []
    for mol in mols:
        desc = compute_rdkit_descriptors(mol)
        desc_array.append(desc)
    desc_array = np.array(desc_array, dtype=float)
    desc_array = np.nan_to_num(desc_array, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"     [OK] Descriptor matrix: {desc_array.shape[0]} x {desc_array.shape[1]}")

    # 3. Tanimoto distance matrix from descriptors
    print("  -> Computing Tanimoto distance matrix on descriptors...")
    dist_matrix = tanimoto_distance_matrix_from_descriptors(desc_array)
    dist_matrix = sanitize_distance_matrix(dist_matrix)

    median_dist = np.median(dist_matrix[np.triu_indices(n, k=1)])
    print(f"     [OK] Median Tanimoto distance: {median_dist:.3f}")
    print(f"     [OK] Distance range: [{dist_matrix.min():.6f}, {dist_matrix.max():.6f}]")

    # 4. Build Tanimoto distance graph + Louvain clustering
    print("  -> Building Tanimoto distance graph & Louvain clustering...")
    # Relax the edge threshold until the graph has at least n edges
    threshold = TANIMOTO_EDGE_THRESHOLD
    G_tan = build_tanimoto_graph(dist_matrix, threshold)
    while G_tan.number_of_edges() < n and threshold < 0.95:
        threshold += 0.05
        G_tan = build_tanimoto_graph(dist_matrix, threshold)
    print(f"     [OK] Tanimoto graph: {G_tan.number_of_nodes()} nodes, {G_tan.number_of_edges()} edges (threshold={threshold:.2f})")

    if G_tan.number_of_edges() >= n // 2:
        labels_array, n_clusters, best_gamma = louvain_plateau_clustering(
            G_tan, LOUVAIN_GAMMA_RANGE, LOUVAIN_GAMMA_STEP,
            LOUVAIN_N_RUNS, seed=RANDOM_SEED)
    else:
        print("     [WARN] Sparse graph, falling back to k-NN Louvain")
        labels_array, n_clusters = louvain_simple_fallback(dist_matrix, n)
        best_gamma = 1.0

    if n_clusters < 2:
        labels_array, n_clusters = louvain_simple_fallback(dist_matrix, n)
        best_gamma = 1.0

    partition = {i: int(labels_array[i]) for i in range(n)}
    print(f"     Best gamma={best_gamma:.2f}, clusters={n_clusters}")

    center_set = find_cluster_centers(labels_array, dist_matrix)
    print(f"     Cluster centers: {len(center_set)} nodes identified")

    # Cluster labels already start from 1
    for i in range(n):
        cluster_assignments.append({
            "original_index": valid_original_indices[i],
            "cluster": int(labels_array[i]),
        })

    # 5. t-SNE
    print("  -> t-SNE embedding...")
    perp = min(TSNE_PERPLEXITY, n - 1)
    tsne = TSNE(n_components=2, perplexity=perp,
                learning_rate=TSNE_LEARNING_RATE,
                max_iter=TSNE_MAX_ITER, random_state=RANDOM_SEED,
                metric="precomputed", init="random")
    coords = tsne.fit_transform(dist_matrix)

    # 6. Build network
    print("  -> Building MST+k-NN network...")
    G = build_network(dist_matrix, KNN_EXTRA, EDGE_DIST_CUTOFF)

    # 7. Functional groups & CVD properties & mol images
    print("  -> Detecting functional groups & properties & generating images...")
    func_groups_list = []
    cvd_props_list = []
    mol_imgs = []
    for mol in mols:
        groups = detect_functional_groups(mol)
        func_groups_list.append(groups)
        cvd_props_list.append(compute_cvd_properties(mol))
        mol_imgs.append(mol_to_base64_svg(mol, size=(200, 150)))

    # 8. Compute per-node average similarity weight
    sim_weights = []
    for i in range(n):
        neighbors = list(G.neighbors(i))
        if neighbors:
            avg_sim = np.mean([1.0 - dist_matrix[i][j] for j in neighbors])
        else:
            avg_sim = 0.0
        sim_weights.append(round(float(avg_sim), 4))

    # 9. Nodes JSON
    degree_dict = dict(G.degree())
    nodes_data = []
    for i in range(n):
        row = valid_rows[i]
        raw_name = str(row.get("Name", f"Compound_{i}"))
        nodes_data.append({
            "id": i,
            "name": title_case_name(raw_name),
            "smiles": str(row.get("SMILES", "")),
            "cluster": int(partition.get(i, 1)),
            "x": float(coords[i][0]) * 2 + 400,
            "y": float(coords[i][1]) * 2 + 400,
            "degree": degree_dict.get(i, 1),
            "funcGroups": func_groups_list[i],
            "cvdProps": cvd_props_list[i],
            "molImg": mol_imgs[i],
            "simWeight": sim_weights[i],
            "isCenter": i in center_set,
        })

    # 10. Links JSON
    links_data = []
    for u, v, d in G.edges(data=True):
        links_data.append({
            "source": int(u),
            "target": int(v),
            "weight": round(d.get("weight", 0.5), 4),
            "sameCluster": partition.get(u, -1) == partition.get(v, -2),
        })

    # 11. Cluster stats
    cluster_stats = {}
    for cid in sorted(set(partition.values())):
        members = [i for i, c in partition.items() if c == cid]
        all_groups = []
        for i in members:
            all_groups.extend(func_groups_list[i])
        group_counts = dict(Counter(all_groups).most_common(8))
        avg_props = {}
        for prop in ["MolLogP", "TPSA", "MolWt", "NumHDonors", "NumHAcceptors",
                      "NumRotatableBonds", "FractionCSP3", "NumAromaticRings", "BertzCT"]:
            vals = [cvd_props_list[i][prop] for i in members if cvd_props_list[i].get(prop) is not None]
            if vals:
                avg_props[prop] = round(np.mean(vals), 2)
        rep_mols = select_representative_mols(mols, valid_rows, labels_array, cid, dist_matrix, max_rep=1)

        center_name = ""
        for i in members:
            if i in center_set:
                raw_center_name = str(valid_rows[i].get("Name", f"Compound_{i}"))
                center_name = title_case_name(raw_center_name)
                break

        cluster_stats[int(cid)] = {
            "size": len(members),
            "groupCounts": group_counts,
            "avgProps": avg_props,
            "repMols": rep_mols,
            "centerName": center_name,
        }

    # 12. Generate HTML
    print("  -> Generating HTML...")
    html_content = generate_html(
        category=cat_name,
        nodes_json=json.dumps(nodes_data, ensure_ascii=False),
        links_json=json.dumps(links_data, ensure_ascii=False),
        cluster_stats_json=json.dumps(cluster_stats, ensure_ascii=False),
        n_clusters=n_clusters,
    )

    safe_name = cat_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
    out_path = os.path.join(output_dir, f"{safe_name}_network.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"  Saved: {out_path}")
    print(f"     nodes={len(nodes_data)}, edges={len(links_data)}, clusters={n_clusters}")

    return cluster_assignments

# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Unsupervised molecular classification and network visualization.")
    parser.add_argument("-i", "--input", default=str(DEFAULT_INPUT_FILE),
                        help="Input Excel/CSV file with Name, Category and SMILES columns "
                             "(default: compounds_list.xlsx next to this script)")
    parser.add_argument("-o", "--output", default=str(DEFAULT_OUTPUT_DIR),
                        help="Output directory (default: ./results next to this script)")
    return parser.parse_args()

def read_input_table(path):
    if str(path).lower().endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_excel(path)

def main():
    args = parse_args()
    input_file = args.input
    output_dir = args.output

    print("=" * 60)
    print("  Cardiovascular Emerging Contaminants")
    print("  Molecular Network Visualization Tool")
    print("  (Weighted Structural Similarity Edition)")
    print("=" * 60)

    print(f"\n-> Reading data: {input_file}")
    df = read_input_table(input_file)
    print(f"   Total rows: {len(df)}")
    print(f"   Columns: {list(df.columns)}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Cluster column, empty for molecules that were not clustered
    df["Cluster"] = pd.NA

    if "Category" not in df.columns:
        print("Warning: 'Category' column not found, processing all data as one group")
        cluster_assignments = process_category("All_Compounds", df, output_dir)
        for item in cluster_assignments:
            df.at[item["original_index"], "Cluster"] = item["cluster"]
    else:
        categories = df["Category"].dropna().unique()
        print(f"\n-> Found {len(categories)} categories: {list(categories)}")
        for cat in sorted(categories):
            df_cat = df[df["Category"] == cat].reset_index(drop=False)
            try:
                cluster_assignments = process_category(str(cat), df_cat, output_dir)
                for item in cluster_assignments:
                    original_df_index = df_cat.at[item["original_index"], "index"]
                    df.at[original_df_index, "Cluster"] = item["cluster"]
            except Exception as e:
                print(f"\n  Error processing '{cat}': {e}")
                traceback.print_exc()

    csv_output_path = os.path.join(output_dir, "compounds_with_clusters.csv")
    df["Cluster"] = df["Cluster"].apply(lambda x: int(x) if pd.notna(x) else "")
    df.to_csv(csv_output_path, index=False, encoding="utf-8-sig")
    print(f"\n  CSV with Cluster column saved: {csv_output_path}")

    if "Category" in df.columns:
        print("\n  Cluster assignment summary per Category:")
        for cat in sorted(df["Category"].dropna().unique()):
            df_sub = df[df["Category"] == cat]
            assigned = df_sub[df_sub["Cluster"] != ""]
            n_clusters_cat = assigned["Cluster"].nunique() if len(assigned) > 0 else 0
            print(f"    {cat}: {len(assigned)}/{len(df_sub)} compounds assigned, "
                  f"{n_clusters_cat} clusters")

    print(f"\n{'=' * 60}")
    print(f"  Done! HTML files saved to: {output_dir}")
    print(f"  CSV file saved to: {csv_output_path}")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
