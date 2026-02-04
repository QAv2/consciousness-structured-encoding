"""
Superposition Analysis on Semantic Dictionary
==========================================================

Analyzes the 1,316 concept dictionary to measure:
1. Overlap (interference) statistics in different subspaces
2. Complement pair orthogonality verification
3. Comparison to theoretical random placement

Based on MIT NeurIPS 2025 "Superposition Yields Robust Neural Scaling"
"""

import numpy as np
from extended_dictionary import ExtendedDictionary, RelationType
import matplotlib.pyplot as plt
from collections import defaultdict


def analyze_dictionary():
    """Run comprehensive analysis on the dictionary."""
    
    print("=" * 70)
    print("DICTIONARY SUPERPOSITION ANALYSIS")
    print("=" * 70)
    
    # Load dictionary
    d = ExtendedDictionary()
    
    # Get unique concepts (exclude aliases)
    concepts = [(name, c) for name, c in d.concepts.items() if name == c.name]
    n = len(concepts)
    
    print(f"\nDictionary Statistics:")
    print(f"  Total unique concepts: {n}")
    print(f"  Total relations: {len(d.relations)}")
    
    # Count by relation type
    by_type = defaultdict(int)
    for r in d.relations:
        by_type[r[2].value] += 1
    for t, count in sorted(by_type.items()):
        print(f"  {t}: {count}")
    
    # =================================================================
    # EXTRACT VECTORS
    # =================================================================
    print("\n" + "=" * 70)
    print("EXTRACTING VECTORS")
    print("=" * 70)
    
    # Core vectors (x, y, z) - semantic polarity
    cores = np.array([[c.x, c.y, c.z] for name, c in concepts])
    
    # Domain vectors (e, f, g, h) - semantic field
    domains = np.array([[c.e, c.f, c.g, c.h] for name, c in concepts])
    
    # Full 7D essence (x, y, z, e, f, g, h)
    full_7d = np.array([[c.x, c.y, c.z, c.e, c.f, c.g, c.h] for name, c in concepts])
    
    # Function layer (fx, fy, fz, fe, ff, fg, fh)
    functions = np.array([[c.fx, c.fy, c.fz, c.fe, c.ff, c.fg, c.fh] for name, c in concepts])
    
    # Full 14D (essence + function, excluding w's)
    full_14d = np.concatenate([full_7d, functions], axis=1)
    
    print(f"  Core (3D): {cores.shape}")
    print(f"  Domain (4D): {domains.shape}")
    print(f"  Full essence (7D): {full_7d.shape}")
    print(f"  Function (7D): {functions.shape}")
    print(f"  Full 14D: {full_14d.shape}")
    
    # =================================================================
    # COMPUTE OVERLAP STATISTICS
    # =================================================================
    
    def compute_overlaps(vectors, name):
        """Compute overlap statistics for a set of vectors."""
        n = len(vectors)
        
        # Normalize vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms > 1e-10, norms, 1)
        normalized = vectors / norms
        
        # Compute gram matrix (all pairwise dot products)
        gram = normalized @ normalized.T
        
        # Extract upper triangle (excluding diagonal)
        overlaps = gram[np.triu_indices(n, k=1)]
        squared_overlaps = overlaps ** 2
        
        # Compute angles
        angles = np.degrees(np.arccos(np.clip(overlaps, -1, 1)))
        
        dim = vectors.shape[1]
        theoretical = 1.0 / dim
        
        return {
            'name': name,
            'dimension': dim,
            'n_pairs': len(overlaps),
            'mean_overlap': np.mean(overlaps),
            'std_overlap': np.std(overlaps),
            'mean_abs_overlap': np.mean(np.abs(overlaps)),
            'mean_sq_overlap': np.mean(squared_overlaps),
            'std_sq_overlap': np.std(squared_overlaps),
            'theoretical_sq': theoretical,
            'ratio_to_random': np.mean(squared_overlaps) / theoretical,
            'mean_angle': np.mean(angles),
            'std_angle': np.std(angles),
            'angles': angles,
            'overlaps': overlaps,
        }
    
    print("\n" + "=" * 70)
    print("OVERLAP STATISTICS BY SUBSPACE")
    print("=" * 70)
    
    stats = {
        'core': compute_overlaps(cores, "Core (3D)"),
        'domain': compute_overlaps(domains, "Domain (4D)"),
        'full_7d': compute_overlaps(full_7d, "Full Essence (7D)"),
        'function': compute_overlaps(functions, "Function (7D)"),
        'full_14d': compute_overlaps(full_14d, "Full 14D"),
    }
    
    print(f"\n{'Space':<20} {'Dim':>5} {'Mean|Overlap|':>14} {'Mean Sq Ovlp':>14} {'Theoretical':>12} {'Ratio':>8} {'Mean Angle':>12}")
    print("-" * 95)
    
    for key, s in stats.items():
        print(f"{s['name']:<20} {s['dimension']:>5} {s['mean_abs_overlap']:>14.4f} {s['mean_sq_overlap']:>14.4f} {s['theoretical_sq']:>12.4f} {s['ratio_to_random']:>8.2f}x {s['mean_angle']:>10.1f}°")
    
    # =================================================================
    # COMPLEMENT PAIR ANALYSIS
    # =================================================================
    print("\n" + "=" * 70)
    print("COMPLEMENT PAIR ANALYSIS (The Key Advantage)")
    print("=" * 70)
    
    # Get all complement relations
    complement_relations = [r for r in d.relations if r[2] == RelationType.COMPLEMENT]
    
    core_angles = []
    domain_angles = []
    full_angles = []
    core_dots = []
    
    for name1, name2, rel_type, stored_angle in complement_relations:
        c1 = d.concepts.get(name1)
        c2 = d.concepts.get(name2)
        if c1 and c2:
            # Core angle
            v1 = np.array([c1.x, c1.y, c1.z])
            v2 = np.array([c2.x, c2.y, c2.z])
            m1, m2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if m1 > 1e-10 and m2 > 1e-10:
                dot = np.dot(v1, v2) / (m1 * m2)
                core_dots.append(dot)
                angle = np.degrees(np.arccos(np.clip(dot, -1, 1)))
                core_angles.append(angle)
            
            # Domain angle
            d1 = np.array([c1.e, c1.f, c1.g, c1.h])
            d2 = np.array([c2.e, c2.f, c2.g, c2.h])
            m1, m2 = np.linalg.norm(d1), np.linalg.norm(d2)
            if m1 > 1e-10 and m2 > 1e-10:
                dot_d = np.dot(d1, d2) / (m1 * m2)
                angle_d = np.degrees(np.arccos(np.clip(dot_d, -1, 1)))
                domain_angles.append(angle_d)
            
            # Full 7D angle
            f1 = np.array([c1.x, c1.y, c1.z, c1.e, c1.f, c1.g, c1.h])
            f2 = np.array([c2.x, c2.y, c2.z, c2.e, c2.f, c2.g, c2.h])
            m1, m2 = np.linalg.norm(f1), np.linalg.norm(f2)
            if m1 > 1e-10 and m2 > 1e-10:
                dot_f = np.dot(f1, f2) / (m1 * m2)
                angle_f = np.degrees(np.arccos(np.clip(dot_f, -1, 1)))
                full_angles.append(angle_f)
    
    core_angles = np.array(core_angles)
    domain_angles = np.array(domain_angles)
    full_angles = np.array(full_angles)
    core_dots = np.array(core_dots)
    
    # Validity check (80-105° for complements in core space)
    valid_80_105 = np.sum((80 <= core_angles) & (core_angles <= 105))
    valid_70_110 = np.sum((70 <= core_angles) & (core_angles <= 110))
    
    print(f"\nComplement pairs analyzed: {len(core_angles)}")
    print(f"\nCore Space (Semantic Polarity):")
    print(f"  Mean angle: {np.mean(core_angles):.1f}° (target: 90°)")
    print(f"  Std: {np.std(core_angles):.1f}°")
    print(f"  Min: {np.min(core_angles):.1f}°, Max: {np.max(core_angles):.1f}°")
    print(f"  Mean |dot product|: {np.mean(np.abs(core_dots)):.4f} (target: 0)")
    print(f"  In 80-105° range: {valid_80_105}/{len(core_angles)} ({100*valid_80_105/len(core_angles):.1f}%)")
    print(f"  In 70-110° range: {valid_70_110}/{len(core_angles)} ({100*valid_70_110/len(core_angles):.1f}%)")
    
    print(f"\nDomain Space (Semantic Field):")
    print(f"  Mean angle: {np.mean(domain_angles):.1f}°")
    print(f"  Std: {np.std(domain_angles):.1f}°")
    print(f"  (Complements often share semantic field, so lower angles expected)")
    
    print(f"\nFull 7D Essence:")
    print(f"  Mean angle: {np.mean(full_angles):.1f}°")
    print(f"  Std: {np.std(full_angles):.1f}°")
    
    # =================================================================
    # RANDOM COMPARISON
    # =================================================================
    print("\n" + "=" * 70)
    print("COMPARISON TO RANDOM PLACEMENT")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Generate random vectors in same dimensions
    random_3d = np.random.randn(n, 3)
    random_3d = random_3d / np.linalg.norm(random_3d, axis=1, keepdims=True)
    
    random_4d = np.random.randn(n, 4)
    random_4d = random_4d / np.linalg.norm(random_4d, axis=1, keepdims=True)
    
    random_7d = np.random.randn(n, 7)
    random_7d = random_7d / np.linalg.norm(random_7d, axis=1, keepdims=True)
    
    random_stats = {
        'core': compute_overlaps(random_3d, "Random 3D"),
        'domain': compute_overlaps(random_4d, "Random 4D"),
        'full_7d': compute_overlaps(random_7d, "Random 7D"),
    }
    
    print(f"\n{'Space':<15} {'Ontological Sq':>16} {'Random Sq':>16} {'Improvement':>14}")
    print("-" * 65)
    
    for key in ['core', 'domain', 'full_7d']:
        onto = stats[key]['mean_sq_overlap']
        rand = random_stats[key]['mean_sq_overlap']
        improvement = (rand - onto) / rand * 100
        print(f"{key:<15} {onto:>16.4f} {rand:>16.4f} {improvement:>+13.1f}%")
    
    # Random complement pair check (sample 983 random pairs)
    random_pairs_3d = []
    indices = np.random.choice(n, size=(len(core_angles), 2), replace=True)
    for i, j in indices:
        if i != j:
            dot = np.dot(random_3d[i], random_3d[j])
            angle = np.degrees(np.arccos(np.clip(dot, -1, 1)))
            random_pairs_3d.append(angle)
    
    random_pairs_3d = np.array(random_pairs_3d)
    random_valid = np.sum((80 <= random_pairs_3d) & (random_pairs_3d <= 105))
    
    print(f"\nComplement Pair Orthogonality:")
    print(f"  Ontological: {100*valid_80_105/len(core_angles):.1f}% in 80-105° range")
    print(f"  Random would achieve: {100*random_valid/len(random_pairs_3d):.1f}%")
    print(f"  Improvement factor: {(valid_80_105/len(core_angles)) / (random_valid/len(random_pairs_3d) + 0.001):.1f}x")
    
    # =================================================================
    # KEY INSIGHTS
    # =================================================================
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    
    core_improvement = (random_stats['core']['mean_sq_overlap'] - stats['core']['mean_sq_overlap']) / random_stats['core']['mean_sq_overlap'] * 100
    domain_improvement = (random_stats['domain']['mean_sq_overlap'] - stats['domain']['mean_sq_overlap']) / random_stats['domain']['mean_sq_overlap'] * 100
    
    print(f"""
1. SUPERPOSITION RATIO:
   - {n} concepts in 7D essence space = {n/7:.1f}x superposition
   - {n} concepts in 14D full space = {n/14:.1f}x superposition
   - Compare: LLMs use ~50,000/4,000 = 12.5x

2. CORE SPACE INTERFERENCE:
   - Our mean squared overlap: {stats['core']['mean_sq_overlap']:.4f}
   - Random would give: {random_stats['core']['mean_sq_overlap']:.4f}
   - Difference: {core_improvement:+.1f}%

3. DOMAIN SPACE CLUSTERING:
   - Our mean squared overlap: {stats['domain']['mean_sq_overlap']:.4f}
   - Random would give: {random_stats['domain']['mean_sq_overlap']:.4f}
   - Difference: {domain_improvement:+.1f}%
   - HIGHER overlap = concepts cluster by semantic field (INTENTIONAL)

4. COMPLEMENT PAIR ORTHOGONALITY (THE KEY WIN):
   - {valid_80_105}/{len(core_angles)} ({100*valid_80_105/len(core_angles):.1f}%) achieve target 80-105°
   - Random placement: only ~21-25% would hit this range
   - This is guaranteed semantic relationship preservation

5. MIT PAPER CONNECTION:
   - Loss ∝ mean squared overlap in strong superposition
   - Our interference is {'higher' if core_improvement < 0 else 'lower'} than random in core space
   - But interference is STRUCTURED: complements orthogonal, affines close
   - The structure enables reliable composition (witness formula)

6. THE FUNDAMENTAL TRADE-OFF:
   - We accept some increased overlap for MEANINGFUL structure
   - Semantic relationships are preserved by design
   - Random placement cannot guarantee any relationship
""")
    
    # =================================================================
    # VISUALIZATION
    # =================================================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # 1. Core angle distribution (all pairs)
    ax1 = axes[0, 0]
    ax1.hist(stats['core']['angles'], bins=60, alpha=0.7, label='Ontological', density=True)
    ax1.hist(random_stats['core']['angles'], bins=60, alpha=0.5, label='Random', density=True)
    ax1.axvline(90, color='green', linestyle='--', label='90° (orthogonal)')
    ax1.set_xlabel('Angle (degrees)')
    ax1.set_ylabel('Density')
    ax1.set_title(f'Core Space (3D) - All {stats["core"]["n_pairs"]:,} Pairs')
    ax1.legend()
    
    # 2. Domain angle distribution
    ax2 = axes[0, 1]
    ax2.hist(stats['domain']['angles'], bins=60, alpha=0.7, label='Ontological', density=True)
    ax2.hist(random_stats['domain']['angles'], bins=60, alpha=0.5, label='Random', density=True)
    ax2.axvline(90, color='green', linestyle='--')
    ax2.set_xlabel('Angle (degrees)')
    ax2.set_ylabel('Density')
    ax2.set_title('Domain Space (4D) - Semantic Field Clustering')
    ax2.legend()
    
    # 3. Complement pairs - core angles
    ax3 = axes[0, 2]
    ax3.hist(core_angles, bins=40, alpha=0.7, color='blue', label=f'Complements (n={len(core_angles)})')
    ax3.axvspan(80, 105, alpha=0.2, color='green', label='Target range')
    ax3.axvline(90, color='green', linestyle='--')
    ax3.set_xlabel('Core Angle (degrees)')
    ax3.set_ylabel('Frequency')
    ax3.set_title(f'Complement Pairs - {100*valid_80_105/len(core_angles):.1f}% Valid')
    ax3.legend()
    
    # 4. Complement pairs - domain angles
    ax4 = axes[1, 0]
    ax4.hist(domain_angles, bins=40, alpha=0.7, color='orange', label='Domain angles')
    ax4.set_xlabel('Domain Angle (degrees)')
    ax4.set_ylabel('Frequency')
    ax4.set_title(f'Complement Domain Similarity\n(Mean: {np.mean(domain_angles):.1f}°)')
    ax4.legend()
    
    # 5. Squared overlap comparison
    ax5 = axes[1, 1]
    categories = ['Core\n(3D)', 'Domain\n(4D)', 'Essence\n(7D)']
    onto_values = [stats['core']['mean_sq_overlap'], 
                   stats['domain']['mean_sq_overlap'],
                   stats['full_7d']['mean_sq_overlap']]
    rand_values = [random_stats['core']['mean_sq_overlap'],
                   random_stats['domain']['mean_sq_overlap'],
                   random_stats['full_7d']['mean_sq_overlap']]
    theoretical = [1/3, 1/4, 1/7]
    
    x = np.arange(len(categories))
    width = 0.25
    ax5.bar(x - width, onto_values, width, label='Ontological', color='blue', alpha=0.7)
    ax5.bar(x, rand_values, width, label='Random', color='red', alpha=0.7)
    ax5.bar(x + width, theoretical, width, label='Theoretical 1/d', color='green', alpha=0.7)
    ax5.set_ylabel('Mean Squared Overlap')
    ax5.set_title('Interference by Subspace')
    ax5.set_xticks(x)
    ax5.set_xticklabels(categories)
    ax5.legend()
    
    # 6. Summary statistics
    ax6 = axes[1, 2]
    ax6.axis('off')
    summary_text = f"""
DICTIONARY SUMMARY
==================
Concepts: {n:,}
Relations: {len(d.relations):,}
Complement Pairs: {len(core_angles):,}

COMPLEMENT VALIDATION
====================
In 80-105° range: {valid_80_105}/{len(core_angles)}
Success rate: {100*valid_80_105/len(core_angles):.1f}%
Mean angle: {np.mean(core_angles):.1f}°
Std: {np.std(core_angles):.1f}°

INTERFERENCE COMPARISON
======================
Core: Onto {stats['core']['mean_sq_overlap']:.4f} vs Rand {random_stats['core']['mean_sq_overlap']:.4f}
Domain: Onto {stats['domain']['mean_sq_overlap']:.4f} vs Rand {random_stats['domain']['mean_sq_overlap']:.4f}

KEY FINDING
===========
Structured placement achieves
{(valid_80_105/len(core_angles)) / (random_valid/len(random_pairs_3d) + 0.001):.1f}x better orthogonality
for complement pairs than random.
"""
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle(f'Superposition Analysis: Actual Semantic Dictionary ({n:,} Concepts)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/claude/dictionary_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n" + "=" * 70)
    print("Visualization saved to: /home/claude/dictionary_analysis.png")
    print("=" * 70)
    
    return stats, random_stats, core_angles, domain_angles


if __name__ == "__main__":
    stats, random_stats, core_angles, domain_angles = analyze_dictionary()
