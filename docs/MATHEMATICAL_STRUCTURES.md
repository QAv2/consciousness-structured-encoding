# Mathematical Structures for Semantic Encoding

**Complete reference for the geometric and algebraic foundations**

---

## Table of Contents

1. [Overview](#1-overview)
2. [Quaternions (4D)](#2-quaternions-4d)
3. [Octonions (8D)](#3-octonions-8d)
4. [Dual Octonions (16D)](#4-dual-octonions-16d)
5. [Subspace Decomposition](#5-subspace-decomposition)
6. [Angle Calculations](#6-angle-calculations)
7. [The Witness Preservation Formula](#7-the-witness-preservation-formula)
8. [Normalization Conventions](#8-normalization-conventions)
9. [Relationship Geometry](#9-relationship-geometry)
10. [Composition Operations](#10-composition-operations)
11. [Validation Mathematics](#11-validation-mathematics)
12. [Implementation Reference](#12-implementation-reference)

---

## 1. Overview

This document describes the mathematical structures used in the Consciousness-Structured Semantic Encoding system. The encoding uses a **16-dimensional dual octonion** structure, decomposed into meaningful subspaces for validation and analysis.

### Dimensional Summary

| Structure | Total Dim | Free Dim | Purpose |
|-----------|-----------|----------|---------|
| Quaternion | 4 | 3 | Core polarity |
| Octonion | 8 | 7 | Full essence |
| Dual Octonion | 16 | 14 | Essence + Function |

**Free dimensions** exclude normalized witness components (always 1.0).

### The Hierarchy

```
Dual Octonion (16D)
├── Essence (8D)
│   ├── Core (4D: w, x, y, z)
│   │   └── Polarity (3D: x, y, z)
│   └── Domain (4D: e, f, g, h)
└── Function (8D)
    ├── Core Function (4D: 1, fx, fy, fz)
    └── Domain Function (4D: fe, ff, fg, fh)
```

---

## 2. Quaternions (4D)

### Definition

A quaternion Q is a 4-dimensional hypercomplex number:

```
Q = w + xi + yj + zk
```

Where:
- w, x, y, z ∈ ℝ (real numbers)
- i, j, k are imaginary units satisfying:
  - i² = j² = k² = -1
  - ij = k, jk = i, ki = j
  - ji = -k, kj = -i, ik = -j

### Algebraic Properties

| Property | Status | Implication |
|----------|--------|-------------|
| Associative | ✓ | (AB)C = A(BC) |
| Commutative | ✗ | AB ≠ BA in general |
| Division algebra | ✓ | Every Q ≠ 0 has inverse |
| Normed | ✓ | \|\|Q\|\| = √(w² + x² + y² + z²) |

### Semantic Mapping

| Component | Axis | Positive (+) | Negative (-) |
|-----------|------|--------------|--------------|
| w | Witness | Pure existence | — (always 1.0) |
| x | Yang-Yin | Active, expansive | Passive, contractive |
| y | Becoming-Abiding | Dynamic, process | Static, state |
| z | Ordinality | First, primary | Last, secondary |

### Quaternion Operations

**Conjugate:**
```
Q* = w - xi - yj - zk
```

**Norm:**
```
||Q|| = √(QQ*) = √(w² + x² + y² + z²)
```

**Inverse:**
```
Q⁻¹ = Q* / ||Q||²
```

**Multiplication:**
```
Q₁Q₂ = [w₁w₂ - v₁·v₂, w₁v₂ + w₂v₁ + v₁×v₂]

Where v = [x, y, z] is the vector part
```

Expanded:
```
Q₁Q₂ = (w₁w₂ - x₁x₂ - y₁y₂ - z₁z₂)
     + (w₁x₂ + x₁w₂ + y₁z₂ - z₁y₂)i
     + (w₁y₂ + y₁w₂ + z₁x₂ - x₁z₂)j
     + (w₁z₂ + z₁w₂ + x₁y₂ - y₁x₂)k
```

---

## 3. Octonions (8D)

### Definition

An octonion O is an 8-dimensional hypercomplex number:

```
O = a₀ + a₁e₁ + a₂e₂ + a₃e₃ + a₄e₄ + a₅e₅ + a₆e₆ + a₇e₇
```

Or in our notation:
```
O = [w, x, y, z, e, f, g, h]
```

### Algebraic Properties

| Property | Status | Implication |
|----------|--------|-------------|
| Associative | ✗ | (AB)C ≠ A(BC) in general |
| Commutative | ✗ | AB ≠ BA |
| Alternative | ✓ | (AA)B = A(AB), (AB)B = A(BB) |
| Division algebra | ✓ | Every O ≠ 0 has inverse |
| Normed | ✓ | \|\|O\|\| well-defined |

**Important:** Octonions are **not associative**. This affects composition but not our primary use case (angle measurement).

### Semantic Mapping (Essence Layer)

| Component | Domain | Meaning |
|-----------|--------|---------|
| w | Witness | Always 1.0 |
| x | Core | Yang-Yin polarity |
| y | Core | Becoming-Abiding polarity |
| z | Core | Ordinality polarity |
| e | Domain | Spatial (physical, bodily) |
| f | Domain | Temporal (time-related) |
| g | Domain | Relational (social, connected) |
| h | Domain | Personal (subjective, inner) |

### Cayley-Dickson Construction

Octonions can be constructed from quaternions:

```
O = (Q₁, Q₂)

Where Q₁, Q₂ are quaternions
Multiplication: (a,b)(c,d) = (ac - d*b, da + bc*)
```

This construction preserves norm but loses associativity.

---

## 4. Dual Octonions (16D)

### Definition

A dual octonion combines two octonions using the dual number ε where ε² = 0:

```
D = O_essence + ε · O_function
```

In component form:
```
D = [w, x, y, z, e, f, g, h] + ε·[1, fx, fy, fz, fe, ff, fg, fh]
```

### The Dual Number ε

The dual unit ε satisfies:
- ε² = 0 (nilpotent)
- ε ≠ 0

This creates a two-layer structure where:
- The "real" part (essence) describes what a concept IS
- The "dual" part (function) describes how it OPERATES

### Semantic Structure

**Essence (8D):** What the concept IS
```
[w=1.0, x, y, z, e, f, g, h]
```

**Function (8D):** How the concept OPERATES
```
[1, fx, fy, fz, fe, ff, fg, fh]
```

### Effective Dimensionality

| Layer | Total | Fixed | Free |
|-------|-------|-------|------|
| Essence | 8 | 1 (w=1) | 7 |
| Function | 8 | 1 (w=1) | 7 |
| **Total** | **16** | **2** | **14** |

For interference analysis, we work with 14 free dimensions.

---

## 5. Subspace Decomposition

### The Three Analysis Subspaces

Our validation framework separates the full space into meaningful subspaces:

```
Full Essence (7D free)
├── Core Space (3D): [x, y, z]
│   └── Measures: Semantic POLARITY
│   └── Complement target: 80-105°
│
└── Domain Space (4D): [e, f, g, h]
    └── Measures: Semantic FIELD
    └── Complement target: 0-45° (shared field)
```

### Why Separate Validation?

**Combined 8D angle is misleading** for complements:

HOT and COLD are:
- **Opposite** in polarity (core angle ~90°)
- **Similar** in field (both are temperature → domain angle ~15°)

If we measure combined 8D angle, these effects cancel out, giving a meaningless middle value.

**Always validate core and domain separately.**

### Subspace Extraction

```python
def extract_subspaces(concept):
    """Extract core and domain subspaces from a concept."""
    
    # Core: semantic polarity
    core = np.array([concept.x, concept.y, concept.z])
    
    # Domain: semantic field
    domain = np.array([concept.e, concept.f, concept.g, concept.h])
    
    # Full essence (excluding w)
    essence = np.array([concept.x, concept.y, concept.z,
                        concept.e, concept.f, concept.g, concept.h])
    
    return core, domain, essence
```

---

## 6. Angle Calculations

### Vector Angle Formula

For two vectors u and v, the angle θ between them:

```
cos(θ) = (u · v) / (||u|| × ||v||)

θ = arccos(cos(θ))
```

### Dot Product

```
u · v = Σᵢ uᵢvᵢ = u₁v₁ + u₂v₂ + ... + uₙvₙ
```

### Angle in Different Subspaces

**Core angle (3D):**
```python
def core_angle(c1, c2):
    u = np.array([c1.x, c1.y, c1.z])
    v = np.array([c2.x, c2.y, c2.z])
    
    cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
```

**Domain angle (4D):**
```python
def domain_angle(c1, c2):
    u = np.array([c1.e, c1.f, c1.g, c1.h])
    v = np.array([c2.e, c2.f, c2.g, c2.h])
    
    cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
```

**Full essence angle (7D):**
```python
def essence_angle(c1, c2):
    u = np.array([c1.x, c1.y, c1.z, c1.e, c1.f, c1.g, c1.h])
    v = np.array([c2.x, c2.y, c2.z, c2.e, c2.f, c2.g, c2.h])
    
    cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
```

### Numerical Stability

Always clip the cosine value to [-1, 1] before taking arccos:

```python
cos_theta = np.clip(cos_theta, -1, 1)
```

Floating-point errors can produce values like 1.0000000001, which would cause arccos to fail.

---

## 7. The Witness Preservation Formula

### Definition

When two concepts compose, the resulting witness component follows:

```
w = 1 - cos(θ)
```

Where θ is the angle between the concept vectors.

### Derivation

The formula emerges from considering what happens when two awareness states interact:

- **θ = 0°**: Identical concepts → cos(0°) = 1 → w = 0 (dissolution into identity)
- **θ = 90°**: Complementary concepts → cos(90°) = 0 → w = 1 (full preservation)
- **θ = 180°**: Contradictory concepts → cos(180°) = -1 → w = 2 (amplified tension)

### Interpretation Table

| Angle θ | cos(θ) | w Value | Semantic Meaning |
|---------|--------|---------|------------------|
| 0° | 1.0 | 0.0 | **Dissolution** — Identity collapses witness |
| 30° | 0.87 | 0.13 | Minimal preservation |
| 45° | 0.71 | 0.29 | Partial preservation |
| 60° | 0.50 | 0.50 | Half preservation |
| 90° | 0.0 | 1.0 | **Full preservation** — Healthy complement |
| 120° | -0.50 | 1.50 | Tension begins |
| 135° | -0.71 | 1.71 | Significant tension |
| 180° | -1.0 | 2.0 | **Maximum tension** — Contradiction |

### Why This Matters

The formula enables **principled composition** because:

1. **Complements preserve**: At 90°, combining HOT and COLD preserves full semantic content
2. **Synonyms dissolve**: At 0°, BIG + LARGE adds no information
3. **Oppositions amplify**: At 180°, TRUE + FALSE creates maximum semantic strain

**Critical**: This only works when θ is controlled by design. Random angles give random preservation.

### Visualization

```
w
2.0 ─┐                          ╱
     │                        ╱
1.5 ─┤                      ╱
     │                    ╱
1.0 ─┤──────────────────●────── ← Complements (90°)
     │                ╱
0.5 ─┤              ╱
     │            ╱
0.0 ─┼──────────●─────────────── ← Synonyms (0°)
     └─────┬────┬────┬────┬────┬─→ θ
           0°  45°  90° 135° 180°
```

---

## 8. Normalization Conventions

### Why Normalize?

Normalization ensures:
1. **Meaningful distances**: All concepts exist on the same scale
2. **Valid angles**: Cosine formula requires finite, non-zero norms
3. **Fair comparisons**: No concept has "more existence" than another

### Our Conventions

**Witness component**: Fixed at 1.0 (not normalized, just constant)
```
w = 1.0 always
```

**Core vector**: Unit normalized
```
||[x, y, z]|| = 1.0
```

**Domain vector**: Unit normalized
```
||[e, f, g, h]|| = 1.0
```

**Function vector**: Unit normalized (except zero-function Unity concepts)
```
||[fx, fy, fz, fe, ff, fg, fh]|| = 1.0  (or 0.0 for BEING, IS, ONE, etc.)
```

### Normalization Code

```python
def normalize_core(x, y, z):
    """Normalize core vector to unit length."""
    mag = np.sqrt(x**2 + y**2 + z**2)
    if mag > 0:
        return x/mag, y/mag, z/mag
    return 0.0, 0.0, 0.0

def normalize_domain(e, f, g, h):
    """Normalize domain vector to unit length."""
    mag = np.sqrt(e**2 + f**2 + g**2 + h**2)
    if mag > 0:
        return e/mag, f/mag, g/mag, h/mag
    return 0.0, 0.0, 0.0, 0.0
```

### Unity Concepts

Unity-level concepts (BEING, IS, ONE, TAO, I) have:
- Core: [0, 0, 0] (no polarity)
- Domain: [0, 0, 0, 0] (no field dominance)
- Function: [0, 0, 0, 0, 0, 0, 0] (pure witness, no operation)

These cannot be normalized (zero vectors) and represent the undifferentiated origin.

---

## 9. Relationship Geometry

### Relationship Types by Angle

| Relation | Core Angle | Domain Angle | Distance | Count in Dictionary |
|----------|------------|--------------|----------|---------------------|
| SYNONYM | 0-15° | 0-15° | ~0-0.26 | 126 |
| AFFINITY | 15-45° | 0-45° | ~0.26-0.77 | 2,010 |
| ADJACENT | 45-75° | Variable | ~0.77-1.22 | 116 |
| COMPLEMENT | 80-105° | 0-45° | ~1.29-1.53 | 983 |
| OPPOSITION | >105° | Variable | >1.53 | 29 |

### Distance Formula

For unit-normalized vectors, distance relates to angle:

```
d = ||u - v|| = √(2 - 2cos(θ)) = 2sin(θ/2)
```

| Angle | Distance |
|-------|----------|
| 0° | 0.00 |
| 30° | 0.52 |
| 45° | 0.77 |
| 60° | 1.00 |
| 90° | 1.41 (√2) |
| 120° | 1.73 (√3) |
| 180° | 2.00 |

### Complementarity Geometry

For complements at exactly 90°:

```
u · v = 0  (orthogonal)
d(u, v) = √2 ≈ 1.414
```

Complements share semantic field (low domain angle) while differing in polarity (high core angle):

```
HOT:  core=[+0.71, +0.50, +0.50]  domain=[0.70, 0.30, 0.30, 0.55]
COLD: core=[-0.71, -0.50, -0.50]  domain=[0.70, 0.30, 0.30, 0.55]

Core angle: ~90° (opposite polarity)
Domain angle: ~0° (same field: temperature)
```

### The Orthogonality Insight

Two vectors are orthogonal (90°) when their dot product equals zero:

```
u · v = 0 ⟺ θ = 90°
```

This can be achieved even when some components are similar, as long as others compensate:

```
u = [+a, +b, +c]
v = [-a, +b, +c]

u · v = -a² + b² + c²

For orthogonality: a² = b² + c²
```

This is why complements can share domain while differing in core.

---

## 10. Composition Operations

### Quaternion Multiplication

For phrase composition, quaternion multiplication captures modifier-head relationships:

```
"hot water" = Q(hot) × Q(water)
"water hot" = Q(water) × Q(hot)
```

These give **different results** because quaternion multiplication is non-commutative.

### Multiplication Formula

```
Q₁ × Q₂ = [w₁w₂ - v₁·v₂, w₁v₂ + w₂v₁ + v₁×v₂]
```

Where:
- v₁·v₂ is the dot product (scalar)
- v₁×v₂ is the cross product (vector)

### Semantic Interpretation

| Operation | Semantic Effect |
|-----------|-----------------|
| w₁w₂ | Witness interaction |
| -v₁·v₂ | Overlap/agreement (negative reduces witness) |
| w₁v₂ | First modifies second |
| w₂v₁ | Second modifies first |
| v₁×v₂ | Novel relationship emerges |

### Linear Combination (Blending)

For intermediate concepts:

```
"lukewarm" ≈ 0.5 × Q(hot) + 0.5 × Q(cold)
```

Normalize after blending to maintain unit norm.

### SLERP (Spherical Interpolation)

For smooth paths between meanings:

```python
def slerp(q1, q2, t):
    """Spherical linear interpolation between quaternions."""
    dot = q1.dot(q2)
    
    # Clamp and compute angle
    dot = np.clip(dot, -1, 1)
    theta = np.arccos(dot)
    
    if theta < 1e-6:
        return q1  # Nearly identical
    
    # Interpolate
    s1 = np.sin((1-t) * theta) / np.sin(theta)
    s2 = np.sin(t * theta) / np.sin(theta)
    
    return s1 * q1 + s2 * q2
```

SLERP maintains constant angular velocity along the interpolation path.

---

## 11. Validation Mathematics

### Complement Validation

A valid complement pair must satisfy:

```python
def is_valid_complement(c1, c2):
    core_ang = core_angle(c1, c2)
    domain_ang = domain_angle(c1, c2)
    
    # Core: opposite polarity
    core_valid = 80 <= core_ang <= 105
    
    # Domain: shared field (usually)
    domain_valid = domain_ang <= 45
    
    return core_valid and domain_valid
```

### Synonym Validation

```python
def is_valid_synonym(c1, c2):
    core_ang = core_angle(c1, c2)
    return core_ang <= 15
```

### Affinity Validation

```python
def is_valid_affinity(c1, c2):
    core_ang = core_angle(c1, c2)
    return 15 < core_ang <= 45
```

### Interference Calculation

Mean squared overlap (interference) for a set of vectors:

```python
def mean_squared_overlap(vectors):
    """Calculate mean squared dot product (interference)."""
    n = len(vectors)
    total = 0
    count = 0
    
    for i in range(n):
        for j in range(i+1, n):
            dot = np.dot(vectors[i], vectors[j])
            total += dot ** 2
            count += 1
    
    return total / count if count > 0 else 0
```

For random isotropic unit vectors in m dimensions:
```
E[⟨u,v⟩²] = 1/m
```

Our dictionary shows higher interference than random (see paper), indicating structured clustering.

### Superposition Ratio

```
superposition_ratio = num_concepts / num_dimensions
```

Our dictionary: 1,316 / 7 = 188× superposition in essence space.

---

## 12. Implementation Reference

### Core Data Structure

```python
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np

@dataclass
class SemanticConcept:
    """A concept encoded as a dual octonion."""
    
    name: str
    
    # Essence (8D)
    w: float = 1.0  # Witness (always 1.0)
    x: float = 0.0  # Yang-Yin
    y: float = 0.0  # Becoming-Abiding
    z: float = 0.0  # Ordinality
    e: float = 0.0  # Spatial domain
    f: float = 0.0  # Temporal domain
    g: float = 0.0  # Relational domain
    h: float = 0.0  # Personal domain
    
    # Function (7D free, w=1 implied)
    fx: float = 0.0
    fy: float = 0.0
    fz: float = 0.0
    fe: float = 0.0
    ff: float = 0.0
    fg: float = 0.0
    fh: float = 0.0
    
    # Metadata
    level: str = "DERIVED"
    trigram: str = "QIAN"
    relations: Dict[str, List[str]] = None
    
    @property
    def core(self) -> np.ndarray:
        """Core polarity vector [x, y, z]."""
        return np.array([self.x, self.y, self.z])
    
    @property
    def domain(self) -> np.ndarray:
        """Domain field vector [e, f, g, h]."""
        return np.array([self.e, self.f, self.g, self.h])
    
    @property
    def essence(self) -> np.ndarray:
        """Full essence vector [x, y, z, e, f, g, h]."""
        return np.array([self.x, self.y, self.z, 
                         self.e, self.f, self.g, self.h])
    
    @property
    def function(self) -> np.ndarray:
        """Function vector [fx, fy, fz, fe, ff, fg, fh]."""
        return np.array([self.fx, self.fy, self.fz,
                         self.fe, self.ff, self.fg, self.fh])
    
    def angle_to(self, other: 'SemanticConcept', space: str = 'core') -> float:
        """Calculate angle to another concept in specified space."""
        if space == 'core':
            u, v = self.core, other.core
        elif space == 'domain':
            u, v = self.domain, other.domain
        elif space == 'essence':
            u, v = self.essence, other.essence
        else:
            raise ValueError(f"Unknown space: {space}")
        
        norm_u, norm_v = np.linalg.norm(u), np.linalg.norm(v)
        if norm_u < 1e-10 or norm_v < 1e-10:
            return 0.0  # Unity concepts
        
        cos_theta = np.dot(u, v) / (norm_u * norm_v)
        return np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
```

### Validation Suite

```python
def validate_dictionary(concepts: List[SemanticConcept]) -> Dict:
    """Run full validation on a concept dictionary."""
    
    results = {
        'total_concepts': len(concepts),
        'complement_pairs': [],
        'complement_validity': 0,
        'mean_core_angle': 0,
        'mean_domain_angle': 0,
    }
    
    # Find all complement pairs
    for c in concepts:
        if c.relations and 'complement' in c.relations:
            for comp_name in c.relations['complement']:
                comp = next((x for x in concepts if x.name == comp_name), None)
                if comp:
                    core_ang = c.angle_to(comp, 'core')
                    domain_ang = c.angle_to(comp, 'domain')
                    valid = 80 <= core_ang <= 105
                    
                    results['complement_pairs'].append({
                        'pair': (c.name, comp_name),
                        'core_angle': core_ang,
                        'domain_angle': domain_ang,
                        'valid': valid
                    })
    
    # Calculate statistics
    if results['complement_pairs']:
        valid_count = sum(1 for p in results['complement_pairs'] if p['valid'])
        results['complement_validity'] = valid_count / len(results['complement_pairs'])
        results['mean_core_angle'] = np.mean([p['core_angle'] for p in results['complement_pairs']])
        results['mean_domain_angle'] = np.mean([p['domain_angle'] for p in results['complement_pairs']])
    
    return results
```

### Quick Reference Functions

```python
def witness_preservation(theta_degrees: float) -> float:
    """Calculate witness preservation from angle."""
    return 1.0 - np.cos(np.radians(theta_degrees))

def angle_to_distance(theta_degrees: float) -> float:
    """Convert angle to Euclidean distance (unit vectors)."""
    return 2 * np.sin(np.radians(theta_degrees) / 2)

def distance_to_angle(distance: float) -> float:
    """Convert Euclidean distance to angle (unit vectors)."""
    return np.degrees(2 * np.arcsin(distance / 2))
```

---

## Appendix A: Trigram Mapping

The eight I Ching trigrams map to domain-polarity combinations:

| Trigram | Symbol | Domain | Polarity | Encoding Pattern |
|---------|--------|--------|----------|------------------|
| QIAN | ☰ | Spatial | Yang | High e, x > 0 |
| KUN | ☷ | Spatial | Yin | High e, x < 0 |
| ZHEN | ☳ | Temporal | Yang | High f, x > 0 |
| XUN | ☴ | Temporal | Yin | High f, x < 0 |
| LI | ☲ | Relational | Yang | High g, x > 0 |
| KAN | ☵ | Relational | Yin | High g, x < 0 |
| DUI | ☱ | Personal | Yang | High h, x > 0 |
| GEN | ☶ | Personal | Yin | High h, x < 0 |

---

## Appendix B: Relationship Angle Targets

| Relation | Core Target | Domain Target | Notes |
|----------|-------------|---------------|-------|
| SYNONYM | 0-15° | 0-15° | Nearly identical |
| AFFINITY | 15-45° | 0-45° | Same family |
| ADJACENT | 45-75° | Variable | Bordering |
| COMPLEMENT | 80-105° | 0-45° | Orthogonal polarity, shared field |
| OPPOSITION | >105° | Variable | True conflict |

---

## Appendix C: Key Constants

| Constant | Value | Meaning |
|----------|-------|---------|
| √2 | 1.414 | Distance between complements |
| π/2 | 1.571 rad = 90° | Complement angle |
| 1/7 | 0.143 | Random interference (7D) |
| 1/3 | 0.333 | Random interference (3D) |

---

*"Structure survives extreme superposition when encoded by design."*
