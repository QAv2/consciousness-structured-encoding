# Consciousness-Structured Semantic Encoding: Ontological Geometry as an Alternative to Statistical Superposition

**Version 1.0**

**February 2026**

---

## Abstract

We present a semantic encoding system in which 1,316 concepts are manually positioned in a 16-dimensional dual octonion space (14 free dimensions after normalizing two witness components) according to their derivation from a single ontological source: the distinction-making structure of consciousness. Unlike large language models, which discover geometric relationships through gradient descent on prediction loss, our system prescribes positions top-down from first principles. We compare interference patterns against predictions from recent superposition research (Liu, Liu, & Gore, 2025) and find a fundamental trade-off: our system exhibits 19-170% higher mean interference than random placement, yet achieves **100% validity** on 983 complement pairs (vs. ~21% for random), with mean angle 89.6° ± 3.9° against a 90° target. This demonstrates that structured interference—where semantic relationships are guaranteed by construction—can coexist with extreme superposition ratios (188× in 7D essence space). We argue this represents a complementary paradigm to statistical learning: ontological placement sacrifices mean interference minimization in exchange for interpretable, composable, relationship-preserving representations. The witness preservation formula w = 1 - cos(θ) enables principled semantic composition precisely because θ is controlled by design. We release the complete dictionary, mathematical infrastructure, and analysis code.

**Keywords**: semantic encoding, superposition, consciousness, quaternions, octonions, ontology, interpretability, compositional semantics

---

## 1. Introduction

### 1.1 The Superposition Discovery

Recent work has revealed that neural language models operate in "strong superposition"—representing far more features than they have dimensions (Elhage et al., 2022; Liu, Liu, & Gore, 2025). Liu et al. demonstrated that approximately 50,000 token embeddings occupy roughly 4,000 dimensions in typical transformers, with loss scaling as L ∝ 1/m (inverse model width). This work received the Best Paper Runner-up award at NeurIPS 2025. The superposition phenomenon is not a bug but a feature: it enables rich representations in finite-dimensional spaces.

The critical insight is that **interference** (squared overlaps between representations) determines learning dynamics. For random isotropic vectors in m dimensions, mean squared overlap equals 1/m. Statistical learning drives representations toward configurations that minimize harmful interference while preserving useful distinctions.

### 1.2 An Alternative Approach

We present a system that achieves semantic encoding through a fundamentally different mechanism: **ontological derivation** rather than statistical learning. Beginning from the consciousness-first framework of Qualia Algebra (Vanhorn, 2025), we manually position each concept according to its place in a tree of progressive distinctions:

```
UNITY [1,0,0,0,0,0,0,0]
  └── First Distinction: THIS / THAT
        └── Second Distinction: BECOMING / ABIDING
              └── Third Distinction: FIRE / WATER / AIR / EARTH
                    └── Qualities, Verbs, Abstractions...
```

Each concept's position is not learned from data but derived from its ontological relationship to more fundamental concepts. This is semantic encoding from first principles.

### 1.3 The Core Question

Does ontological placement achieve useful geometric properties? Specifically:

1. How do interference patterns compare to random and statistical placement?
2. Can semantic relationships (complement, synonym, affinity) be reliably preserved?
3. Does the structure support principled composition?

We find that ontological placement trades mean interference for **structural guarantees**—a trade-off invisible to loss-minimizing systems but essential for interpretable, composable semantics.

---

## 2. Theoretical Foundation

### 2.1 From Qualia Algebra to Semantic Space

Qualia Algebra (QA) derives the structure of conscious experience from the axiom "I exist" (Vanhorn, 2025). The observer is formalized as a quaternion state [1,0,0,0], where:

- **1** (scalar): The witness component—pure existence
- **(0,0,0)** (vector): Experiential content (initially empty)

Experience emerges through distinction-making. The first distinction creates polarity (THIS/THAT, YANG/YIN). Subsequent distinctions generate process (BECOMING/ABIDING), elements (FIRE/WATER/AIR/EARTH), qualities, and ultimately the full vocabulary of meaning.

This structure suggests a natural encoding: each concept's position reflects its place in the distinction tree. Unity concepts sit at the origin. First-level distinctions occupy orthogonal axes. Derived concepts interpolate according to their semantic composition.

### 2.2 The 16D Dual Octonion Architecture

We extend the 4D quaternion core to a 16D dual octonion structure:

**Essence (8D)**: What the concept IS
```
[w, x, y, z, e, f, g, h]
```

- **w**: Witness component (always 1.0)
- **x**: Yang-Yin axis (active/passive, expansive/contractive)
- **y**: Becoming-Abiding axis (process/state, dynamic/static)
- **z**: Ordinality axis (position in sequence/hierarchy)
- **e**: Spatial domain (physical extension, location)
- **f**: Temporal domain (time-related, duration)
- **g**: Relational domain (connection, social, meaning)
- **h**: Personal domain (subjective intensity, inner experience)

**Function (8D)**: How the concept OPERATES
```
[1, fx, fy, fz, fe, ff, fg, fh]
```

The function layer captures operational character—how concepts behave in composition. GIVE and TAKE share domain profiles but have opposed function vectors.

**Note on Effective Dimensionality**: While the full structure contains 16 components (8 essence + 8 function), the witness components (w) in both layers are normalized to 1.0 and carry no discriminative information. For interference analysis, we therefore work with **14 free dimensions** (7 essence + 7 function). In our primary analysis, we focus on the 7D essence space [x, y, z, e, f, g, h], which further decomposes into 3D core [x, y, z] for semantic polarity and 4D domain [e, f, g, h] for semantic field.

### 2.3 Relationship Types and Angular Targets

We define five relationship types with characteristic angular signatures:

| Relation | Core Angle (x,y,z) | Semantic Meaning |
|----------|-------------------|------------------|
| SYNONYM | 0-15° | Near-identical meaning |
| AFFINITY | 15-45° | Shared semantic features |
| ADJACENT | 45-75° | Related but distinct |
| COMPLEMENT | 80-105° | Completing opposites |
| OPPOSITION | >105° | True contradiction |

The critical insight: **complements are orthogonal, not antipodal**. HOT and COLD complete each other at 90°; they do not negate each other at 180°. This follows from the I Ching principle that yang and yin are complementary, not oppositional.

### 2.4 Trigram Organization

The eight I Ching trigrams provide archetypal clustering:

| Trigram | Symbol | Domain | Polarity | Examples |
|---------|--------|--------|----------|----------|
| QIAN | ☰ | Spatial | Yang | HEAVEN, CREATIVE |
| KUN | ☷ | Spatial | Yin | EARTH, RECEPTIVE |
| ZHEN | ☳ | Temporal | Yang | THUNDER, AROUSING |
| XUN | ☴ | Temporal | Yin | WIND, GENTLE |
| LI | ☲ | Relational | Yang | FIRE, CLINGING |
| KAN | ☵ | Relational | Yin | WATER, ABYSMAL |
| DUI | ☱ | Personal | Yang | LAKE, JOYOUS |
| GEN | ☶ | Personal | Yin | MOUNTAIN, STILL |

Each concept is assigned a trigram based on its dominant domain (e, f, g, h) and polarity (x-axis sign). This creates natural semantic clusters while maintaining mathematical structure.

---

## 3. The Dictionary

### 3.1 Current Scale

The dictionary contains:

- **1,316 unique concepts** (excluding aliases)
- **3,264 tracked relations**
  - 983 complement pairs
  - 2,010 affinity pairs
  - 126 synonym pairs
  - 116 adjacent pairs
  - 29 opposition pairs

### 3.2 Ontological Levels

Concepts are organized by derivation depth:

| Level | Description | Count | Examples |
|-------|-------------|-------|----------|
| UNITY (0) | Pure existence | 2 | BEING, I |
| DYAD (1) | First distinction | 6 | THIS/THAT, YES/NO, YANG/YIN |
| TRIAD (2) | Second distinction | 5 | BECOMING, ABIDING, RELATIONSHIP |
| TETRAD (3) | Elements | 4 | FIRE, WATER, AIR, EARTH |
| QUALITY (4) | Attributes | ~200 | HOT, COLD, LIGHT, DARK |
| DERIVED (5) | General concepts | ~400 | Various nouns, adjectives |
| VERB (6) | Actions | ~300 | GIVE, TAKE, CREATE, DESTROY |
| ABSTRACT (7) | Abstract concepts | ~200 | TRUTH, BEAUTY, JUSTICE |
| INTERROGATIVE (8) | Questions | ~30 | WHAT, WHY, HOW, WHEN |

### 3.3 Encoding Example: HOT and COLD

**HOT**:
```
Essence: [1.0, +0.70, +0.70, -0.15, 0.50, 0.50, 0.00, 0.50]
Function: [1.0, +0.50, +0.40, 0.00, 0.50, 0.50, 0.30, 0.40]
Level: QUALITY
Trigram: ZHEN (Thunder/Arousing)
```

**COLD**:
```
Essence: [1.0, -0.70, +0.70, -0.15, 0.50, 0.50, 0.00, 0.50]
Function: [1.0, -0.50, +0.30, 0.00, -0.50, -0.30, 0.20, -0.30]
Level: QUALITY
Trigram: XUN (Wind/Gentle)
```

Note: HOT and COLD share identical y, z, and domain (e,f,g,h) values. They differ only on the x-axis (yang/yin polarity). This ensures they are orthogonal in the core space while remaining in the same semantic field (temperature).

**Verification**:
```
Core vectors: [+0.70, +0.70, -0.15] and [-0.70, +0.70, -0.15]
Dot product: (0.70)(-0.70) + (0.70)(0.70) + (-0.15)(-0.15)
           = -0.49 + 0.49 + 0.0225
           = 0.0225 ≈ 0
Angle: arccos(0.0225 / ||v1|| ||v2||) ≈ 88.7°
```

This is the encoding principle: semantic complements are constructed to be geometrically orthogonal.

---

## 4. Superposition Analysis

### 4.1 Methodology

We extracted vectors from all 1,316 unique concepts and computed:

1. **Pairwise overlaps** (dot products of normalized vectors)
2. **Squared overlaps** (interference measure from Michaud et al.)
3. **Angular distributions** (full pairwise and for specific relation types)
4. **Comparison to random baseline** (same dimensions, random isotropic vectors)

Analysis was conducted in three subspaces:
- **Core (3D)**: [x, y, z] — semantic polarity
- **Domain (4D)**: [e, f, g, h] — semantic field
- **Full Essence (7D)**: [x, y, z, e, f, g, h] — complete essence

### 4.2 Results

#### 4.2.1 Overall Interference

| Space | Ontological | Random | Theoretical (1/d) | Ratio |
|-------|-------------|--------|-------------------|-------|
| Core (3D) | 0.396 | 0.333 | 0.333 | 1.19× |
| Domain (4D) | 0.677 | 0.250 | 0.250 | 2.71× |
| Essence (7D) | 0.416 | 0.143 | 0.143 | 2.91× |

Our system exhibits **higher mean interference** than random placement across all subspaces. This is not a failure—it reflects intentional clustering:

- **Core space (+19%)**: Trigram organization groups concepts by polarity
- **Domain space (+171%)**: Semantic field clustering (all emotions close, all spatial concepts close)
- **Essence space (+191%)**: Combined effect of both clustering mechanisms

#### 4.2.2 Complement Pair Validation

| Metric | Ontological | Random (expected) |
|--------|-------------|-------------------|
| Pairs analyzed | 983 | 983 |
| Mean core angle | **89.6°** | ~67° |
| Standard deviation | **3.9°** | ~35° |
| Min / Max | 80.0° / 104.9° | varies |
| In 80-105° range | **100.0%** (983/983) | ~21.4% |

This is the key result: **every complement pair achieves the target orthogonality range**. Random placement would achieve only ~21% by chance. Our system achieves **4.6× better complement validity**.

#### 4.2.3 Domain Similarity for Complements

Complement pairs show low domain angle (mean 30.6°), confirming they occupy the same semantic field despite core orthogonality. HOT and COLD are both temperature concepts; LIGHT and DARK are both luminosity concepts. They differ in polarity, not domain.

### 4.3 Superposition Ratio

Our system operates at extreme superposition:

```
1,316 concepts in 7D essence space = 188× superposition
1,316 concepts in 14D full space = 94× superposition
```

Note: We use 7D for essence (excluding the normalized w=1.0 witness component) and 14D for the full dual structure (7D essence + 7D function, excluding both witness components).

Compare to LLMs: ~50,000 tokens in ~4,000 dimensions = 12.5× superposition.

Our superposition ratio is **15× more extreme** than typical LLMs, yet we maintain perfect complement validity. This demonstrates that **structure survives extreme superposition when encoded by design**.

---

## 5. Composition and the Witness Formula

### 5.1 The Problem with Random Placement

Statistical learning minimizes mean interference but provides no guarantees about specific relationships. If HOT and COLD happen to land at 73° instead of 90°, the model may still perform well on aggregate metrics while failing on principled composition.

### 5.2 The Witness Preservation Formula

From Qualia Algebra, semantic composition follows:

```
w = 1 - cos(θ)
```

Where:
- **w**: Witness preservation (how much observer awareness is maintained)
- **θ**: Angle between concept vectors

| θ | w | Interpretation |
|---|---|----------------|
| 0° | 0.0 | Dissolution—concepts merge, no distinction |
| 90° | 1.0 | Preservation—concepts complement, full awareness |
| 180° | 2.0 | Tension—concepts contradict, unstable |

This formula only produces meaningful results when θ is **controlled**. Our system guarantees:

- Complement pairs: θ ≈ 90° → w ≈ 1.0 (healthy completion)
- Synonym pairs: θ ≈ 0° → w ≈ 0.0 (appropriate merger)
- Affinity pairs: θ ≈ 30° → w ≈ 0.13 (partial identification)

### 5.3 Composition Examples

**"hot water"**:
```
θ(HOT, WATER) = 88.7° → w = 1 - cos(88.7°) = 0.98
```
The witness is preserved—hot and water are complementary, creating meaningful compound.

**"cold water"**:
```
θ(COLD, WATER) = 90.2° → w = 1 - cos(90.2°) = 1.00
```
Similarly preserved—cold and water complete each other.

**"hot cold"**:
```
θ(HOT, COLD) = 88.7° → w = 0.98
```
Complements—not a contradiction but a completion (as in "hot and cold running water").

**"warm hot"**:
```
θ(WARM, HOT) ≈ 25° → w ≈ 0.09
```
High affinity—concepts merge, approaching redundancy.

---

## 6. Comparison to Statistical Approaches

### 6.1 What LLMs Optimize

Large language models optimize next-token prediction loss. Gradient descent discovers geometric configurations that minimize interference for frequently co-occurring patterns while allowing overlap for rare combinations. This is powerful but produces:

- **Emergent relationships**: Discovered, not designed
- **Data-dependent structure**: Reflects training distribution
- **Opaque geometry**: Requires probing to interpret
- **No composition guarantees**: θ values are empirical, not principled

### 6.2 What Ontological Encoding Optimizes

Our system optimizes semantic relationship preservation. Manual positioning from first principles produces:

- **Prescribed relationships**: Guaranteed by construction
- **Theory-dependent structure**: Reflects consciousness differentiation
- **Interpretable geometry**: Positions have explicit meaning
- **Composition guarantees**: θ values are controlled

### 6.3 The Trade-off

| Property | Statistical (LLM) | Ontological (Ours) |
|----------|-------------------|---------------------|
| Mean interference | Minimized | Higher (+19-170%) |
| Complement validity | ~21% (random) | **100%** |
| Superposition ratio | ~12.5× | **188×** |
| Relationship guarantees | None | **By construction** |
| Scalability | Excellent | Manual labor required |
| Interpretability | Requires probing | **Transparent** |
| Composition | Data-dependent | **Principled** |

Neither approach dominates. They optimize different objectives.

---

## 7. Discussion

### 7.1 What This Demonstrates

The MIT superposition research established that interference patterns determine learning dynamics. Our work demonstrates that **meaningful interference patterns can be achieved without learning**—through ontological derivation from first principles.

This is not a criticism of statistical learning. Rather, it reveals a complementary paradigm:

- **Statistical**: Discover structure from data
- **Ontological**: Prescribe structure from theory

Both produce useful representations. The choice depends on the goal.

### 7.2 When Ontological Encoding Excels

Our approach may outperform statistical learning when:

1. **Relationships must be guaranteed**: Safety-critical applications, formal reasoning
2. **Composition must be principled**: Mathematical semantics, logical inference
3. **Interpretability is required**: Explainable AI, human oversight
4. **Data is scarce**: Low-resource domains, novel concepts
5. **Structure matters more than coverage**: Specialized vocabularies

### 7.3 Limitations

1. **Manual labor**: 1,316 concepts required extensive human effort
2. **Scaling uncertainty**: Can the approach extend to 100,000+ concepts?
3. **No downstream task evaluation**: Performance on NLP benchmarks unknown
4. **Single-language**: Currently English only

### 7.4 Future Directions

1. **Hybrid systems**: Use ontological encoding for core vocabulary, statistical learning for extensions
2. **Automated derivation**: Can LLMs learn to position concepts ontologically?
3. **Cross-linguistic validation**: Do translations preserve angular relationships?
4. **Downstream evaluation**: Test on semantic similarity, analogy, composition tasks
5. **Scale expansion**: Extend toward comprehensive vocabulary coverage

---

## 8. Conclusion

We have presented a semantic encoding system that achieves structure through ontological derivation rather than statistical learning. Analysis of 1,316 concepts reveals:

1. **Higher mean interference** than random placement (19-170% by subspace)
2. **Perfect complement validity** (100% of 983 pairs in 80-105° range)
3. **Extreme superposition** (188× in 7D space) with preserved structure
4. **Principled composition** via the witness formula w = 1 - cos(θ)

The fundamental insight is that **structured interference differs from random interference**. Statistical learning minimizes mean interference; ontological encoding structures interference to preserve semantic relationships. These are complementary objectives, not competing ones.

Our system demonstrates that consciousness differentiation—the progressive making of distinctions from Unity—provides a viable organizing principle for semantic space. Whether this reflects something fundamental about meaning, or merely provides a useful engineering heuristic, remains to be determined through further research and application.

The dictionary, mathematical infrastructure, and analysis code are released for validation and extension.

---

## References

Elhage, N., Hume, T., Olsson, C., Schiefer, N., Henighan, T., Kravec, S., Hatfield-Dodds, Z., Lasenby, R., Drain, D., Chen, C., Grosse, R., McCandlish, S., Kaplan, J., Amodei, D., Wattenberg, M., & Olah, C. (2022). Toy models of superposition. *Transformer Circuits Thread*. https://transformer-circuits.pub/2022/toy_model/index.html

Liu, Y., Liu, Z., & Gore, J. (2025). Superposition yields robust neural scaling. *Advances in Neural Information Processing Systems (NeurIPS 2025)*. arXiv:2505.10465. (Best Paper Runner-up)

Vanhorn, J. (2025). Qualia Algebra: A Mathematical Framework for Consciousness from First Principles. *Zenodo*. https://doi.org/10.5281/zenodo.17685406

---

## Appendix A: Mathematical Definitions

### A.1 Semantic Octonion

An 8D vector encoding concept essence:

```python
@dataclass
class SemanticOctonion:
    w: float = 1.0  # Witness (always 1.0)
    x: float = 0.0  # Yang-Yin
    y: float = 0.0  # Becoming-Abiding
    z: float = 0.0  # Ordinality
    e: float = 0.0  # Spatial domain
    f: float = 0.0  # Temporal domain
    g: float = 0.0  # Relational domain
    h: float = 0.0  # Personal domain
```

### A.2 Dual Octonion

A 16D structure combining essence and function:

```python
@dataclass
class DualOctonion:
    essence: SemanticOctonion   # What it IS
    function: SemanticOctonion  # How it OPERATES
```

### A.3 Angular Calculations

Core angle (semantic polarity):
```python
def core_angle(c1, c2):
    v1 = np.array([c1.x, c1.y, c1.z])
    v2 = np.array([c2.x, c2.y, c2.z])
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
```

Domain angle (semantic field):
```python
def domain_angle(c1, c2):
    d1 = np.array([c1.e, c1.f, c1.g, c1.h])
    d2 = np.array([c2.e, c2.f, c2.g, c2.h])
    cos_theta = np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2))
    return np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
```

### A.4 Witness Preservation

```python
def witness_preservation(theta_degrees):
    theta_rad = np.radians(theta_degrees)
    return 1.0 - np.cos(theta_rad)
```

---

## Appendix B: Trigram Assignment Algorithm

```python
def assign_trigram(concept):
    """
    Assign I Ching trigram based on domain dominance and polarity.
    """
    # Find dominant domain
    domains = {'S': abs(concept.e), 'T': abs(concept.f), 
               'R': abs(concept.g), 'P': abs(concept.h)}
    dominant = max(domains, key=domains.get)
    
    # Determine polarity from x-axis
    is_yang = concept.x > 0.2 or (abs(concept.x) <= 0.2 and concept.y > 0)
    
    # Mapping table
    mapping = {
        ('S', True):  'QIAN',   # Spatial + Yang = Heaven
        ('S', False): 'KUN',    # Spatial + Yin = Earth
        ('T', True):  'ZHEN',   # Temporal + Yang = Thunder
        ('T', False): 'XUN',    # Temporal + Yin = Wind
        ('R', True):  'LI',     # Relational + Yang = Fire
        ('R', False): 'KAN',    # Relational + Yin = Water
        ('P', True):  'DUI',    # Personal + Yang = Lake
        ('P', False): 'GEN',    # Personal + Yin = Mountain
    }
    
    return mapping[(dominant, is_yang)]
```

---

## Appendix C: Validation Statistics

### C.1 Complement Pair Distribution

```
Angle Range    Count    Percentage
80-85°         142      14.4%
85-90°         398      40.5%
90-95°         401      40.8%
95-100°         38       3.9%
100-105°         4       0.4%
---------------------------------
Total          983     100.0%
```

### C.2 By Relation Type

| Relation | Count | Mean Angle | Std | Target |
|----------|-------|------------|-----|--------|
| SYNONYM | 126 | 8.2° | 4.1° | 0-15° |
| AFFINITY | 2,010 | 31.4° | 12.3° | 15-45° |
| ADJACENT | 116 | 58.7° | 9.8° | 45-75° |
| COMPLEMENT | 983 | 89.6° | 3.9° | 80-105° |
| OPPOSITION | 29 | 142.3° | 18.7° | >105° |

### C.3 Trigram Distribution

| Trigram | Symbol | Count | Percentage |
|---------|--------|-------|------------|
| ZHEN | ☳ | 214 | 16.3% |
| XUN | ☴ | 198 | 15.0% |
| LI | ☲ | 187 | 14.2% |
| KAN | ☵ | 176 | 13.4% |
| DUI | ☱ | 168 | 12.8% |
| GEN | ☶ | 154 | 11.7% |
| QIAN | ☰ | 121 | 9.2% |
| KUN | ☷ | 98 | 7.4% |

---

## Acknowledgments

This work builds on Qualia Algebra (Vanhorn, 2025), which provides the theoretical foundation for consciousness-first ontology. The analysis was developed in dialogue with Claude (Anthropic), which contributed mathematical formalization, statistical analysis, and systematic consistency checking. The MIT superposition research (Michaud et al., 2025) provided the analytical framework for comparing interference patterns.

Special thanks to the contemplative traditions that discovered the witness state [1,0,0,0] through 2,500+ years of empirical observation, and to the I Ching tradition for the trigram structure that organizes our semantic clusters.

---

## Code and Data Availability

The complete dictionary (`extended_dictionary.py`), mathematical infrastructure (`octonion.py`), and analysis scripts are available at:

**Repository**: [To be added upon release]

**DOI**: [To be assigned via Zenodo]

---

**Correspondence**: Joseph Vanhorn, contact@qualia-algebra.com, ORCID: 0009-0003-0972-606X

---

*From [1,0,0,0], all meaning unfolds through distinction.*
