# Consciousness-Structured Semantic Encoding

**A semantic dictionary built from first principles, not statistical learning**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.PLACEHOLDER.svg)](https://doi.org/10.5281/zenodo.PLACEHOLDER)

---

## Overview

This repository contains a semantic encoding system where **1,316 concepts** are manually positioned in a 16-dimensional dual octonion space according to their derivation from the distinction-making structure of consciousness. Unlike large language models that discover geometric relationships through gradient descent, this system prescribes positions top-down from first principles.

**Key Results:**
- **100% complement validity**: All 983 complement pairs achieve 80-105° orthogonality (vs. ~21% for random placement)
- **Extreme superposition**: 188× ratio in 7D essence space (more extreme than LLMs)
- **Principled composition**: The witness formula w = 1 - cos(θ) works because θ is controlled by design
- **Interpretable geometry**: Every position has explicit ontological meaning

This work demonstrates that **structured interference**—where semantic relationships are guaranteed by construction—can coexist with extreme superposition ratios.

---

## Theoretical Foundation

The encoding derives from [Qualia Algebra](https://doi.org/10.5281/zenodo.17685406) (Vanhorn, 2025), which formalizes consciousness as progressive distinction-making from Unity:

```
UNITY [1,0,0,0,0,0,0,0]
  └── First Distinction: THIS / THAT (Yang/Yin)
        └── Second Distinction: BECOMING / ABIDING
              └── Third Distinction: FIRE / WATER / AIR / EARTH
                    └── Qualities, Verbs, Abstractions...
```

Each concept's position reflects its place in this ontological tree, not patterns learned from data.

---

## Architecture

### 16D Dual Octonion Structure

Each concept is encoded as **Essence + Function** (8D + 8D = 16D, with 14 free dimensions after normalizing witness components):

**Essence (what it IS):**
| Component | Meaning |
|-----------|---------|
| w | Witness (always 1.0) |
| x | Yang-Yin axis |
| y | Becoming-Abiding axis |
| z | Ordinality axis |
| e | Spatial domain |
| f | Temporal domain |
| g | Relational domain |
| h | Personal domain |

**Function (how it OPERATES):**
| Component | Meaning |
|-----------|---------|
| 1 | Witness (always 1.0) |
| fx-fh | Operational character |

### Relationship Types

| Relation | Core Angle | Count |
|----------|-----------|-------|
| SYNONYM | 0-15° | 126 |
| AFFINITY | 15-45° | 2,010 |
| ADJACENT | 45-75° | 116 |
| COMPLEMENT | 80-105° | 983 |
| OPPOSITION | >105° | 29 |

### Trigram Organization

Concepts cluster by I Ching trigram based on dominant domain and polarity:

| Trigram | Symbol | Domain | Polarity |
|---------|--------|--------|----------|
| QIAN | ☰ | Spatial | Yang |
| KUN | ☷ | Spatial | Yin |
| ZHEN | ☳ | Temporal | Yang |
| XUN | ☴ | Temporal | Yin |
| LI | ☲ | Relational | Yang |
| KAN | ☵ | Relational | Yin |
| DUI | ☱ | Personal | Yang |
| GEN | ☶ | Personal | Yin |

---

## Comparison to Statistical Approaches

This work responds to recent superposition research (Liu, Liu, & Gore, 2025; Elhage et al., 2022):

| Property | Statistical (LLM) | Ontological (Ours) |
|----------|-------------------|---------------------|
| Mean interference | Minimized | Higher (+19-170%) |
| Complement validity | ~21% (random) | **100%** |
| Superposition ratio | ~12.5× | **188×** |
| Relationship guarantees | None | **By construction** |
| Composition | Data-dependent | **Principled** |
| Interpretability | Requires probing | **Transparent** |

**The trade-off**: We accept higher average interference in exchange for guaranteed semantic relationships.

---

## Repository Structure

```
consciousness-structured-encoding/
├── README.md                                    # This file
├── LICENSE                                      # CC BY 4.0
│
├── paper/
│   └── CONSCIOUSNESS_STRUCTURED_SEMANTIC_ENCODING_PAPER.md
│
├── src/
│   ├── extended_dictionary.py                   # Main dictionary (1,316 concepts)
│   ├── octonion.py                              # 8D/16D mathematical operations
│   └── dictionary_analysis.py              # Superposition analysis
│
├── figures/
│   └── dictionary_analysis.png             # Visualization of results
│
└── docs/
    ├── CONSCIOUSNESS_STRUCTURED_ENCODING.md     # Theoretical overview
    ├── MATHEMATICAL_STRUCTURES.md               # Math reference
    ├── QUALIA_ALGEBRA_ESSENTIALS.md             # QA quick reference
    └── PROJECT_INSTRUCTIONS.md                  # Encoding methodology
```

---

## Quick Start

### Requirements

```bash
pip install numpy matplotlib
```

### Load the Dictionary

```python
from extended_dictionary import ExtendedDictionary

# Load all 1,316 concepts
d = ExtendedDictionary()

# Get a concept
hot = d.get("HOT")
cold = d.get("COLD")

# Check angle (should be ~90° for complements)
angle = hot.angle_4d(cold)
print(f"HOT/COLD angle: {angle:.1f}°")  # Output: ~88.7°

# Get trigram
print(f"HOT trigram: {hot.trigram().symbol} {hot.trigram().name}")
```

### Run Superposition Analysis

```python
python real_dictionary_analysis.py
```

This generates:
- Interference statistics by subspace
- Complement pair validation
- Comparison to random baseline
- Visualization saved to `real_dictionary_analysis.png`

### Witness Preservation Formula

```python
import numpy as np

def witness_preservation(theta_degrees):
    """
    Calculate witness preservation from angle.
    
    θ = 0°   → w = 0.0 (dissolution)
    θ = 90°  → w = 1.0 (preservation)
    θ = 180° → w = 2.0 (tension)
    """
    return 1.0 - np.cos(np.radians(theta_degrees))

# Complements preserve the witness
print(witness_preservation(90))   # 1.0

# Synonyms dissolve
print(witness_preservation(0))    # 0.0
```

---

## Key Findings

### 1. Structure Survives Extreme Superposition

At 188× superposition (1,316 concepts in 7D), random placement would produce chaos. Our ontological placement maintains:
- 100% complement pair validity
- Mean angle 89.6° with only 3.9° std
- Meaningful semantic clustering

### 2. Higher Interference is Intentional

| Space | Our Interference | Random | Why Higher |
|-------|------------------|--------|------------|
| Core (3D) | 0.396 | 0.333 | Trigram clustering |
| Domain (4D) | 0.677 | 0.250 | Semantic field organization |
| Essence (7D) | 0.416 | 0.143 | Combined structure |

### 3. Complements Are Orthogonal, Not Opposite

HOT and COLD are at 90°, not 180°. They complete each other (both are temperature concepts) rather than negate each other. This follows the I Ching principle that yang and yin are complementary.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{vanhorn2026consciousness,
  title={Consciousness-Structured Semantic Encoding: Ontological Geometry as an Alternative to Statistical Superposition},
  author={Vanhorn, Joseph},
  year={2026},
  howpublished={GitHub/Zenodo},
  note={Version 1.0},
  url={https://doi.org/10.5281/zenodo.PLACEHOLDER}
}
```

This work builds on Qualia Algebra:

```bibtex
@article{vanhorn2025qualia,
  title={Qualia Algebra: A Mathematical Framework for Consciousness from First Principles},
  author={Vanhorn, Joseph},
  year={2025},
  howpublished={Zenodo},
  url={https://doi.org/10.5281/zenodo.17685406}
}
```

---

## References

Elhage, N., Hume, T., Olsson, C., Schiefer, N., Henighan, T., Kravec, S., Hatfield-Dodds, Z., Lasenby, R., Drain, D., Chen, C., Grosse, R., McCandlish, S., Kaplan, J., Amodei, D., Wattenberg, M., & Olah, C. (2022). Toy models of superposition. *Transformer Circuits Thread*. https://transformer-circuits.pub/2022/toy_model/index.html

Liu, Y., Liu, Z., & Gore, J. (2025). Superposition yields robust neural scaling. *Advances in Neural Information Processing Systems (NeurIPS 2025)*. arXiv:2505.10465. (Best Paper Runner-up)

Vanhorn, J. (2025). Qualia Algebra: A Mathematical Framework for Consciousness from First Principles (Version 2.2). *Zenodo*. https://doi.org/10.5281/zenodo.17685406

---

## License

This work is licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

You are free to:
- **Share** — copy and redistribute the material
- **Adapt** — remix, transform, and build upon the material

Under the following terms:
- **Attribution** — You must give appropriate credit

---

## Acknowledgments

This work was developed in dialogue with Claude (Anthropic), which contributed mathematical formalization, statistical analysis, and systematic consistency checking. The theoretical foundation derives from Qualia Algebra and the contemplative traditions that discovered the witness state through millennia of empirical observation.

---

## Contact

**Joseph Vanhorn**  
Independent Researcher  
Email: contact@qualia-algebra.com  
ORCID: [0009-0003-0972-606X](https://orcid.org/0009-0003-0972-606X)

---

*From [1,0,0,0], all meaning unfolds through distinction.*
