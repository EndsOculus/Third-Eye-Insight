# Attribution and third-party notices

## Original project license and authorship

The root [LICENSE](LICENSE) restores the original MIT text byte-for-byte from commit `9965821` (initial commit). The same text also appeared at `old_beta2_tmp/Third-Eye-Insight-Beta2.0.0/LICENSE` in commit `ae8cc9c`; that temporary tree was removed in `2d44d85`. Its copyright remains **Copyright (c) 2025 Ends_Oculus**. The historical README at `2d44d85` and published Beta2.4 stated MIT in both language sections. No replacement license, new copyright holder or changed year is introduced.

The public repository identifies EndsOculus as its contributor; individual work remains attributed by Git history. The README's original Touhou Project / Komeiji Koishi name inspiration is retained without carrying forward unsupported claims about measuring real relationships. No third-party character artwork is bundled.

## Preserved acknowledgements

All nine acknowledgements from the historical Chinese/English README are retained, without removing the historical external-service credit:

- Pandas — https://pandas.pydata.org/
- NumPy — https://numpy.org/
- Matplotlib — https://matplotlib.org/
- Seaborn — https://seaborn.pydata.org/
- NetworkX — https://networkx.org/
- PyTorch — https://pytorch.org/
- SentenceTransformers — https://www.sbert.net/
- SciPy — https://www.scipy.org/
- DeepSeek API — https://api.deepseek.com/

DeepSeek API belongs to the historical optional report route, not the current default desktop/CLI. Acknowledgement of a service is not a grant to redistribute that service or its models.

## Dependency license check

The following is a record of locally installed distribution metadata inspected on 2026-10-19. Generic BSD/Apache/PSF labels below are exactly the metadata's level of specificity, not a claim that every bundled subcomponent has one uniform license. Consult the actual distribution's license/notice files before redistributing its binaries. This source release does not bundle third-party dependency binaries, model weights or the optional llama.cpp/CUDA runtime.

| Distribution | Inspected version | License metadata |
|---|---|---|
| pandas | 3.0.1 | BSD classification |
| numpy | 2.4.3 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| matplotlib | 3.10.8 | Python Software Foundation classification |
| seaborn | 0.13.2 | BSD classification |
| networkx | 3.6.1 | BSD-3-Clause |
| torch | 2.8.0 | BSD classification |
| sentence-transformers | 5.7.0 | Apache-2.0 |
| scipy | 1.17.1 | BSD classification |
| openai | 2.30.0 | Apache classification |
| pillow | 12.3.0 | MIT-CMU |
| transformers | 5.10.4 | Apache 2.0 License |
| anyio | 4.14.2 | MIT |
| setuptools | 83.0.0 | MIT |
| idna | 3.20 | BSD-3-Clause |
| scikit-learn | 1.8.0 | BSD-3-Clause |

This checks direct Python distributions, not every transitive/native component or optional model license. Project MIT terms do not relicense dependencies, models, CUDA libraries, services, names or artwork. Model/runtime reuse requires its own applicable terms; no rights to redistribute those resources are asserted here.
