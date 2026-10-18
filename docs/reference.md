---
title: Reference
---

## Governance

DeepEcoHab is developed by Ewelina Knapska's Laboratory of Emotions Neurobiology at the
Nencki Institute of Experimental Biology, Polish Academy of Sciences, Warsaw. The lab sets
the project's scientific direction and decides what goes into each release.

Contributions from outside the lab are welcome:

- **Bugs and requests** go to the
  [issue tracker](https://github.com/DeepEcoHab/DeepEcoHab/issues).
- **Changes** come as pull requests. The maintainers review and merge them once the checks
  pass: linting, type checking and the test suite run on every pull request.
  [CONTRIBUTING.md](https://github.com/DeepEcoHab/DeepEcoHab/blob/main/CONTRIBUTING.md)
  describes the development setup and conventions.

DeepEcoHab is released under the
[MIT license](https://github.com/DeepEcoHab/DeepEcoHab/blob/main/LICENSE).

## Contributors

| name | role |
|---|---|
| Konrad Danielewski ([@KonradDanielewski](https://github.com/KonradDanielewski)) | lead developer and maintainer |
| Ula Włodkowska ([@uwlodkowska](https://github.com/uwlodkowska)) | developer |
| Marcin Lipiec | principal investigator |
| Ewelina Knapska | head of the laboratory |

With contributions from [@Winiarsky](https://github.com/Winiarsky) and
[@Brosnan-neuro](https://github.com/Brosnan-neuro). Everyone who has contributed code is on
the [contributors graph](https://github.com/DeepEcoHab/DeepEcoHab/graphs/contributors).

## How to cite

DeepEcoHab has no paper of its own yet. If you use it in published work, cite the package
[@deepecohab] together with the paper introducing the Eco-HAB system [@puscian2016ecohab].
GitHub's **Cite this repository** button, on the
[repository page](https://github.com/DeepEcoHab/DeepEcoHab), gives the same entry in APA
and BibTeX, read from the repository's `CITATION.cff`.

```bibtex
@software{deepecohab,
  title     = {{DeepEcoHab}: fast and intuitive data analysis platform for {EcoHab} experiments},
  author    = {Danielewski, Konrad and Wlodkowska, Ula and Lipiec, Marcin and Knapska, Ewelina},
  year      = {2026},
  version   = {0.6.1},
  publisher = {GitHub},
  url       = {https://github.com/DeepEcoHab/DeepEcoHab}
}
```

Name the version you used, as `deepecohab.__version__` reports it, so others can repeat
your analysis.
