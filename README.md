# Detecting Unintentional Bilingual and Translation Instances in NLP Datasets
Python implementation of Google's "*Searching for Needles in a Haystack*: On the Role of Incidental Bilingualism in PaLM’s Translation Capability" (Briakou et al. 2023),
using open source tools. Some differences with respect to the paper:
1. Per-token language detection is done with Kevers (2022)'s CoSwID model, instead of Google's CMX model (Zhang et al., 2018).
2. In case the CoSwID model is very unsure over a sequence of tokens, we use Facebook's FastText-langdetect (Joulin et al., 2016) to label the entire uncertain sequence.

## Installation


## Usage

## References
- Briakou, E., Cherry, C., & Foster, G. (2023). Searching for Needles in a Haystack: On the Role of Incidental Bilingualism in PaLM’s Translation Capability. *arXiv preprint* arXiv:2305.10266. [[Link](http://arxiv.org/abs/2305.10266)]
- Kevers, L. (2022). CoSwID, a Code Switching Identification Method Suitable for Under-Resourced Languages. In *Proceedings of 1st Annual Meeting of the ELRA/ISCA Special Interest Group on Under-Resourced Languages (SIGUL 2022)* (pp. 112-121). Marseille, France. [[Link](http://www.lrec-conf.org/proceedings/lrec2022/workshops/SIGUL/pdf/2022.sigul-1.15.pdf)]
- Zhang, Y., Riesa, J., Gillick, D., Bakalov, A., Baldridge, J., & Weiss, D. (2018). A Fast, Compact, Accurate Model for Language Identification of Codemixed Text. *arXiv preprint* arXiv:1810.04142. [[Link](http://arxiv.org/abs/1810.04142)]
- Joulin, A., Grave, E., Bojanowski, P., Douze, M., Jégou, H., & Mikolov, T. (2016). FastText.zip: Compressing text classification models. *arXiv preprint* arXiv:1612.03651. [[Link](https://arxiv.org/abs/1612.03651)]
- Joulin, A., Grave, E., Bojanowski, P., & Mikolov, T. (2016). Bag of tricks for efficient text classification. *arXiv preprint* arXiv:1607.01759. [[Link](https://arxiv.org/abs/1607.01759)]
