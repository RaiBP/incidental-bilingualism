# Detecting Unintentional Bilingual and Translation Instances in NLP Datasets
This repository contains code developed as part of a research internship at the Machine Learning Professorship at the Technical University of Munich (TUM). The project implements Google's "*Searching for Needles in a Haystack*: On the Role of Incidental Bilingualism in PaLM's Translation Capability" (Briakou et al. 2023) using open source tools.

The codebase is mainly aimed at reproduction of research results detailed in `internship_report.pdf`, but the main script can be used by anyone looking to detect bilingual and translation instances in their NLP datasets.

Some differences with respect to the original paper:
1. Per-token language detection is done with Kevers (2022)'s CoSwID model, instead of Google's CMX model (Zhang et al., 2018).
2. In case the CoSwID model is very unsure over a subsequent series of tokens, we use Facebook's FastText-langdetect (Joulin et al., 2016) to label the entire uncertain sequence.

## Installation
### 1. Install CoSwID as indicated in the CoSwID [repository](https://github.com/lkevers/coswid):
1. Install the pre-requisites.
   
Required libraries:
```
unzip
g++
python3
```
Required Python packages:
```
python-daemon
numpy
fasttext-langdetect
iso-639
levenshtein
```
2. Clone the CoSwID repositories. Let's call the root directory where you want to clone them `<WORKING_DIR>`, then you'd want to do:
```
cd <WORKING_DIR>
git clone https://github.com/lkevers/ldig-python3.git
git clone https://github.com/lkevers/dicServer.git
git clone https://github.com/lkevers/coswid.git
```
2. Generate the language model following the instructions on the CoSwID repository. For example, the FILTER2 model was proposed in the CoSwID paper. It detects English, Italian, German, French, Portuguese, Spanish, Dutch, Romanian and Corsican and works well for our purposes. You can generate FILTER2 (default settings) with: 
```
cd <WORKING_DIR>/coswid/data_lgID_learn
unzip filter2.zip
cat filter2/LEARN_data_filter2_* >>filter2/LEARN_data_filter2_ALL.txt
mkdir ../models/filter2
cd ../../ldig-python3/maxsubst
g++ -o maxsubst maxsubst.cpp -Icybozulib/include
chmod +x maxsubst
cd ../
python3 ldig.py -m ../coswid/models/filter2 -x maxsubst/maxsubst --init ../coswid/data_lgID_learn/filter2/LEARN_data_filter2_ALL.txt
python3 ldig.py -m ../coswid/models/filter2 --learn ../coswid/data_lgID_learn/filter2/LEARN_data_filter2_ALL.txt -e 0.5
python3 ldig.py -m ../coswid/models/filter2 --shrink
```

3. Modify the script `<WORKING_DIR>/coswid/src/coswid.py` as instructed in the CoSwID [repository](https://github.com/lkevers/coswid)'s README. 

4. Run the dicServer with:
```
cd <WORKING_DIR>/dicServer
python3 dicServer.py .
```

6. Test if the coswid.py installation was successful:
```
cd <WORKING_DIR>/coswid/src
python3 coswid.py -m FILTER2 -t "Voici un texte à analyser in order to predict the languages" -c 2 -f 0 -g 0.1 -v dico
```
The results will be written to the default.out file.

Once the `coswid.py` script is working properly, you can move on to the next step.

### 2. Install Required Dependencies
Install the Python packages:
```pip install -r requirements.txt```
## Usage
To detect bilingual and translation instances in your dataset:
```bash
python main.py --repo_id REPO_ID --filename FILENAME [options]

Required arguments:
--repo_id        # HuggingFace dataset repository ID
--filename       # File to process within the repository

Optional arguments:
--max_tokens     # Maximum tokens per instance (default: 1024)
--num_workers    # Number of parallel workers (default: 1) 
--N              # Minimum consecutive tokens threshold (default: 10)
--coswid_model   # CoSwID model name (default: FILTER2)
--coswid_path    # Path to coswid.py (default: ./coswid/src/coswid.py)
--cache_dir      # Cache directory for downloaded files
```

This will identify monolingual, bilingual and translation instances in your dataset using CoSwID for language detection and LABSE for translation detection.
The script outputs results in HuggingFace dataset format containing instance labels and language information, ready for downstream tasks or analysis.


## References
- Briakou, E., Cherry, C., & Foster, G. (2023). Searching for Needles in a Haystack: On the Role of Incidental Bilingualism in PaLM’s Translation Capability. *arXiv preprint* arXiv:2305.10266. [[Link](http://arxiv.org/abs/2305.10266)]
- Kevers, L. (2022). CoSwID, a Code Switching Identification Method Suitable for Under-Resourced Languages. In *Proceedings of 1st Annual Meeting of the ELRA/ISCA Special Interest Group on Under-Resourced Languages (SIGUL 2022)* (pp. 112-121). Marseille, France. [[Link](http://www.lrec-conf.org/proceedings/lrec2022/workshops/SIGUL/pdf/2022.sigul-1.15.pdf)]
- Zhang, Y., Riesa, J., Gillick, D., Bakalov, A., Baldridge, J., & Weiss, D. (2018). A Fast, Compact, Accurate Model for Language Identification of Codemixed Text. *arXiv preprint* arXiv:1810.04142. [[Link](http://arxiv.org/abs/1810.04142)]
- Joulin, A., Grave, E., Bojanowski, P., Douze, M., Jégou, H., & Mikolov, T. (2016). FastText.zip: Compressing text classification models. *arXiv preprint* arXiv:1612.03651. [[Link](https://arxiv.org/abs/1612.03651)]
- Joulin, A., Grave, E., Bojanowski, P., & Mikolov, T. (2016). Bag of tricks for efficient text classification. *arXiv preprint* arXiv:1607.01759. [[Link](https://arxiv.org/abs/1607.01759)]

## License
The source code in this repository falls under the MIT License. For CoSwID's license, please refer to its [repository](https://github.com/lkevers/coswid).
