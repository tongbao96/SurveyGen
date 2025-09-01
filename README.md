<h2 align="center">🚧 <b>This project is continuously updating. Please check back soon!</b> 🚧</h2>



# SurveyGen: Quality-Aware Scientific Survey Generation with Large Language Models



This is the official repository for the dataset and code of the paper:  ["SurveyGen: Quality-Aware Scientific Survey Generation with Large Language Models"](https://arxiv.org/abs/2508.17647),  *accepted at **EMNLP 2025** (Main Conference)*.


## 📂 SurveyGen

Our dataset is constructed based on resources from [S2ORC](https://allenai.org/data/s2orc) (Lo et al., 2020) and [OpenAlex](https://openalex.org/) (Priem et al., 2022).  
The dataset can be accessed at:  [SurveyGen (Google Drive)](https://drive.google.com/drive/folders/1ky6FAd2rs9XPjmOrTMScPbPu_tBv4veh?usp=sharing)  

It contains three files:  

- **survey_full_text**: Parsed full texts of the surveys.  
- **references_for_surveys**: Metadata of the references directly cited in the surveys (named *first-level references*). 
- **second_level_references**: Metadata of the references cited by the first-level references.  



## 🛠️ Preparation Before Starting

Before using the SurveyGen framework, please ensure you have the following resources ready:

1. **Semantic Scholar API** — Apply for an API key at [Semantic Scholar API](https://www.semanticscholar.org/product/api#api-key).  
2. **S2ORC metadata** — Download the full S2ORC metadata to your local environment from [S2ORC](https://api.semanticscholar.org/api-docs/).
3. **LLM API access** — Apply for access to the LLMs (e.g., [OpenAI](https://platform.openai.com/), [Google Gemini](https://ai.google/discover/gemini/), or [Anthropic Claude](https://www.anthropic.com/claude)).  



## 💻 Code for QUAL-SG


## 📜 License

SurveyGen is released under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/legalcode) license.  The dataset follows the same policy as [S2ORC](https://allenai.org/data/s2orc) and [OpenAlex](https://openalex.org/): **for non-commercial academic research use only**.


## 📖 References

If you use this dataset, please cite the following works:

```bibtex

@inproceedings{surveygen,
  author    = {Tong Bao and Mir Tafseer Nayeem and Davood Rafiei and Chengzhi Zhang},
  title     = {SurveyGen: Quality-Aware Scientific Survey Generation with Large Language Models},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2025},
  address   = {Suzhou, China}
}

@inproceedings{s2orc,
  title     = {S2ORC: The Semantic Scholar Open Research Corpus},
  author    = {Lo, Kyle and Wang, Lucy Lu and Neumann, Mark and Kinney, Rodney and Weld, Daniel S.},
  booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  year      = {2020},
  url       = {https://aclanthology.org/2020.acl-main.447},
  doi       = {10.18653/v1/2020.acl-main.447}
}

@inproceedings{openalex,
  author    = {Jason Priem and Heather Piwowar and Richard Orr},
  title     = {OpenAlex: A fully-open index of scholarly works, authors, venues, institutions, and concepts},
  booktitle = {Proceedings of the 26th International Conference on Science, Technology and Innovation Indicators (STI 2022)},
  year      = {2022}
}

