# Prompt-based fine-tuning experiments for cancer response inference


This repository contains the code and scripts to conduct the prompt-based fine-tuning experiments for cancer response inference.

For software, we build upon the open source code of [ADAPET (Tam et al., 2021)](https://aclanthology.org/2021.emnlp-main.407/) for the experiments in this work. We conduct experiments on the [GatorTron](https://huggingface.co/AshtonIsNotHere/GatorTron-OG/tree/main) model.


**Table of contents**

[Prerequisites](#prerequisites)

[Example](#example)

[License](#license)


## Prerequisites
Go to your experiment directory, run
```
	source setup_env.sh
```
Activate your environment,
```
	source env/bin/activate
```


## Example
Place your data in your home directory, e.g., /home/data/medical. Run prompt-based fine-tuning experiments following
```
	example.sh

```

## License
The source code and models in this repository are licensed under the GNU General Public License v3.0 (see [LICENSE](LICENSE)). For commercial use of this code and models, separate commercial licensing is also available.




