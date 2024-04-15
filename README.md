# Climate Language Processing

Author(s): Bryan Flores & Vivian Do
## Introduction
Climate change, also often referred to as the climate crisis, has garnered significant attention in recent decades due to its profound impact on the environment, ecosystems, and human societies worldwide. Scientific consensus overwhelmingly supports that climate change is primarily driven by human activities. 

Despite the overwhelming scientific consensus on the reality and severity of climate change, a significant portion of the population remains skeptical and even dismissive of the issue. Recent studies indicate that approximately one in seven Americans do not believe that a climate crisis exists (Dewan, 2024). This skepticism can be attributed, at least in part, to the diverse and often contradictory messaging surrounding climate change in the media. The framing of climate change narratives and the selection of topics for coverage both contribute to shaping public perceptions and attitudes. 

Adaptation of language used for climate change communication so that it's more receptive to different audiences. The goal is *not* to manipulate, but to create a path towards an object goal between different peoples. 

## Installation

### Pre-requisites

For development, verify [Pipenv](https://pipenv.pypa.io/en/latest/) setup.

Install Pipenv

If Pipenv is not yet installed, please see [Pipenv Installation](https://pipenv.pypa.io/en/latest/installation.html#installing-packages-for-your-project). If using Windows, also see [Pipenv Windows Installation](https://www.pythontutorial.net/python-basics/install-pipenv-windows/)

Install the packages

``` pipenv install ```

Activate the virtual environment

``` pipenv shell ```

Open notebook

``` jupyter lab ```

Exit the virtual environment

``` exit ```

## Usage

### GDELT Project Dataset

Data was downloaded from the [The GDELT Project](https://blog.gdeltproject.org/a-new-dataset-for-exploring-climate-change-narratives-on-television-news-2009-2020/) website. If you seeking to perform your own analysis, store the CSVs in the [data folder](./scripts/data).

### Analysis & Modeling

First, activate the virtual environment and open Jupyter Lab or any IDE of your choosing. 

Next, 0pen the [Main notebook](main.ipynb) for exploratory analysis and descriptive statistics.

Once you have completed the initial analysis, both methods of topic modeling will be available in their respective notebooks. 

#### Reminders

1. LDA Topic Modeling could take over 20 minutes per task. Ensure the proper resources are available.
2. Running models with ChatGPT require an API key. If one is not available, LLM Topic Modeling will not work.

## References

Dewan, P. (2024, February 28). Map Shows Climate Change Denialism by US State. 	Newsweek. Retrieved on March 4, 2024, from https://www.newsweek.com/map-climate-change-denialism-us-state-1874412