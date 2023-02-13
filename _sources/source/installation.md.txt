# Getting Started


## Install Geno4SD


Using [Conda](https://conda.io/docs/) is recommended for package
management, in order to create self contained environments with specific
versions of packages. Optionally main external packages required are the
Jupyter notebook:

```{code-block}
# create a new virtual environment with Python 3.8
conda create -y -n geno4sd python=3.8
conda activate geno4sd

#download latest version
curl -L  https://github.com/ComputationalGenomics/Geno4SD/archive/main.zip

# install Geno4SD
pip install ./Geno4SD-main.zip

# install interactive tools
pip install jupyterlab
```


## Troubleshooting

For installation issues, or if you're encountering some oddities in the API, please create a new issue on our Issue Tracker, explaining as detailed as possible, how to reproduce the issue, including what you expected to happen, as well as what actually happened. Feel free to attach a screenshot illustrating the issue.

We will try to fix it as fast as possible.

### Pull Requests

Know how to fix something? We love pull requests! Here's a quick guide:

- Check for open issues, or open a fresh issue to start a discussion around a feature idea or a bug. Opening a separate issue to discuss the change is less important for smaller changes, as the discussion can be done in the pull request.
- Fork the relevant repository on GitHub, and start making your changes.
- Check out the README for the project for information specific to that repository.
- Push the change (we recommend using a separate branch for your feature). Please also consider to add test/coverage of your code.
- Open a pull request.

We try to merge and deploy changes as soon as possible, or at least leave some feedback.
