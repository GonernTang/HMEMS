# Conda Environment Preparation

We use miniconda for environment setup.

```
# Create a new conda environment with Python 3.9
conda create -n vecmem python=3.9 -y

# Activate the environment
conda activate vecmem

# Install the dependencies
pip install -r requirements.txt
```

# Dataset preparation

## Locomo
Download locomo dataset and place it under `dataset/Locomo/`. You can find the file at the repo of Locomo [link](https://github.com/snap-research/locomo/blob/main/data/locomo10.json)

```
mkdir dataset/Locomo

# Move locomo10.json into the folder
# dataset/Locomo/locomo10.json
```

# Prepare .env file

Current version rely heavily on OpenAi API calls (For both embedding and LLM calls). Make sure your .env file has
the following fields:

OPENAI_API_KEY

OPENAI_BASE_URL (Optional. Use it when you are using third-party provider)

LOCOMO_PATH=`<Path to locomo10.json>`

LOCOMO_EMBEDDING_PATH=`<PathToVecMm>/VecMem/embeddings/Locomo/`

LOCOMO_INDEX_PATH=`<PathToVecMm>/VecMem/index/Locomo/`

LOCOMO_RES_PATH=`<PathToVecMm>/VecMem/results/Locomo/anwsers/`

LOCOMO_SCORE_PATH="`<PathToVecMm>/VecMem/results/Locomo/scores/`

MODEL="gpt-4o-mini" (Or other models you want to use, note that currently we only supports OpenAI models)

# Prepare embedding and index for the Locomo dataset

You can simply run the following command to initialize everything after having .env setup:

```
cd src
python3 ./run_experiments.py --init_env
```

# Run evaluation on vecmem

You could rely on `run.sh` script to run the evaluations on vecmem. Raw anwsers will be geenrated in
`LOCOMO_RES_PATH`, and the detailed evaluations will be generated in `LOCOMO_SCORE_PATH`.

Each run will be executed using `nohup`, thus multiple configs can be executed in parallel. You can use

`./run.sh <config name1> <config name2> ...` to initialize multiple configs. Execution log will be printed in `logs` folder

A complete execution cycle (complete locomo testing) will takes around 2-3 hours per config when iterative anwsering is enabled.