################################# scale_lr

DEFAULTS="--seed 32348 --group WebKB"
DEFAULTS_CITATION="--seed 32348 --group Citation"
GAT="--attention gat"
DOT="--attention dot"

python run.py --dataset Cora-10 --name DSGNN-Cora-10-Splits-Dot $DEFAULTS_CITATION \
    --temporal-encoding --attention-activation tanh --attention-dropout 0.18 $DOT \
    --attention-units 64 --input-dropout 0.64 --units 64 --state-model-activation relu \
    --state-model-dropout 0.65 --n-steps 2
python run.py --dataset Cora-10 --name DSGNN-Cora-10-Splits-Gat $DEFAULTS_CITATION \
    --temporal-encoding --attention-activation tanh --attention-dropout 0.18 $GAT \
    --attention-units 64 --input-dropout 0.64 --units 64 --state-model-activation relu \
    --state-model-dropout 0.65 --n-steps 2

python run.py --dataset Pubmed-10 --name DSGNN-Pubmed-10-Splits-Dot $DEFAULTS_CITATION \
    --selfloops --walkers-per-node 14 --attention-activation elu --attention-dropout 0.0645 \
    $DOT --attention-units 32 --input-dropout 0.2775 --units 64 --state-model-activation tanh \
    --state-model-dropout 0.6349 --n-steps 2 --lr 0.01
python run.py --dataset Pubmed-10 --name DSGNN-Pubmed-10-Splits-Gat $DEFAULTS_CITATION \
    --selfloops --temporal-encoding --walkers-per-node 14 --attention-activation elu \
    --attention-dropout 0.0645 $GAT --attention-units 32 --input-dropout 0.2775 --units 64 \
    --state-model-activation tanh --state-model-dropout 0.6349 \
    --n-steps 2 --lr 0.01

python run.py --dataset Citeseer-10 --name DSGNN-Citeseer-10-Splits-Dot $DEFAULTS_CITATION \
    --selfloops --temporal-encoding --walkers-per-node 5 --attention-activation leaky_relu \
    $DOT --attention-dropout 0.4563 --attention-units 32 --input-dropout 0.8264 \
    --units 64 --state-model-activation relu --state-model-dropout 0.3029 \
    --n-steps 1 --lr 0.0001
python run.py --dataset Citeseer-10 --name DSGNN-Citeseer-10-Splits-Gat $DEFAULTS_CITATION \
    --selfloops --temporal-encoding --walkers-per-node 5 --attention-activation leaky_relu \
    -$GAT --attention-dropout 0.4563 --attention-units 32 --input-dropout 0.8264 \
    --units 64 --state-model-activation relu --state-model-dropout 0.3029 \
    --n-steps 2 --lr 0.0001

python run.py --dataset Cornell --name DSGNN-Cornell-10-Splits-Dot $DEFAULTS \
    --attention-activation tanh $DOT --attention-dropout 0.1238 \
    --attention-units 32 --walker-embds --input-dropout 0.5999 --units 256 \
    --state-model-activation elu --state-model-dropout 0.2389 \
    --n-steps 1 --lr 0.0001
python run.py --dataset Cornell --name DSGNN-Cornell-10-Splits-Gat $DEFAULTS \
    --attention-activation tanh $GAT --attention-dropout 0.1238 --attention-units 32 \
    --walker-embds --input-dropout 0.5999 --units 256 --state-model-activation elu \
    --state-model-dropout 0.2389 --n-steps 1 --lr 0.0001 &

python run.py --dataset Chameleon --name DSGNN-Chameleon-10-Splits-Dot $DEFAULTS \
    --walkers-per-node 8 --attention-activation tanh --attention-dropout 0.2390 $DOT \
    --attention-units 32 --state-model-dropout 0.3235 --input-dropout 0.7232 --units 64 \
    --state-model-activation elu --n-steps 2 --scale-lr 1.
python run.py --dataset Chameleon --name DSGNN-Chameleon-10-Splits-Gat $DEFAULTS \
    --temporal-encoding --walkers-per-node 8 --attention-activation leaky_relu \
    --attention-dropout 0.7812 $GAT --attention-units 32 --walker-embds \
    --state-model-dropout 0.1526 --input-dropout 0.4209 --units 64 \
    --state-model-activation elu --n-steps 3

python run.py --dataset Squirrel --name DSGNN-Squirrel-10-Splits-Gat $DEFAULTS \
    --temporal-encoding --walkers-per-node 8 --attention-activation relu \
    --attention-dropout 0.7812 $GAT --attention-units 64 --walker-embds \
    --state-model-dropout 0.1526 --input-dropout 0.55 --units 64 \
    --state-model-activation tanh --n-steps 2

python run.py --dataset Film --name DSGNN-Actor-10-Splits-Dot $DEFAULTS \
    --temporal-encoding --walkers-per-node 1 --attention-activation elu \
    --attention-dropout 0.6708 --attention-units 64 $DOT --walker-embds \
    --input-dropout 0.2332 --units 32 --state-model-activation leaky_relu \
    --state-model-dropout 0.1921 --n-steps 2
python run.py --dataset Film --name DSGNN-Actor-10-Splits-Gat $DEFAULTS \
    --temporal-encoding --walkers-per-node 1 --attention-activation elu \
    --attention-dropout 0.6708 --attention-units 64 --attention $GAT \
    --walker-embds --input-dropout 0.2332 --units 32 --state-model-activation leaky_relu \
    --state-model-dropout 0.1921 --n-steps 2

python run.py --dataset Wisconsin --name Wisconsin-10-Splits-Dot $DEFAULTS \
    --temporal-encoding --attention-activation elu --attention-dropout 0.8502 \
    --attention-units 32 --attention dot --input-dropout 0.3038 --units 64 \
    --state-model-activation relu --state-model-dropout 0.0286 \
    --n-steps 1 $DOT --walkers-per-node 3
python run.py --dataset Wisconsin --name Wisconsin-10-Splits-Gat $DEFAULTS \
    --selfloops --n-steps 2 --state-model-activation relu \
    --state-model-dropout 0.0286 --input-dropout 0.3038 --units 64 \
    --attention-units 32 --attention-activation elu --attention-dropout 0.8502 \
    $GAT --walkers-per-node 3

python run.py --dataset Texas --name Texas-10-Splits-Dot $DEFAULTS \
    --selfloops --n-steps 3 --temporal-encoding --state-model-activation relu  \
    --walker-embds --state-model-dropout 0.2376 --input-dropout 0.4466 --units 64 \
    --attention-units 32 --attention-activation leaky_relu $DOT \
    --attention-dropout 0.5024 --walkers-per-node 8 &
python run.py --dataset Texas --name Texas-10-Splits-Gat $DEFAULTS \
    --selfloops --n-steps 3 --state-model-activation relu \
    --state-model-dropout 0.2376 --input-dropout 0.4466 --units 64 \
    --attention-units 32 --attention-activation leaky_relu $GAT \
    --attention-dropout 0.5024 --walkers-per-node 8


# Evaluation
python benchmark_transductive.py --dataset Cora-10 --name DSGNN-Cora-10-Splits-Dot
python benchmark_transductive.py --dataset Cora-10 --name DSGNN-Cora-10-Splits-Gat
python benchmark_transductive.py --dataset Pubmed-10 --name DSGNN-Pubmed-10-Splits-Dot
python benchmark_transductive.py --dataset Pubmed-10 --name DSGNN-Pubmed-10-Splits-Gat
python benchmark_transductive.py --dataset Citeseer-10 --name DSGNN-Citeseer-10-Splits-Dot
python benchmark_transductive.py --dataset Citeseer-10 --name DSGNN-Citeseer-10-Splits-Gat
python benchmark_transductive.py --dataset Cornell --name DSGNN-Cornell-10-Splits-Dot
python benchmark_transductive.py --dataset Cornell --name DSGNN-Cornell-10-Splits-Gat
python benchmark_transductive.py --dataset Chameleon --name DSGNN-Chameleon-10-Splits-Dot
python benchmark_transductive.py --dataset Chameleon --name DSGNN-Chameleon-10-Splits-Gat
python benchmark_transductive.py --dataset Squirrel --name DSGNN-Squirrel-10-Splits-Gat
python benchmark_transductive.py --dataset Film --name DSGNN-Actor-10-Splits-Dot
python benchmark_transductive.py --dataset Film --name DSGNN-Actor-10-Splits-Gat
python benchmark_transductive.py --dataset Wisconsin --name Wisconsin-10-Splits-Dot
python benchmark_transductive.py --dataset Wisconsin --name Wisconsin-10-Splits-Gat
python benchmark_transductive.py --dataset Texas --name Texas-10-Splits-Dot
python benchmark_transductive.py --dataset Texas --name Texas-10-Splits-Gat