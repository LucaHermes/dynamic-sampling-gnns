DEFAULTS="--dataset Texas --group Manual-Runs__Texas__Grid --seed 32348 --selfloops --n-steps 3 
           --state-model-activation relu --state-model-dropout 0.2376 --input-dropout 0.4466 --units 64 
           --attention-units 32 --attention-activation leaky_relu --attention-dropout 0.5024"

UNIFORM="--walkers-per-node 8"
UNIFORM_EMBDS="--walkers-per-node 8 --walker-embds"
DEGREE=""

TEMP_ENC="--temporal-encoding"
PARALLEL="--train-parallel"
GAT="--attention gat"
DOT="--attention dot"

# 1
python run.py --name Texas-Grid-1 $DEFAULTS $DOT \
    $TEMP_ENC $UNIFORM &
# 2
python run.py --name Texas-Grid-2 $DEFAULTS $DOT \
    $UNIFORM
# 3
python run.py --name Texas-Grid-3 $DEFAULTS $DOT \
    $TEMP_ENC $UNIFORM_EMBDS &
# 4
python run.py --name Texas-Grid-4 $DEFAULTS $DOT \
    $UNIFORM_EMBDS
# 5
python run.py --name Texas-Grid-5 $DEFAULTS $DOT \
    $TEMP_ENC $DEGREE &
# 6
python run.py --name Texas-Grid-6 $DEFAULTS $DOT \
    $DEGREE
# 7
python run.py --name Texas-Grid-7 $DEFAULTS $DOT \
    $UNIFORM $PARALLEL &
# 8
python run.py --name Texas-Grid-8 $DEFAULTS $DOT \
    $UNIFORM_EMBDS $PARALLEL
# 9
python run.py --name Texas-Grid-9 $DEFAULTS $DOT \
    $DEGREE $PARALLEL &
# 10
python run.py --name Texas-Grid-10 $DEFAULTS \
    $UNIFORM $GAT
# 11
python run.py --name Texas-Grid-11 $DEFAULTS \
    $UNIFORM_EMBDS $GAT &
# 12
python run.py --name Texas-Grid-12 $DEFAULTS \
    $DEGREE $GAT

python benchmark_transductive.py --dataset Texas --name Texas-Grid-1 &
python benchmark_transductive.py --dataset Texas --name Texas-Grid-2 
python benchmark_transductive.py --dataset Texas --name Texas-Grid-3 &
python benchmark_transductive.py --dataset Texas --name Texas-Grid-4 
python benchmark_transductive.py --dataset Texas --name Texas-Grid-5 &
python benchmark_transductive.py --dataset Texas --name Texas-Grid-6 
python benchmark_transductive.py --dataset Texas --name Texas-Grid-7 &
python benchmark_transductive.py --dataset Texas --name Texas-Grid-8 
python benchmark_transductive.py --dataset Texas --name Texas-Grid-9 &
python benchmark_transductive.py --dataset Texas --name Texas-Grid-10
python benchmark_transductive.py --dataset Texas --name Texas-Grid-11 &
python benchmark_transductive.py --dataset Texas --name Texas-Grid-12