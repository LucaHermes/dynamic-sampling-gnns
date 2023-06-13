DEFAULTS="--dataset Wisconsin --group Manual-Runs__Wisconsin__Grid --seed 32348 --selfloops --n-steps 2 
           --state-model-activation relu --state-model-dropout 0.0286 --input-dropout 0.3038 --units 64 
           --attention-units 32 --attention-activation elu --attention-dropout 0.8502"

UNIFORM="--walkers-per-node 3"
UNIFORM_EMBDS="--walkers-per-node 3 --walker-embds"
DEGREE=""

TEMP_ENC="--temporal-encoding"
PARALLEL="--train-parallel"
GAT="--attention gat"
DOT="--attention dot"

# 1
python run.py --name Wisconsin-Grid-1 $DEFAULTS $DOT \
    $TEMP_ENC $UNIFORM &
# 2
python run.py --name Wisconsin-Grid-2 $DEFAULTS $DOT \
    $UNIFORM
# 3 (completely default)
python run.py --name Wisconsin-Grid-3 $DEFAULTS $DOT \
    $TEMP_ENC $UNIFORM_EMBDS &
# 4
python run.py --name Wisconsin-Grid-4 $DEFAULTS $DOT \
    $UNIFORM_EMBDS
# 5
python run.py --name Wisconsin-Grid-5 $DEFAULTS $DOT \
    $TEMP_ENC $DEGREE &
# 6
python run.py --name Wisconsin-Grid-6 $DEFAULTS $DOT \
    $DEGREE
# 7
python run.py --name Wisconsin-Grid-7 $DEFAULTS $DOT \
    $UNIFORM $PARALLEL &
# 8
python run.py --name Wisconsin-Grid-8 $DEFAULTS $DOT \
    $UNIFORM_EMBDS $PARALLEL
# 9
python run.py --name Wisconsin-Grid-9 $DEFAULTS $DOT \
    $DEGREE $PARALLEL &
# 10
python run.py --name Wisconsin-Grid-10 $DEFAULTS \
    $UNIFORM $GAT
# 11
python run.py --name Wisconsin-Grid-11 $DEFAULTS \
    $UNIFORM_EMBDS $GAT &
# 12
python run.py --name Wisconsin-Grid-12 $DEFAULTS \
    $DEGREE $GAT

python benchmark_transductive.py --dataset Wisconsin --name Wisconsin-Grid-1 &
python benchmark_transductive.py --dataset Wisconsin --name Wisconsin-Grid-2 
python benchmark_transductive.py --dataset Wisconsin --name Wisconsin-Grid-3 &
python benchmark_transductive.py --dataset Wisconsin --name Wisconsin-Grid-4 
python benchmark_transductive.py --dataset Wisconsin --name Wisconsin-Grid-5 &
python benchmark_transductive.py --dataset Wisconsin --name Wisconsin-Grid-6 
python benchmark_transductive.py --dataset Wisconsin --name Wisconsin-Grid-7 &
python benchmark_transductive.py --dataset Wisconsin --name Wisconsin-Grid-8 
python benchmark_transductive.py --dataset Wisconsin --name Wisconsin-Grid-9 &
python benchmark_transductive.py --dataset Wisconsin --name Wisconsin-Grid-10
python benchmark_transductive.py --dataset Wisconsin --name Wisconsin-Grid-11 &
python benchmark_transductive.py --dataset Wisconsin --name Wisconsin-Grid-12