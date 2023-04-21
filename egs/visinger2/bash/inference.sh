
exp_dir=$(pwd)
base_dir=$(dirname $(dirname $exp_dir))

export PYTHONPATH=$base_dir
export PYTHONIOENCODING=UTF-8

CUDA_VISIBLE_DEVICES=0 python inference.py \
    -model_dir /Users/matthewrice/Developer/VISinger2/ckpt/visinger_out \
    -input_dir /Users/matthewrice/Developer/VISinger2/test_inference.txt \
    -output_dir /Users/matthewrice/Developer/VISinger2/inference \

