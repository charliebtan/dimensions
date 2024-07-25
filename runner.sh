for i in 0 1 2 3; do
    export CUDA_VISIBLE_DEVICES=$i && nohup python -m PHDim --model='alexnet' --dataset='cifar10' --cat=$i --seeds=[0] > nohup_${i}${j}.out &
    export CUDA_VISIBLE_DEVICES=$i && nohup python -m PHDim --model='alexnet' --dataset='cifar10' --cat=$i+4 --seeds=[0] > nohup_${i}${j}.out &
done