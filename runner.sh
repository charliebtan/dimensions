for i in 0 1 2 3; do
    for j in 0 1 2 3 4; do
        export CUDA_VISIBLE_DEVICES=$i && nohup python -m PHDim --model='alexnet' --dataset='cifar10' --cat=$i --seed=$j > nohup_${i}${j}.out &
    done
done