python main.py --dataset Stanford40 --name S_base0 --wandb
sleep 5

python main.py --dataset Stanford40 --name S_base1 --wandb -lrh 0.001
sleep 5

python main.py --dataset Stanford40 --name S_base2 --wandb -lrh 0.001 -lrb 0.0001 
sleep 5