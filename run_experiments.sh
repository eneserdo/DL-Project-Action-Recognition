# Varying lr for head (freeze body)

# python main.py --dataset Stanford40 --name S_lrh4 --wandb -lrh 0.0001
# sleep 5

# python main.py --dataset Stanford40 --name S_lrh3 --wandb -lrh 0.001
# sleep 5

# python main.py --dataset Stanford40 --name S_lrh2 --wandb -lrh 0.01
# sleep 5

# Varying lr for body (keep head lr3)
python main.py --dataset Stanford40 --name S_lrh3_lrb5 --epochs 15 --wandb -lrh 0.001 -lrb 0.00001 --batch-size 50
sleep 5

python main.py --dataset Stanford40 --name S_lrh3_lrb4 --epochs 15 --wandb -lrh 0.001 -lrb 0.0001 --batch-size 50
sleep 5

python main.py --dataset Stanford40 --name S_lrh3_lrb3 --epochs 15 --wandb -lrh 0.001 -lrb 0.001 --batch-size 50
sleep 5
# Use weighted loss



# Use regularizer