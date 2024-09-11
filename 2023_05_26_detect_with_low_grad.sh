CUDA_VISIBLE_DEVICES=2 python 2023_05_26_detect_with_low_grad.py --batch_size 1 --model resnet18 --data cifar10 --kernel NFK --base_method knn --learning_rate 0.01 --ifbn True --ood_data SVHN


CUDA_VISIBLE_DEVICES=2 python 2023_05_26_detect_with_low_grad.py --batch_size 1 --model resnet18 --data cifar100 --kernel NFK --base_method knn --learning_rate 0.01 --ifbn True --ood_data SVHN

CUDA_VISIBLE_DEVICES=1 python 2023_05_26_detect_with_low_grad.py --batch_size 1 --model vit --data imagenet --kernel NFK --base_method knn --learning_rate 0.01 --ifbn True --ood_data inat




CUDA_VISIBLE_DEVICES=0 python 2023_05_26_detect_with_low_grad.py --batch_size 1 --model resnet50 --data imagenet --kernel NFK --base_method knn --learning_rate 0.01 --ifbn True --ood_data sun50


