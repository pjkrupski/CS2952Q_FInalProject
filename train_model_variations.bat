python train.py --batch-size=16 --epochs 120 --lr=2e-4 --model-name cnn --percent-perturbed 5 --eps 0.03  --adv-training-type fgsm --output-file models/cnn_none.pt
python train.py --batch-size=16 --epochs 120 --lr=2e-4 --model-name cnn --percent-perturbed 5 --eps 0.03  --adv-training-type fgsm --output-file models/cnn_5percent_0.03eps_fgsm.pt
python train.py --batch-size=16 --epochs 120 --lr=2e-4 --model-name cnn --percent-perturbed 5 --eps 0.03  --adv-training-type pgd --output-file models/cnn_5percent_0.03eps_pgd.pt
python train.py --batch-size=16 --epochs 120 --lr=2e-4 --model-name cnn --percent-perturbed 5 --eps 0.03  --adv-training-type mixed --output-file models/cnn_5percent_0.03eps_mixed.pt

python train.py --batch-size=16 --epochs 120 --lr=1e-5 --model-name vit --percent-perturbed 5 --eps 0.03  --adv-training-type fgsm --output-file models/vit_none.pt
python train.py --batch-size=16 --epochs 120 --lr=1e-5 --model-name vit --percent-perturbed 5 --eps 0.03  --adv-training-type fgsm --output-file models/vit_5percent_0.03eps_fgsm.pt
python train.py --batch-size=16 --epochs 120 --lr=1e-5 --model-name vit --percent-perturbed 5 --eps 0.03  --adv-training-type pgd --output-file models/vit_5percent_0.03eps_pgd.pt
python train.py --batch-size=16 --epochs 120 --lr=1e-5 --model-name vit --percent-perturbed 5 --eps 0.03  --adv-training-type mixed --output-file models/vit_5percent_0.03eps_mixed.pt
