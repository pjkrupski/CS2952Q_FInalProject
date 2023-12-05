@REM python train.py --batch-size=16 --epochs 120 --lr=2e-4 --model-name cnn --output-file models/cnn_none.pt
@REM python train.py --batch-size=16 --epochs 120 --lr=2e-4 --model-name cnn --percent-perturbed 10 --eps 0.05  --adv-training-type fgsm --output-file models/cnn_10percent_0.05eps_fgsm.pt
@REM python train.py --batch-size=16 --epochs 120 --lr=2e-4 --model-name cnn --percent-perturbed 10 --eps 0.05  --adv-training-type pgd --output-file models/cnn_10percent_0.05eps_pgd.pt
@REM python train.py --batch-size=16 --epochs 120 --lr=2e-4 --model-name cnn --percent-perturbed 5 --eps 0.03  --adv-training-type mixed --output-file models/cnn_5percent_0.03eps_mixed.pt

@REM python train.py --batch-size=16 --epochs 120 --lr=1e-5 --model-name vit --output-file models/vit_none.pt
@REM python train.py --batch-size=16 --epochs 120 --lr=1e-5 --model-name vit --percent-perturbed 10 --eps 0.05  --adv-training-type fgsm --output-file models/vit_10percent_0.05eps_fgsm.pt
@REM python train.py --batch-size=16 --epochs 120 --lr=1e-5 --model-name vit --percent-perturbed 10 --eps 0.05  --adv-training-type pgd --output-file models/vit_10percent_0.05eps_pgd.pt
@REM python train.py --batch-size=16 --epochs 120 --lr=1e-5 --model-name vit --percent-perturbed 5 --eps 0.03  --adv-training-type mixed --output-file models/vit_5percent_0.03eps_mixed.pt

python train.py --batch-size=16 --epochs 120 --lr=2e-4 --model-name cnn --temp 2 --teacher True --output-file models/cnn_teacher_2.pt
python train.py --batch-size=16 --epochs 120 --lr=2e-4 --model-name cnn --temp 2 --teacher True --output-file models/cnn_teacher_5.pt
python train.py --batch-size=16 --epochs 120 --lr=2e-4 --model-name cnn --temp 2 --teacher True --output-file models/cnn_teacher_10.pt
