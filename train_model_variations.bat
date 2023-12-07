@REM python train.py --batch-size=16 --epochs 120 --lr=2e-4 --model-name cnn --output-file models/cnn_none.pt
@REM python train.py --batch-size=16 --epochs 120 --lr=2e-4 --model-name cnn --percent-perturbed 10 --eps 0.05  --adv-training-type fgsm --output-file models/cnn_10percent_0.05eps_fgsm.pt
@REM python train.py --batch-size=16 --epochs 120 --lr=2e-4 --model-name cnn --percent-perturbed 10 --eps 0.05  --adv-training-type pgd --output-file models/cnn_10percent_0.05eps_pgd.pt
@REM python train.py --batch-size=16 --epochs 120 --lr=2e-4 --model-name cnn --percent-perturbed 5 --eps 0.03  --adv-training-type mixed --output-file models/cnn_5percent_0.03eps_mixed.pt

@REM python train.py --batch-size=16 --epochs 120 --lr=1e-5 --model-name vit --output-file models/vit_none.pt
@REM python train.py --batch-size=16 --epochs 120 --lr=1e-5 --model-name vit --percent-perturbed 10 --eps 0.05  --adv-training-type fgsm --output-file models/vit_10percent_0.05eps_fgsm.pt
@REM python train.py --batch-size=16 --epochs 120 --lr=1e-5 --model-name vit --percent-perturbed 10 --eps 0.05  --adv-training-type pgd --output-file models/vit_10percent_0.05eps_pgd.pt
@REM python train.py --batch-size=16 --epochs 120 --lr=1e-5 --model-name vit --percent-perturbed 5 --eps 0.03  --adv-training-type mixed --output-file models/vit_5percent_0.03eps_mixed.pt

@REM python train.py --batch-size=16 --epochs 120 --lr=2e-4 --model-name cnn --temp 1 --is-teacher --output-file models/cnn_teacher_1.pt
@REM python train.py --batch-size=16 --epochs 120 --lr=2e-4 --model-name cnn --temp 1.5 --is-teacher --output-file models/cnn_teacher_1.5.pt
@REM python train.py --batch-size=16 --epochs 120 --lr=2e-4 --model-name cnn --temp 2 --is-teacher --output-file models/cnn_teacher_2.pt
@REM python train.py --batch-size=16 --epochs 120 --lr=2e-4 --model-name cnn --temp 5 --is-teacher --output-file models/cnn_teacher_5.pt
@REM python train.py --batch-size=16 --epochs 120 --lr=2e-4 --model-name cnn --temp 10 --is-teacher --output-file models/cnn_teacher_10.pt

python train.py --batch-size=16 --epochs 120 --lr=2e-4 --model-name cnn --temp 1 --no-is-teacher --teacher-model models/cnn_teacher_1.5.pt --output-file models/cnn_student_1.pt
@REM python train.py --batch-size=16 --epochs 120 --lr=2e-4 --model-name cnn --temp 1.5 --no-is-teacher --teacher-model models/cnn_teacher_1.5.pt --output-file models/cnn_student_1.5.pt
@REM python train.py --batch-size=16 --epochs 120 --lr=2e-4 --model-name cnn --temp 2 --no-is-teacher --teacher-model models/cnn_teacher_2.pt --output-file models/cnn_student_2.pt
@REM python train.py --batch-size=16 --epochs 120 --lr=2e-4 --model-name cnn --temp 5 --no-is-teacher --teacher-model models/cnn_teacher_5.pt --output-file models/cnn_student_5.pt
@REM python train.py --batch-size=16 --epochs 120 --lr=2e-4 --model-name cnn --temp 10 --no-is-teacher --teacher-model models/cnn_teacher_10.pt --output-file models/cnn_student_10.pt