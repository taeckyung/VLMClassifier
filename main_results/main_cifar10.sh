# original

python main.py --method vlm --model_id llava-hf/llava-1.5-7b-hf --data_path ../data/cifar10c_jsonl/cifar10_test.jsonl --class_path ../data/cifar10_classes.json --split test --output_path outputs/cifar10_c/llava7b_original.jsonl --including_label True --n_labels 10 --batch_size 8

# corruption sev 5 (gaussian_noise, shot_noise, impulse_noise)

python main.py --method vlm --model_id llava-hf/llava-1.5-7b-hf --data_path ../data/cifar10c_jsonl/cifar10c_gaussian_noise_severity_5.jsonl --class_path ../data/cifar10_classes.json --split test --output_path outputs/cifar10_c/llava7b_gaussian_noise.jsonl --including_label True --n_labels 10 --batch_size 8
python main.py --method vlm --model_id llava-hf/llava-1.5-7b-hf --data_path ../data/cifar10c_jsonl/cifar10c_shot_noise_severity_5.jsonl --class_path ../data/cifar10_classes.json --split test --output_path outputs/cifar10_c/llava7b_shot_noise.jsonl --including_label True --n_labels 10 --batch_size 8
python main.py --method vlm --model_id llava-hf/llava-1.5-7b-hf --data_path ../data/cifar10c_jsonl/cifar10c_impulse_noise_severity_5.jsonl --class_path ../data/cifar10_classes.json --split test --output_path outputs/cifar10_c/llava7b_impulse_noise.jsonl --including_label True --n_labels 10 --batch_size 8


