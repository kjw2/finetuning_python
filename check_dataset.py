from datasets import load_dataset
import json

# IMDB 데이터셋 로드
dataset = load_dataset('stanfordnlp/imdb', split='train', cache_dir='./datasets/temp_cache')

# 컬럼 이름 출력
print("컬럼 이름:", dataset.column_names)

# 첫 번째 샘플 출력
if len(dataset) > 0:
    sample = dataset[0]
    print("첫 번째 샘플:")
    for key, value in sample.items():
        if isinstance(value, str) and len(value) > 100:
            print(f"{key}: {value[:100]}...")
        else:
            print(f"{key}: {value}")
else:
    print("샘플이 없습니다.")

# 데이터셋 정보 파일로 저장
with open('dataset_info.txt', 'w', encoding='utf-8') as f:
    f.write(f"컬럼 이름: {dataset.column_names}\n\n")
    if len(dataset) > 0:
        f.write("첫 번째 샘플:\n")
        for key, value in dataset[0].items():
            if isinstance(value, str):
                f.write(f"{key}: {value[:200]}...\n")
            else:
                f.write(f"{key}: {value}\n")
    else:
        f.write("샘플이 없습니다.\n")

print("\n데이터셋 정보가 dataset_info.txt 파일에 저장되었습니다.") 