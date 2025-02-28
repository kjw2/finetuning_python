import yaml

# config 파일 로드
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# config 값을 사용하여 토크나이저 설정
tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
encoded = tokenizer(
    text, 
    truncation=True, 
    max_length=config['data']['max_length']
) 