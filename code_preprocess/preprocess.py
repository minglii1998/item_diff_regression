import json

raw_json_path_train = 'data/json_file/train_final.json'
raw_json_path_test = 'data/json_file/test_final.json'
raw_json_path_gold = 'data/json_file/gold_final.json'

with open(raw_json_path_train) as f:
    data_train = json.load(f)
with open(raw_json_path_test) as f:
    data_test = json.load(f)
with open(raw_json_path_gold) as f:
    data_gold = json.load(f)

answer_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

def preprocess_type1(data_train, data_test, data_gold):
    # Type 1, only question is given
    data_list_train = []
    for i, data_i in enumerate(data_train):

        data_temp = {}
        text = data_i['ItemStem_Text']
        for option in answer_list:
            key = 'Answer__' + option
            if isinstance(data_i[key], str):
                text = text + '\n' + option + '. ' + data_i[key]
        data_temp['text'] = text
        data_temp['Difficulty'] = data_i['Difficulty']
        data_temp['Response_Time'] = data_i['Response_Time']
        data_list_train.append(data_temp)

    with open('train_final_type1.json', 'w') as f:
        json.dump(data_list_train, f, indent=4)

    assert len(data_test) == len(data_gold)
    data_list_test = []
    for i in range(len(data_test)):

        data_i = data_test[i]
        gold_i = data_gold[i]

        data_temp = {}
        text = data_i['ItemStem_Text']
        for option in answer_list:
            key = 'Answer__' + option
            if isinstance(data_i[key], str):
                text = text + '\n' + option + '. ' + data_i[key]

        data_temp['text'] = text
        data_temp['Difficulty'] = gold_i['Difficulty']
        data_temp['Response_Time'] = gold_i['Response_Time']
        data_list_test.append(data_temp)

    with open('test_final_type1.json', 'w') as f:
        json.dump(data_list_test, f, indent=4)

def preprocess_type2(data_train, data_test, data_gold):
    # Type 2, both question and answer are given
    data_list_train = []
    for i, data_i in enumerate(data_train):

        data_temp = {}
        text = 'Question: ' + data_i['ItemStem_Text']

        for option in answer_list:
            key = 'Answer__' + option
            if isinstance(data_i[key], str):
                text = text + '\n' + option + '. ' + data_i[key]
        answer_key = 'Answer__' + data_i['Answer_Key']
        asnwer_text = data_i[answer_key]
        text = text + '\n' + 'Answer: ' + data_i['Answer_Key'] + '. ' + asnwer_text

        data_temp['text'] = text

        data_temp['Difficulty'] = data_i['Difficulty']
        data_temp['Response_Time'] = data_i['Response_Time']
        data_list_train.append(data_temp)

    with open('train_final_type2.json', 'w') as f:
        json.dump(data_list_train, f, indent=4)

    assert len(data_test) == len(data_gold)
    data_list_test = []
    for i in range(len(data_test)):
        data_i = data_test[i]
        gold_i = data_gold[i]

        data_temp = {}
        text = data_i['ItemStem_Text']
        for option in answer_list:
            key = 'Answer__' + option
            if isinstance(data_i[key], str):
                text = text + '\n' + option + '. ' + data_i[key]
        answer_key = 'Answer__' + data_i['Answer_Key']
        asnwer_text = data_i[answer_key]
        text = text + '\n' + 'Answer: ' + data_i['Answer_Key'] + '. ' + asnwer_text

        data_temp['text'] = text

        data_temp['Difficulty'] = gold_i['Difficulty']
        data_temp['Response_Time'] = gold_i['Response_Time']
        data_list_test.append(data_temp)

    with open('test_final_type2.json', 'w') as f:
        json.dump(data_list_test, f, indent=4)

def preprocess_type3(data_train, data_test, data_gold):
    # Type 3, Question, Answer, Exam Name, Exam Step
    data_list_train = []
    for i, data_i in enumerate(data_train):

        text = 'Below is a Multiple Choice Question from United States Medical Licensing Examination (USMLE) ' + data_i['EXAM'] + '.'

        data_temp = {}
        text = text + '\nQuestion: ' + data_i['ItemStem_Text']

        for option in answer_list:
            key = 'Answer__' + option
            if isinstance(data_i[key], str):
                text = text + '\n' + option + '. ' + data_i[key]
        answer_key = 'Answer__' + data_i['Answer_Key']
        asnwer_text = data_i[answer_key]
        text = text + '\n' + 'Answer: ' + data_i['Answer_Key'] + '. ' + asnwer_text

        data_temp['text'] = text

        data_temp['Difficulty'] = data_i['Difficulty']
        data_temp['Response_Time'] = data_i['Response_Time']
        data_list_train.append(data_temp)

    with open('train_final_type3.json', 'w') as f:
        json.dump(data_list_train, f, indent=4)

    assert len(data_test) == len(data_gold)
    data_list_test = []
    for i in range(len(data_test)):
        data_i = data_test[i]
        gold_i = data_gold[i]

        text = 'Below is a Multiple Choice Question from United States Medical Licensing Examination (USMLE) ' + data_i['EXAM'] + '.'

        data_temp = {}
        text = text + '\nQuestion: ' + data_i['ItemStem_Text']

        for option in answer_list:
            key = 'Answer__' + option
            if isinstance(data_i[key], str):
                text = text + '\n' + option + '. ' + data_i[key]
        answer_key = 'Answer__' + data_i['Answer_Key']
        asnwer_text = data_i[answer_key]
        text = text + '\n' + 'Answer: ' + data_i['Answer_Key'] + '. ' + asnwer_text

        data_temp['text'] = text

        data_temp['Difficulty'] = gold_i['Difficulty']
        data_temp['Response_Time'] = gold_i['Response_Time']
        data_list_test.append(data_temp)

    with open('test_final_type3.json', 'w') as f:
        json.dump(data_list_test, f, indent=4)

def preprocess_type4(data_train, data_test, data_gold):
    # Type 4, Question, Answer, Exam Name, Exam Step, Analysis

    with open('data/train_final_gpt4turbo_analysis.jsonl', 'r', encoding='utf-8') as file:
        analysis_train = [json.loads(line.strip()) for line in file]
    assert len(data_train) == len(analysis_train)

    data_list_train = []
    for i, data_i in enumerate(data_train):

        text = 'Below is a Multiple Choice Question from United States Medical Licensing Examination (USMLE) ' + data_i['EXAM'] + '.'

        data_temp = {}
        text = text + '\nQuestion: ' + data_i['ItemStem_Text']

        for option in answer_list:
            key = 'Answer__' + option
            if isinstance(data_i[key], str):
                text = text + '\n' + option + '. ' + data_i[key]
        answer_key = 'Answer__' + data_i['Answer_Key']
        asnwer_text = data_i[answer_key]
        text = text + '\n' + 'Answer: ' + data_i['Answer_Key'] + '. ' + asnwer_text

        # Add analysis
        text = text + '\n' + analysis_train[i]['gpt4_turbo_analysis']

        data_temp['text'] = text

        data_temp['Difficulty'] = data_i['Difficulty']
        data_temp['Response_Time'] = data_i['Response_Time']
        data_list_train.append(data_temp)

    with open('train_final_type4.json', 'w') as f:
        json.dump(data_list_train, f, indent=4)

    with open('data/test_final_gpt4turbo_analysis.jsonl', 'r', encoding='utf-8') as file:
        analysis_test = [json.loads(line.strip()) for line in file]
    assert len(data_test) == len(analysis_test)

    assert len(data_test) == len(data_gold)
    data_list_test = []
    for i in range(len(data_test)):
        data_i = data_test[i]
        gold_i = data_gold[i]

        text = 'Below is a Multiple Choice Question from United States Medical Licensing Examination (USMLE) ' + data_i['EXAM'] + '.'

        data_temp = {}
        text = text + '\nQuestion: ' + data_i['ItemStem_Text']

        for option in answer_list:
            key = 'Answer__' + option
            if isinstance(data_i[key], str):
                text = text + '\n' + option + '. ' + data_i[key]
        answer_key = 'Answer__' + data_i['Answer_Key']
        asnwer_text = data_i[answer_key]
        text = text + '\n' + 'Answer: ' + data_i['Answer_Key'] + '. ' + asnwer_text

        # Add analysis
        text = text + '\n' + analysis_test[i]['gpt4_turbo_analysis']

        data_temp['text'] = text

        data_temp['Difficulty'] = gold_i['Difficulty']
        data_temp['Response_Time'] = gold_i['Response_Time']
        data_list_test.append(data_temp)

    with open('test_final_type4.json', 'w') as f:
        json.dump(data_list_test, f, indent=4)

def preprocess_type5(data_train, data_test, data_gold):
    # Type 5, Question, Answer, Exam Name, Exam Step, Analysis without scores

    with open('data/train_final_gpt4o_analysis_pure.jsonl', 'r', encoding='utf-8') as file:
        analysis_train = [json.loads(line.strip()) for line in file]
    assert len(data_train) == len(analysis_train)

    data_list_train = []
    for i, data_i in enumerate(data_train):

        text = 'Below is a Multiple Choice Question from United States Medical Licensing Examination (USMLE) ' + data_i['EXAM'] + '.'

        data_temp = {}
        text = text + '\nQuestion: ' + data_i['ItemStem_Text']

        for option in answer_list:
            key = 'Answer__' + option
            if isinstance(data_i[key], str):
                text = text + '\n' + option + '. ' + data_i[key]
        answer_key = 'Answer__' + data_i['Answer_Key']
        asnwer_text = data_i[answer_key]
        text = text + '\n' + 'Answer: ' + data_i['Answer_Key'] + '. ' + asnwer_text

        # Add analysis
        text = text + '\n' + analysis_train[i]['gpt4o_analysis']

        data_temp['text'] = text

        data_temp['Difficulty'] = data_i['Difficulty']
        data_temp['Response_Time'] = data_i['Response_Time']
        data_list_train.append(data_temp)

    with open('train_final_type5.json', 'w') as f:
        json.dump(data_list_train, f, indent=4)

    with open('data/test_final_gpt4o_analysis_pure.jsonl', 'r', encoding='utf-8') as file:
        analysis_test = [json.loads(line.strip()) for line in file]
    assert len(data_test) == len(analysis_test)

    assert len(data_test) == len(data_gold)
    data_list_test = []
    for i in range(len(data_test)):
        data_i = data_test[i]
        gold_i = data_gold[i]

        text = 'Below is a Multiple Choice Question from United States Medical Licensing Examination (USMLE) ' + data_i['EXAM'] + '.'

        data_temp = {}
        text = text + '\nQuestion: ' + data_i['ItemStem_Text']

        for option in answer_list:
            key = 'Answer__' + option
            if isinstance(data_i[key], str):
                text = text + '\n' + option + '. ' + data_i[key]
        answer_key = 'Answer__' + data_i['Answer_Key']
        asnwer_text = data_i[answer_key]
        text = text + '\n' + 'Answer: ' + data_i['Answer_Key'] + '. ' + asnwer_text

        # Add analysis
        text = text + '\n' + analysis_test[i]['gpt4o_analysis']

        data_temp['text'] = text

        data_temp['Difficulty'] = gold_i['Difficulty']
        data_temp['Response_Time'] = gold_i['Response_Time']
        data_list_test.append(data_temp)

    with open('test_final_type5.json', 'w') as f:
        json.dump(data_list_test, f, indent=4)

def preprocess_type6(data_train, data_test, data_gold):
    # Type 6, Analysis only

    with open('data/train_final_gpt4o_analysis_pure.jsonl', 'r', encoding='utf-8') as file:
        analysis_train = [json.loads(line.strip()) for line in file]
    assert len(data_train) == len(analysis_train)

    data_list_train = []
    for i, data_i in enumerate(data_train):
        
        data_temp = {}
        text = analysis_train[i]['gpt4o_analysis']

        data_temp['text'] = text

        data_temp['Difficulty'] = data_i['Difficulty']
        data_temp['Response_Time'] = data_i['Response_Time']
        data_list_train.append(data_temp)

    with open('train_final_type6.json', 'w') as f:
        json.dump(data_list_train, f, indent=4)

    with open('data/test_final_gpt4o_analysis_pure.jsonl', 'r', encoding='utf-8') as file:
        analysis_test = [json.loads(line.strip()) for line in file]
    assert len(data_test) == len(analysis_test)

    assert len(data_test) == len(data_gold)
    data_list_test = []
    for i in range(len(data_test)):
        data_i = data_test[i]
        gold_i = data_gold[i]

        data_temp = {}
        text = analysis_test[i]['gpt4o_analysis']

        data_temp['text'] = text

        data_temp['Difficulty'] = gold_i['Difficulty']
        data_temp['Response_Time'] = gold_i['Response_Time']
        data_list_test.append(data_temp)

    with open('test_final_type6.json', 'w') as f:
        json.dump(data_list_test, f, indent=4)


# preprocess_type1(data_train, data_test, data_gold)
# preprocess_type2(data_train, data_test, data_gold)
# preprocess_type3(data_train, data_test, data_gold)
# preprocess_type4(data_train, data_test, data_gold)
# preprocess_type5(data_train, data_test, data_gold)
preprocess_type6(data_train, data_test, data_gold)