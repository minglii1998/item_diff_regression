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

def preprocess_type2(data_train, data_test, data_gold):
    # Type 2, both question and answer are given
    data_list_train = []
    data_list_train_step1 = []
    data_list_train_step2 = []
    data_list_train_step3 = []
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
        
        if data_i['EXAM'] == 'STEP 1':
            data_list_train_step1.append(data_temp)
        elif data_i['EXAM'] == 'STEP 2':
            data_list_train_step2.append(data_temp)
        elif data_i['EXAM'] == 'STEP 3':
            data_list_train_step3.append(data_temp)
        
    # with open('train_final_type2.json', 'w') as f:
    #     json.dump(data_list_train, f, indent=4)
    with open('train_final_type2_step1.json', 'w') as f:
        json.dump(data_list_train_step1, f, indent=4)
    with open('train_final_type2_step2.json', 'w') as f:
        json.dump(data_list_train_step2, f, indent=4)
    with open('train_final_type2_step3.json', 'w') as f:
        json.dump(data_list_train_step3, f, indent=4)

    assert len(data_test) == len(data_gold)
    data_list_test = []
    data_list_test_step1 = []
    data_list_test_step2 = []
    data_list_test_step3 = []
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

        if data_i['EXAM'] == 'STEP 1':
            data_list_test_step1.append(data_temp)
        elif data_i['EXAM'] == 'STEP 2':
            data_list_test_step2.append(data_temp)
        elif data_i['EXAM'] == 'STEP 3':
            data_list_test_step3.append(data_temp)

    # with open('test_final_type2.json', 'w') as f:
    #     json.dump(data_list_test, f, indent=4)
    with open('test_final_type2_step1.json', 'w') as f:
        json.dump(data_list_test_step1, f, indent=4)
    with open('test_final_type2_step2.json', 'w') as f:
        json.dump(data_list_test_step2, f, indent=4)
    with open('test_final_type2_step3.json', 'w') as f:
        json.dump(data_list_test_step3, f, indent=4)

def preprocess_type3(data_train, data_test, data_gold):
    # Type 3, Question, Answer, Exam Name, Exam Step
    data_list_train = []
    data_list_train_step1 = []
    data_list_train_step2 = []
    data_list_train_step3 = []
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
        if data_i['EXAM'] == 'STEP 1':
            data_list_train_step1.append(data_temp)
        elif data_i['EXAM'] == 'STEP 2':
            data_list_train_step2.append(data_temp)
        elif data_i['EXAM'] == 'STEP 3':
            data_list_train_step3.append(data_temp)

    # with open('train_final_type3.json', 'w') as f:
    #     json.dump(data_list_train, f, indent=4)
    with open('train_final_type3_step1.json', 'w') as f:
        json.dump(data_list_train_step1, f, indent=4)
    with open('train_final_type3_step2.json', 'w') as f:
        json.dump(data_list_train_step2, f, indent=4)
    with open('train_final_type3_step3.json', 'w') as f:
        json.dump(data_list_train_step3, f, indent=4)

    assert len(data_test) == len(data_gold)
    data_list_test = []
    data_list_test_step1 = []
    data_list_test_step2 = []
    data_list_test_step3 = []
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
        if data_i['EXAM'] == 'STEP 1':
            data_list_test_step1.append(data_temp)
        elif data_i['EXAM'] == 'STEP 2':
            data_list_test_step2.append(data_temp)
        elif data_i['EXAM'] == 'STEP 3':
            data_list_test_step3.append(data_temp)

    # with open('test_final_type3.json', 'w') as f:
    #     json.dump(data_list_test, f, indent=4)
    with open('test_final_type3_step1.json', 'w') as f:
        json.dump(data_list_test_step1, f, indent=4)
    with open('test_final_type3_step2.json', 'w') as f:
        json.dump(data_list_test_step2, f, indent=4)
    with open('test_final_type3_step3.json', 'w') as f:
        json.dump(data_list_test_step3, f, indent=4)

# preprocess_type2(data_train, data_test, data_gold)
preprocess_type3(data_train, data_test, data_gold)
