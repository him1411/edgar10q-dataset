import pandas as pd
import openai

df = pd.read_csv("test_data.csv") 
ip_list = list(df['input'])
op_list = list(df['output'])

prompt1 = '''Based on the example given below, Given an entity and a sentence conataining the entity, generate a phrase that decribes the enity in the sentence.

Input: $15.Issuance of common stock in May 2019 public offering at $243.00 per share, net of issuance costs of $15.
Output: Common stock public offering issuance costs 

Input: '''

prompt2 = "\nOutput: "

#insert api key
openai.api_key = ""

def chat_gpt_response(sample):
    final_prompt = prompt1 + sample + prompt2
#     print("final prompt----------------",final_prompt)
    response=openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages= [{"role": "user", "content": final_prompt}],
        temperature=0.7,
        max_tokens=128,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0)
    
    reply = response["choices"][0]["message"]["content"]
    return reply

count = 0

batch_size = 100  #Tunable parameter

while len(ip_list2) > 0:
    temp_list = ip_list2[:batch_size]
    temp_op_list = op_list2[:batch_size]    
    output_list = []

    for item in temp_list:
        output_list.append(chat_gpt_response(item))
#         output_list.append(sample_response(item))
    
    data = {'Input': temp_list, 'Ground_truth':temp_op_list ,'Output': output_list }
    df_ans = pd.DataFrame(data)
    df_ans.to_csv("output_shard_" + str(count) + ".csv")
    count+=1
    print("shard number:",count,"Done")
    ip_list2 = ip_list2[batch_size:]
    op_list2 = op_list2[batch_size:]
#     print("len ip List", len(ip_list2))