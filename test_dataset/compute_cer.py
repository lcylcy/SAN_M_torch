ground_truth = "aishell_text"
infer = "aishell_text_infer"

def normal_leven(str1, str2):
    len_str1 = len(str1) + 1
    len_str2 = len(str2) + 1
    matrix = [0 for n in range(len_str1 * len_str2)]
    for i in range(len_str1):
        matrix[i] = i
    for j in range(0, len(matrix), len_str1):
        if j % len_str1 == 0:
            matrix[j] = j // len_str1
    for i in range(1, len_str1):
        for j in range(1, len_str2):
            if str1[i - 1] == str2[j - 1]:
                cost = 0
            else:
                cost = 1
            matrix[j * len_str1 + i] = min(matrix[(j - 1) * len_str1 + i] + 1,
                                           matrix[j * len_str1 + (i - 1)] + 1,
                                           matrix[(j - 1) * len_str1 + (i - 1)] + cost)
    return matrix[-1]


url_true_text = {}
url_infer_text={}
with open(ground_truth,'r',encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        url,true_text = line.strip().split(',')
        true_text = true_text.replace('@','').replace(' ','')
        url_true_text[url] = true_text

with open(infer,'r',encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        a_list = line.strip().split()
        url = a_list[0]
        infer_text = a_list[1].replace('@','')
        url_infer_text[url] = infer_text


total_num=0
err=0

for key in url_true_text:
    if key in url_infer_text:
        true_text = url_true_text[key]
        infer_text = url_infer_text[key]
        print(true_text)
        print(infer_text)
        total_num += len(true_text)
        error = normal_leven(true_text,infer_text)
        err+=error

print(err/total_num)

