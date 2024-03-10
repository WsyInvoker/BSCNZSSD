# 假设您的数据集存储在一个名为 dataset.txt 的文件中
input_file = 'datasets/train-3000-seg-rmstopwords.txt'
output_file = 'datasets/train.txt'

# 打开原始文件并读取内容
with open(input_file, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
    lines = fin.readlines()

# 打开新的输出文件，准备写入修改后的内容
with open(output_file, 'w', encoding='utf-8', newline='\n') as fout:
    for i in range(0, len(lines), 3):  # 每隔三行进行一次循环
        # 读取评论原文
        text = lines[i].strip()
        # 读取并修改话题所属主题
        target = lines[i+1].strip().replace('iPhone   SE', 'iPhoneSE').replace('IphoneSE', 'iPhoneSE').replace('俄罗斯 叙利亚 反恐 行动', '俄罗斯 在 叙利亚 的 反恐 行动')
        # 读取立场标签,必须为0~n的整数
        polarity = lines[i+2].strip()
        int_polarity = int(polarity)
        new_polarity = int_polarity + 1
        
        # 写入修改后的内容到新文件
        fout.write(text + '\n')
        fout.write(target + '\n')
        fout.write(str(new_polarity) + '\n')