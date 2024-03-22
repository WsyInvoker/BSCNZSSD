import os
import argparse
import tkinter as tk
from tkinter import messagebox
from utils.data_utils import Tokenizer4Bert
import torch
from torch import optim
import random
import numpy as np
from transformers import BertModel
from models.bert_scl import BERT_SCL

# 立场列表
stance_mapping = {0: "反对", 1: "中立", 2: "支持"}
# 主题列表
weibo_topics = ["深圳禁摩限电", "春节放鞭炮", "iPhoneSE", "俄罗斯在叙利亚的反恐行动", "开放二胎"]
selected_topics = weibo_topics

# gpu_id = 0
# torch.cuda.set_device(gpu_id)
# os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class Instructor(object):
    def __init__(self, opt):
        self.opt = opt
        tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
        bert_proto = BertModel.from_pretrained(opt.pretrained_bert_name, return_dict=False)
        self.model = opt.model_class(opt,bert_proto, ).to(opt.device)
        print("using model: ",opt.model_name)
        print("running dataset: ", opt.dataset)
        if 'scl' in self.opt.model_name:
            print("SCL")
            # 调用zeroshot的.pth来预测
        elif 'cross' in self.opt.model_name:
            print("cross:")
            # 调用fewshot的.pth来预测
            model_state_dict = torch.load('test_outputs/bert-cross-topic/weibo/2024-03-16 22-18-29/state_dict/my_model.pth')
            # 加载模型
            model = BERT_SCL(self.opt, bert_proto).to(opt.device)
            model.load_state_dict(model_state_dict)
            # model.eval()
                
        print(type(model))
        # 确认模型具有 forward 方法
        if hasattr(model, 'forward'):
            print("Model has a forward method.")
        else:
            print("Model does not have a forward method.")

        # 创建 GUI 窗口
        root = tk.Tk()
        root.title("中文立场检测")
        # 创建评论输入框
        entry_comment = tk.Entry(root, width=50)
        entry_comment.pack(padx=10, pady=10)
        # 创建主题输入框（下拉菜单）
        variable = tk.StringVar(root)
        variable.set(selected_topics[0])  # 设置默认选中的主题
        dropdown_topic = tk.OptionMenu(root, variable, *selected_topics)
        dropdown_topic.pack(pady=10)
        # 创建预测按钮
        predict_button = tk.Button(root, text="预测立场", command=lambda: self.on_predict(tokenizer, entry_comment, variable, model))
        command_str = predict_button.cget('command')
        print(command_str)
        predict_button.pack(pady=5)
        print("button created!")

        # 运行 GUI 循环
        root.mainloop()

    # 预测立场并显示    
    def on_predict(self, tokenizer, entry_comment, variable, model):
        comment = entry_comment.get()
        print(type(comment)," comment: ", comment)
        tokenized_comment = tokenizer.text_to_sequence("[CLS] " + comment + " [SEP]")   # 获取评论并分词
        topic = variable.get()
        tokenized_topic = tokenizer.text_to_sequence("[CLS] " + topic + " [SEP]")   # 获取选中的主题并分词

        tokenized_comment_tensor = torch.tensor(tokenized_comment).unsqueeze(0)#.cuda()
        bert_segments_ids_tensor = torch.zeros_like(tokenized_comment_tensor)#.cuda()
        # 将 bert_segments_ids 转换为张量
        # bert_segments_ids_tensor = torch.tensor([[0 if i < len(seq) else 1 for i in range(self.opt.max_seq_len)] for seq in bert_segments_ids])
        # 将两个张量放入列表中
        input_features = [tokenized_comment_tensor, bert_segments_ids_tensor]
        # 打印输入特征的格式
        print(f"item type of input_features:  {type(input_features)}")
        print("input_features:")
        for i, feature in enumerate(input_features):
            print(f"Index: {i}, Feature: {feature}")
        with torch.no_grad():
            logits = model(input_features)
        
        # print(type(tokenized_comment))
        # input_ids = np.concatenate((tokenized_comment, tokenized_topic), axis=0)   

        # 将评论和主题转化为Tensor并移到gpu 错错错错错
        # 和训练程序一样，此处不需要转换为tensor!
        # tokenized_comment = torch.tensor(tokenized_comment).cuda()
        # tokenized_topic = torch.tensor(tokenized_topic).cuda()

        # 创建注意力掩码  也不能以Tensor形式创建而要以numpy数组形式
        # attention_mask = torch.ones_like(tokenized_comment)
        # attention_mask = attention_mask.to(tokenized_comment.device)
        # attention_mask = np.ones_like(tokenized_comment, dtype=np.float32)
        # 再转移到cuda上备用
        # attention_mask = torch.from_numpy(attention_mask).cuda()

        # print("tokenized_comment: ", tokenized_comment)
        # print("tokenized_comment type: ",type(tokenized_comment))
        # print("Length of tokenized_comment:", len(tokenized_comment))


        # input_ids_comment = torch.tensor(tokenized_comment, dtype=torch.long)
        # input_ids_topic = torch.tensor(tokenized_topic, dtype=torch.long)
        # input_ids = [input_ids_comment, input_ids_topic,]

        # tokenized_comment_str = ','.join(map(str, tokenized_comment))
        # tokenized_topic_str = ','.join(map(str, tokenized_topic))

        # 将字符串转换为PyTorch张量
        # input_ids_comment = torch.tensor([int(x) for x in tokenized_comment_str.split()], dtype=torch.long)
        # input_ids_topic = torch.tensor([int(x) for x in tokenized_topic_str.split()], dtype=torch.long)
        # 将两个张量拼接在一起
        # input_ids = torch.cat((input_ids_comment.unsqueeze(0), input_ids_topic.unsqueeze(0)), dim=0)
        
        # input_ids = torch.tensor([tokenized_comment, tokenized_topic])
        # input_ids = torch.cat((input_ids_comment, input_ids_topic), dim=0)

        # input_features = [batch[feat_name].to(self.opt.device) for feat_name in self.opt.input_features]

        # input_ids_comment = torch.tensor(tokenized_comment).unsqueeze(0)#.cuda()  # 添加batch维度
        # input_ids_topic = torch.tensor(tokenized_topic).unsqueeze(0)#.cuda()  # 添加batch维度
        # input_ids = torch.cat((input_ids_comment, input_ids_topic))
        # input_ids = (input_ids_comment,input_ids_topic)
        # print("inputs: ", input_ids)
        # print("inputs.max: ", input_ids.max())
        # print("inputs.min: ", input_ids.min())

        # logits = self.model((input_ids_comment,input_ids_topic))
        # print("max_seq_len: ", tokenized_comment.size)
        # logits = model.forward_predict(input_ids, tokenized_comment.size)
        # print("input_ids.shape: ",input_ids.shape)
        
        # print("Length of inputs:", len(input_ids))
        # print("inputs type: ",type(input_ids))
        
        
        # with torch.no_grad():
        #     logits = model.forward(input_ids)
        # with torch.no_grad():
        #     logits = model.forward(tokenized_comment)


        print("logits: ", logits)
        prediction = logits[1].argmax().item()
        stance = stance_mapping[prediction]  # 将数字映射到汉字描述
        messagebox.showinfo("结果", f"立场预测: {stance}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert-cross-topic', type=str,required=False)    # bert-cross-topic, bert-scl-prototype-graph
    parser.add_argument('--type', default=1, help='2 for all,0 for zero shot ,1 for few shot',type=str, required=False)
    parser.add_argument('--dataset', default='weibo', type=str,required=False)  #数据集选择
    parser.add_argument('--polarities', default='weibo_bs', nargs='+', help="if just two polarity switch to ['positive', 'negtive']",required=False)     #立场极性选择
    parser.add_argument('--optimizer', default='adam', type=str,required=False)
    parser.add_argument('--temperature', default=0.07, type=float,required=False)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str,required=False)
    parser.add_argument('--lr', default=5e-5, type=float, help='try 5e-5, 2e-5, 1e-3 for others',required=False)
    parser.add_argument('--dropout', default=0.1, type=float,required=False)
    parser.add_argument('--l2reg', default=1e-5, type=float,required=False)
    parser.add_argument('--log_step', default=10, type=int,required=False)
    # parser.add_argument('--log_path', default="./log", type=str,required=False)
    parser.add_argument('--embed_dim', default=300, type=int,required=False)
    parser.add_argument('--hidden_dim', default=128, type=int,required=False,help="lstm encoder hidden size")
    parser.add_argument('--feature_dim', default=2*128, type=int,required=False,help="feature dim after encoder depends on encoder")
    parser.add_argument('--output_dim', default=64, type=int,required=False)
    parser.add_argument('--relation_dim',default=100,type=int,required=False)
    parser.add_argument('--bert_dim', default=768, type=int,required=False)
    parser.add_argument('--pretrained_bert_name', default='bert-base-chinese', type=str,required=False)      #bert-base-chinese
    parser.add_argument('--max_seq_len', default=85, type=int,required=False)
    parser.add_argument('--stance_loss_weight',default=0.5,type=float,required=False)
    parser.add_argument('--prototype_loss_weight',default=0.01,type=float,required=False)
    parser.add_argument('--alpha', default=0.8, type=float,required=False)
    parser.add_argument('--beta', default=1.2, type=float,required=False)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0',required=False)
    parser.add_argument('--seed', default=1, type=int, help='set seed for reproducibility')
    parser.add_argument("--batch_size", default=1, type=int, required=False)
    parser.add_argument("--eval_batch_size", default=1, type=int, required=False)
    parser.add_argument("--epochs", default=2, type=int, required=False)        #15
    parser.add_argument("--eval_steps", default=10, type=int, required=False)
    parser.add_argument("--cluster_times", default=1, type=int, required=False)
    # graph para
    parser.add_argument('--gnn_dims', default='192,192', type=str,required=False)
    parser.add_argument('--att_heads', default='4,4', type=str,required=False)
    parser.add_argument('--dp', default=0.1, type=float)
    opt = parser.parse_args()

    if opt.seed:
        set_seed(opt.seed)
    model_classes = {
        # 'bert-scl-prototype-graph': BERT_SCL_Proto_Graph,
        'bert-cross-topic': BERT_SCL,
    }
    input_features = {
        'bert-cross-topic': ['concat_bert_indices', 'concat_segments_indices'], # for cross-topic
        'bert-scl-prototype-graph':['concat_bert_indices', 'concat_segments_indices'], # for zero-shot
    }
    dataset_files = {
        'weibo': {
            'train': './datasets/train.txt',
            'test': './datasets/test.txt',
        },
    }
    polarities = {
        'weibo_bs': [0, 1, 2],
    }
    optimizers = {
        'adam':optim.Adam,
    }
    opt.device = torch.device('cpu')     # 'cuda' if torch.cuda.is_available() else 'cpu'
    opt.n_gpus = torch.cuda.device_count()
    opt.polarities = polarities[opt.polarities]
    opt.num_labels = len(opt.polarities)
    opt.model_class = model_classes[opt.model_name]
    opt.optim_class = optimizers[opt.optimizer]
    opt.input_features = input_features[opt.model_name]
    opt.dataset_files = dataset_files[opt.dataset]
    print("opt: ",opt)
    ins = Instructor(opt)