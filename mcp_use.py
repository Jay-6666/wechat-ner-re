import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'relation-extraction-master'))
import re
import torch
from gqlalchemy import Memgraph
from relation_extraction.hparams import hparams
from relation_extraction.model import SentenceRE
from relation_extraction.data_utils import MyTokenizer, get_idx2tag, convert_pos_to_mask

# 云端Memgraph连接参数
MEMGRAPH_HOST = '18.159.132.161'
MEMGRAPH_PORT = 7687
MEMGRAPH_USERNAME = 'b300000.de@gmail.com'
MEMGRAPH_PASSWORD = '159951Tjk.'  # 请替换为你的真实密码
MEMGRAPH_ENCRYPTED = True

# 连接memgraph云数据库
def get_memgraph_conn():
    return Memgraph(
        MEMGRAPH_HOST,
        MEMGRAPH_PORT,
        MEMGRAPH_USERNAME,
        MEMGRAPH_PASSWORD,
        encrypted=MEMGRAPH_ENCRYPTED
    )

# 单句预测，返回三元组
class RelationPredictor:
    def __init__(self, hparams):
        self.device = hparams.device
        torch.manual_seed(hparams.seed)
        self.idx2tag = get_idx2tag(hparams.tagset_file)
        hparams.tagset_size = len(self.idx2tag)
        self.model = SentenceRE(hparams).to(self.device)
        self.model.load_state_dict(torch.load(hparams.model_file))
        self.model.eval()
        self.tokenizer = MyTokenizer(hparams.pretrained_model_path)

    def predict_one(self, text, entity1, entity2):
        match_obj1 = re.search(entity1, text)
        match_obj2 = re.search(entity2, text)
        if not (match_obj1 and match_obj2):
            return None
        e1_pos = match_obj1.span()
        e2_pos = match_obj2.span()
        item = {
            'h': {'name': entity1, 'pos': e1_pos},
            't': {'name': entity2, 'pos': e2_pos},
            'text': text
        }
        tokens, pos_e1, pos_e2 = self.tokenizer.tokenize(item)
        encoded = self.tokenizer.bert_tokenizer.batch_encode_plus([(tokens, None)], return_tensors='pt')
        input_ids = encoded['input_ids'].to(self.device)
        token_type_ids = encoded['token_type_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        e1_mask = torch.tensor([convert_pos_to_mask(pos_e1, max_len=attention_mask.shape[1])]).to(self.device)
        e2_mask = torch.tensor([convert_pos_to_mask(pos_e2, max_len=attention_mask.shape[1])]).to(self.device)
        with torch.no_grad():
            logits = self.model(input_ids, token_type_ids, attention_mask, e1_mask, e2_mask)[0]
            logits = logits.to(torch.device('cpu'))
        relation = self.idx2tag[logits.argmax(0).item()]
        return entity1, relation, entity2

# 写入memgraph
def insert_to_memgraph(memgraph, entity1, relation, entity2):
    memgraph.execute(
        "MERGE (a:Entity {name: $name1})",
        {"name1": entity1}
    )
    memgraph.execute(
        "MERGE (b:Entity {name: $name2})",
        {"name2": entity2}
    )
    memgraph.execute(
        f"MATCH (a:Entity {{name: $name1}}), (b:Entity {{name: $name2}}) MERGE (a)-[:{relation}]->(b)",
        {"name1": entity1, "name2": entity2}
    )

# 主流程

def main():
    memgraph = get_memgraph_conn()
    predictor = RelationPredictor(hparams)
    print("请输入句子和两个实体，识别关系并写入Memgraph。输入exit退出。")
    while True:
        text = input("输入中文句子：")
        if text.strip().lower() == 'exit':
            break
        entity1 = input("句子中的实体1：")
        if entity1.strip().lower() == 'exit':
            break
        entity2 = input("句子中的实体2：")
        if entity2.strip().lower() == 'exit':
            break
        result = predictor.predict_one(text, entity1, entity2)
        if result is None:
            print("实体未在句子中找到，请重试。")
            continue
        entity1, relation, entity2 = result
        insert_to_memgraph(memgraph, entity1, relation, entity2)
        print(f"已写入Memgraph：({entity1})-[:{relation}]->({entity2})")

if __name__ == '__main__':
    main()
