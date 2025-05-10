import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline, AutoModel
import gradio as gr
import re
import os
import json
import chardet
from sklearn.metrics import precision_score, recall_score, f1_score
import time
from functools import lru_cache  # 添加这行导入
# ======================== 数据库模块 ========================
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
import logging
import networkx as nx
from pyvis.network import Network
import pandas as pd
import matplotlib.pyplot as plt

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 使用SQLAlchemy的连接池来管理数据库连接
DATABASE_URL = "mysql+pymysql://user:password@host/dbname"  # 请根据实际情况修改连接字符串

# 创建引擎（连接池）
engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20, echo=True)

# 创建session类
Session = sessionmaker(bind=engine)

@contextmanager
def get_db_connection():
    """
    使用上下文管理器获取数据库连接
    """
    session = None
    try:
        session = Session()  # 从连接池中获取一个连接
        logging.info("✅ 数据库连接已建立")
        yield session  # 使用session进行数据库操作
    except Exception as e:
        logging.error(f"❌ 数据库操作时发生错误: {e}")
        if session:
            session.rollback()  # 回滚事务
    finally:
        if session:
            try:
                session.commit()  # 提交事务
                logging.info("✅ 数据库事务已提交")
            except Exception as e:
                logging.error(f"❌ 提交事务时发生错误: {e}")
            finally:
                session.close()  # 关闭会话，释放连接
                logging.info("✅ 数据库连接已关闭")

def save_to_db(table, data):
    """
    将数据保存到数据库
    :param table: 表名
    :param data: 数据字典
    """
    try:
        valid_tables = ["entities", "relations"]  # 只允许保存到这些表
        if table not in valid_tables:
            raise ValueError(f"Invalid table: {table}")
        
        with get_db_connection() as conn:
            if conn:
                # 这里的操作假设使用了ORM模型来处理插入，实际根据你数据库的表结构来调整
                table_model = get_table_model(table)  # 假设你有一个方法来根据表名获得ORM模型
                new_record = table_model(**data)
                conn.add(new_record)
                conn.commit()  # 提交事务
    except Exception as e:
        logging.error(f"❌ 保存数据时发生错误: {e}")
        return False
    return True

def get_table_model(table_name):
    """
    根据表名获取ORM模型（这里假设你有一个映射到数据库表的模型）
    :param table_name: 表名
    :return: 对应的ORM模型
    """
    if table_name == "entities":
        from models import Entity  # 假设你已经定义了ORM模型
        return Entity
    elif table_name == "relations":
        from models import Relation  # 假设你已经定义了ORM模型
        return Relation
    else:
        raise ValueError(f"Unknown table: {table_name}")


# ======================== 模型加载 ========================
NER_MODEL_NAME = "uer/roberta-base-finetuned-cluener2020-chinese"

@lru_cache(maxsize=1)
def get_ner_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_NAME)
    return pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="first"
    )

@lru_cache(maxsize=1)
def get_re_pipeline():
    return pipeline(
        "text2text-generation",
        model=NER_MODEL_NAME,
        tokenizer=NER_MODEL_NAME,
        max_length=512,
        device=0 if torch.cuda.is_available() else -1
    )


# chatglm_model, chatglm_tokenizer = None, None
# use_chatglm = False
# try:
#     chatglm_model_name = "THUDM/chatglm-6b-int4"
#     chatglm_tokenizer = AutoTokenizer.from_pretrained(chatglm_model_name, trust_remote_code=True)
#     chatglm_model = AutoModel.from_pretrained(
#         chatglm_model_name,
#         trust_remote_code=True,
#         device_map="cpu",
#         torch_dtype=torch.float32
#     ).eval()
#     use_chatglm = True
#     print("✅ 4-bit量化版ChatGLM加载成功")
# except Exception as e:
#     print(f"❌ ChatGLM加载失败: {e}")

# ======================== 知识图谱结构 ========================
knowledge_graph = {"entities": set(), "relations": set()}


# 优化知识图谱更新函数，增加全局变量更新
def update_knowledge_graph(entities, relations):
    """
    更新知识图谱并保存到数据库
    """
    global knowledge_graph  # 明确声明使用全局变量
    # 保存实体
    for e in entities:
        if isinstance(e, dict) and 'text' in e and 'type' in e:
            save_to_db('entities', {
                'text': e['text'],
                'type': e['type'],
                'start_pos': e.get('start', -1),
                'end_pos': e.get('end', -1),
                'source': 'user_input'
            })
            knowledge_graph["entities"].add((e['text'], e['type']))

    # 保存关系
    for r in relations:
        if isinstance(r, dict) and all(k in r for k in ("head", "tail", "relation")):
            save_to_db('relations', {
                'head_entity': r['head'],
                'tail_entity': r['tail'],
                'relation_type': r['relation'],
                'source_text': ''  # 可添加原文关联
            })
            knowledge_graph["relations"].add((r['head'], r['tail'], r['relation']))


# 优化知识图谱文本格式生成函数，增加排序和去重
def visualize_kg_text():
    """
    生成知识图谱的文本格式
    """
    nodes = sorted(set([f"{ent[0]} ({ent[1]})" for ent in knowledge_graph["entities"]]))
    edges = sorted(set([f"{h} --[{r}]-> {t}" for h, t, r in knowledge_graph["relations"]]))
    return "\n".join(["📌 实体:"] + nodes + ["", "📎 关系:"] + edges)


# 优化知识图谱可视化函数，动态生成HTML文件名，避免覆盖
def visualize_kg_interactive(entities, relations):
    """
    生成交互式的知识图谱可视化
    """
    # 创建一个新的网络图
    net = Network(height="700px", width="100%", bgcolor="#ffffff", font_color="black")
    
    # 定义实体类型颜色
    entity_colors = {
        'PER': '#FF6B6B',  # 人物-红色
        'ORG': '#4ECDC4',  # 组织-青色
        'LOC': '#45B7D1',  # 地点-蓝色
        'TIME': '#96CEB4', # 时间-绿色
        'TITLE': '#D4A5A5' # 职位-灰色
    }
    
    # 添加实体节点
    for entity in entities:
        node_color = entity_colors.get(entity['type'], '#D3D3D3')  # 默认灰色
        net.add_node(entity['text'], 
                     label=f"{entity['text']} ({entity['type']})",
                     color=node_color,
                     title=f"类型: {entity['type']}")
    
    # 添加关系边
    for relation in relations:
        net.add_edge(relation['head'], 
                     relation['tail'], 
                     label=relation['relation'], 
                     arrows='to')
    
    # 设置物理布局
    net.set_options('''
    var options = {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 100,
                "springConstant": 0.08
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": {"iterations": 150}
        }
    }
    ''')
    
    # 动态生成HTML文件名
    timestamp = int(time.time())
    html_path = f"knowledge_graph_{timestamp}.html"
    net.save_graph(html_path)
    return html_path

# ======================== 实体识别（NER） ========================
def merge_adjacent_entities(entities):
    if not entities:
        return entities

    merged = [entities[0]]
    for entity in entities[1:]:
        last = merged[-1]
        # 合并相邻的同类型实体
        if (entity["type"] == last["type"] and
                entity["start"] == last["end"]):
            last["text"] += entity["text"]
            last["end"] = entity["end"]
        else:
            merged.append(entity)

    return merged


def ner(text, model_type="bert"):
    start_time = time.time()

    # 如果使用的是 ChatGLM 模型，执行 ChatGLM 的NER
    if model_type == "chatglm" and use_chatglm:
        try:
            prompt = f"""请从以下文本中识别所有实体，严格按照JSON列表格式返回，每个实体包含text、type、start、end字段：
示例：[{{"text": "北京", "type": "LOC", "start": 0, "end": 2}}]
文本：{text}"""
            response = chatglm_model.chat(chatglm_tokenizer, prompt, temperature=0.1)
            if isinstance(response, tuple):
                response = response[0]

            try:
                json_str = re.search(r'\[.*\]', response, re.DOTALL).group()
                entities = json.loads(json_str)
                valid_entities = [ent for ent in entities if all(k in ent for k in ("text", "type", "start", "end"))]
                return valid_entities, time.time() - start_time
            except Exception as e:
                print(f"JSON解析失败: {e}")
                return [], time.time() - start_time
        except Exception as e:
            print(f"ChatGLM调用失败: {e}")
            return [], time.time() - start_time

    # 使用BERT NER
    text_chunks = [text[i:i + 510] for i in range(0, len(text), 510)]  # 安全分段
    raw_results = []
    
    # 获取NER pipeline
    ner_pipeline = get_ner_pipeline()  # 使用缓存的pipeline
    
    for idx, chunk in enumerate(text_chunks):
        chunk_results = ner_pipeline(chunk)  # 使用获取的pipeline
        for r in chunk_results:
            r["start"] += idx * 510
            r["end"] += idx * 510
        raw_results.extend(chunk_results)

    entities = [{
        "text": r['word'].replace(' ', ''),
        "start": r['start'],
        "end": r['end'],
        "type": LABEL_MAPPING.get(r.get('entity_group') or r.get('entity'), r.get('entity_group') or r.get('entity'))
    } for r in raw_results]

    entities = merge_adjacent_entities(entities)
    return entities, time.time() - start_time


# ------------------ 实体类型标准化 ------------------
LABEL_MAPPING = {
    "address": "LOC",
    "company": "ORG",
    "name": "PER",
    "organization": "ORG",
    "position": "TITLE",
    "government": "ORG",
    "scene": "LOC",
    "book": "WORK",
    "movie": "WORK",
    "game": "WORK"
}

# 提取实体
entities, processing_time = ner("Google in New York met Alice")

# 标准化实体类型
for e in entities:
    e["type"] = LABEL_MAPPING.get(e.get("type"), e.get("type"))

# 打印标准化后的实体
print(f"[DEBUG] 标准化后实体列表: {[{'text': e['text'], 'type': e['type']} for e in entities]}")

# 打印处理时间
print(f"处理时间: {processing_time:.2f}秒")


# ======================== 关系抽取（RE） ========================
@lru_cache(maxsize=1)
def get_re_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_NAME)
    return pipeline(
        "ner",  # 使用NER pipeline
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="first"
    )

def re_extract(entities, text, use_bert_model=True):
    if not entities or not text:
        return [], 0
    
    start_time = time.time()
    try:
        # 使用规则匹配关系
        relations = []
        
        # 定义关系关键词和对应的实体类型约束
        relation_rules = {
            "位于": {
                "keywords": ["位于", "在", "坐落于"],
                "valid_types": {
                    "head": ["ORG", "PER", "LOC"],
                    "tail": ["LOC"]
                }
            },
            "属于": {
                "keywords": ["属于", "是", "为"],
                "valid_types": {
                    "head": ["ORG", "PER"],
                    "tail": ["ORG", "LOC"]
                }
            },
            "任职于": {
                "keywords": ["任职于", "就职于", "工作于"],
                "valid_types": {
                    "head": ["PER"],
                    "tail": ["ORG"]
                }
            }
        }
        
        # 预处理实体，去除重复和部分匹配
        processed_entities = []
        for e in entities:
            # 检查是否与已有实体重叠
            is_subset = False
            for pe in processed_entities:
                if e["text"] in pe["text"] and e["text"] != pe["text"]:
                    is_subset = True
                    break
            if not is_subset:
                processed_entities.append(e)
        
        # 遍历文本中的每个句子
        sentences = re.split('[。！？.!?]', text)
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            # 获取当前句子中的实体
            sentence_entities = [e for e in processed_entities if e["text"] in sentence]
            
            # 检查每个关系类型
            for rel_type, rule in relation_rules.items():
                for keyword in rule["keywords"]:
                    if keyword in sentence:
                        # 在句子中查找符合类型约束的实体对
                        for i, ent1 in enumerate(sentence_entities):
                            for j, ent2 in enumerate(sentence_entities):
                                if i != j:  # 避免自循环
                                    # 检查实体类型是否符合规则
                                    if (ent1["type"] in rule["valid_types"]["head"] and 
                                        ent2["type"] in rule["valid_types"]["tail"]):
                                        # 检查实体在句子中的位置关系
                                        if sentence.find(ent1["text"]) < sentence.find(ent2["text"]):
                                            relations.append({
                                                "head": ent1["text"],
                                                "tail": ent2["text"],
                                                "relation": rel_type
                                            })
        
        # 去重
        unique_relations = []
        seen = set()
        for rel in relations:
            rel_key = (rel["head"], rel["tail"], rel["relation"])
            if rel_key not in seen:
                seen.add(rel_key)
                unique_relations.append(rel)
        
        return unique_relations, time.time() - start_time
            
    except Exception as e:
        logging.error(f"关系抽取失败: {e}")
        return [], time.time() - start_time


# ======================== 文本分析主流程 ========================
def create_knowledge_graph(entities, relations):
    """
    创建知识图谱可视化（文本格式）
    """
    # 设置实体类型的颜色映射
    entity_colors = {
        'PER': '🔴',  # 人物-红色
        'ORG': '🔵',  # 组织-蓝色
        'LOC': '🟢',  # 地点-绿色
        'TIME': '🟡', # 时间-黄色
        'TITLE': '🟣' # 职位-紫色
    }
    
    # 生成实体列表
    entity_list = []
    for entity in entities:
        emoji = entity_colors.get(entity['type'], '⚪')
        entity_list.append(f"{emoji} {entity['text']} ({entity['type']})")
    
    # 生成关系列表
    relation_list = []
    for relation in relations:
        relation_list.append(f"{relation['head']} --[{relation['relation']}]--> {relation['tail']}")
    
    # 生成HTML内容
    html_content = f"""
    <div style="font-family: Arial, sans-serif; padding: 20px;">
        <h3 style="color: #333; margin-bottom: 15px;">📌 实体列表：</h3>
        <div style="margin-bottom: 20px;">
            {chr(10).join(f'<div style="margin: 5px 0;">{entity}</div>' for entity in entity_list)}
        </div>
        
        <h3 style="color: #333; margin-bottom: 15px;">📎 关系列表：</h3>
        <div>
            {chr(10).join(f'<div style="margin: 5px 0;">{relation}</div>' for relation in relation_list)}
        </div>
        
        <div style="margin-top: 20px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
            <h4 style="color: #666; margin-bottom: 10px;">图例说明：</h4>
            <div style="display: flex; gap: 15px; flex-wrap: wrap;">
                {chr(10).join(f'<div style="display: flex; align-items: center; gap: 5px;"><span>{emoji}</span><span>{label}</span></div>' for label, emoji in entity_colors.items())}
            </div>
        </div>
    </div>
    """
    
    return html_content

def process_text(text, model_type="bert"):
    """
    处理文本，进行实体识别和关系抽取
    """
    start_time = time.time()
    
    # 实体识别
    entities, ner_duration = ner(text, model_type)
    if not entities:
        return "", "", "", f"{time.time() - start_time:.2f} 秒"
    
    # 关系抽取
    relations, re_duration = re_extract(entities, text)
    
    # 生成文本格式的实体和关系描述
    ent_text = "📌 实体:\n" + "\n".join([f"{e['text']} ({e['type']})" for e in entities])
    rel_text = "\n\n📎 关系:\n" + "\n".join([f"{r['head']} --[{r['relation']}]--> {r['tail']}" for r in relations])
    
    # 生成知识图谱
    kg_text = create_knowledge_graph(entities, relations)
    
    total_duration = time.time() - start_time
    return ent_text, rel_text, kg_text, f"{total_duration:.2f} 秒"

# ======================== 知识图谱可视化 ========================
def generate_kg_image(entities, relations):
    """
    生成知识图谱的图片并保存到临时文件（Hugging Face适配版）
    """
    try:
        import tempfile
        import matplotlib.pyplot as plt
        import networkx as nx
        import os

        # === 1. 强制设置中文字体 ===
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']  # Hugging Face内置字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

        # === 2. 检查输入数据 ===
        if not entities or not relations:
            return None

        # === 3. 创建图谱 ===
        G = nx.DiGraph()
        entity_colors = {
            'PER': '#FF6B6B',  # 红色
            'ORG': '#4ECDC4',  # 青色
            'LOC': '#45B7D1',  # 蓝色
            'TIME': '#96CEB4', # 绿色
            'TITLE': '#D4A5A5' # 灰色
        }

        # 添加节点（实体）
        for entity in entities:
            G.add_node(
                entity["text"],
                label=f"{entity['text']} ({entity['type']})",
                color=entity_colors.get(entity['type'], '#D3D3D3')
            )

        # 添加边（关系）
        for relation in relations:
            if relation["head"] in G.nodes and relation["tail"] in G.nodes:
                G.add_edge(
                    relation["head"],
                    relation["tail"],
                    label=relation["relation"]
                )

        # === 4. 绘图配置 ===
        plt.figure(figsize=(12, 8), dpi=150)  # 降低DPI以节省内存
        pos = nx.spring_layout(G, k=0.7, seed=42)  # 固定随机种子保证布局稳定

        # 绘制节点和边
        nx.draw_networkx_nodes(
            G, pos,
            node_color=[G.nodes[n]['color'] for n in G.nodes],
            node_size=800
        )
        nx.draw_networkx_edges(
            G, pos,
            edge_color='#888888',
            width=1.5,
            arrows=True,
            arrowsize=20
        )

        # === 5. 绘制中文标签（关键修改点）===
        nx.draw_networkx_labels(
            G, pos,
            labels={n: G.nodes[n]['label'] for n in G.nodes},
            font_size=10,
            font_family='Noto Sans CJK SC'  # 显式指定字体
        )
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=nx.get_edge_attributes(G, 'label'),
            font_size=8,
            font_family='Noto Sans CJK SC'  # 显式指定字体
        )

        plt.axis('off')
        
        # === 6. 保存到临时文件 ===
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, "kg.png")
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()
        
        return file_path
        
    except Exception as e:
        logging.error(f"生成知识图谱图片失败: {str(e)}")
        return None


def process_file(file, model_type="bert"):
    try:
        with open(file.name, 'rb') as f:
            content = f.read()

        if len(content) > 5 * 1024 * 1024:
            return "❌ 文件太大", "", "", "", None

        # 检测编码
        try:
            encoding = chardet.detect(content)['encoding'] or 'utf-8'
            text = content.decode(encoding)
        except UnicodeDecodeError:
            # 尝试常见中文编码
            for enc in ['gb18030', 'utf-16', 'big5']:
                try:
                    text = content.decode(enc)
                    break
                except:
                    continue
            else:
                return "❌ 编码解析失败", "", "", "", None

        # 调用现有流程处理文本
        ent_text, rel_text, kg_text, duration = process_text(text, model_type)
        
        # 生成知识图谱图片
        entities, _ = ner(text, model_type)
        relations, _ = re_extract(entities, text)
        kg_image_path = generate_kg_image(entities, relations)  # 返回文件路径
        
        return ent_text, rel_text, kg_text, duration, kg_image_path
        
    except Exception as e:
        logging.error(f"文件处理错误: {str(e)}")
        return f"❌ 文件处理错误: {str(e)}", "", "", "", None


# ======================== 模型评估与自动标注 ========================
def convert_telegram_json_to_eval_format(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "text" in data:
        return [{"text": data["text"], "entities": [
            {"text": data["text"][e["start"]:e["end"]]} for e in data.get("entities", [])
        ]}]
    elif isinstance(data, list):
        return data
    elif isinstance(data, dict) and "messages" in data:
        result = []
        for m in data.get("messages", []):
            if isinstance(m.get("text"), str):
                result.append({"text": m["text"], "entities": []})
            elif isinstance(m.get("text"), list):
                txt = ''.join([x["text"] if isinstance(x, dict) else x for x in m["text"]])
                result.append({"text": txt, "entities": []})
        return result
    return []


def evaluate_ner_model(data, model_type):
    tp, fp, fn = 0, 0, 0
    POS_TOLERANCE = 1

    for item in data:
        text = item["text"]
        # 处理标注数据
        gold_entities = validate_gold_entities([
            {
                "text": e["text"],
                "type": LABEL_MAPPING.get(e["type"], e["type"]),
                "start": e.get("start", -1),
                "end": e.get("end", -1)
            }
            for e in item.get("entities", [])
        ])

        # 获取预测结果
        pred_entities, _ = ner(text, model_type)

        # 初始化匹配状态
        matched_gold = [False] * len(gold_entities)
        matched_pred = [False] * len(pred_entities)

        # 遍历预测实体寻找匹配
        for p_idx, p in enumerate(pred_entities):
            for g_idx, g in enumerate(gold_entities):
                if not matched_gold[g_idx] and \
                        p["text"] == g["text"] and \
                        p["type"] == g["type"] and \
                        abs(p["start"] - g["start"]) <= POS_TOLERANCE and \
                        abs(p["end"] - g["end"]) <= POS_TOLERANCE:
                    matched_gold[g_idx] = True
                    matched_pred[p_idx] = True
                    break

        # 统计指标
        tp += sum(matched_pred)
        fp += len(pred_entities) - sum(matched_pred)
        fn += len(gold_entities) - sum(matched_gold)

    # 处理除零情况
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return (f"Precision: {precision:.2f}\n"
            f"Recall: {recall:.2f}\n"
            f"F1: {f1:.2f}")


def auto_annotate(file, model_type):
    data = convert_telegram_json_to_eval_format(file.name)
    for item in data:
        ents, _ = ner(item["text"], model_type)
        item["entities"] = ents
    return json.dumps(data, ensure_ascii=False, indent=2)


def save_json(json_text):
    fname = f"auto_labeled_{int(time.time())}.json"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(json_text)
    return fname


# ======================== 数据集导入 ========================
def import_dataset(path="D:/云边智算/暗语识别/filtered_results"):
    import os
    import json

    for filename in os.listdir(path):
        if filename.endswith('.json'):
            filepath = os.path.join(path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 调用现有处理流程
                process_text(data['text'])
                print(f"已处理文件: {filename}")


# ======================== Gradio 界面 ========================
with gr.Blocks(css="""
    .kg-graph {height: 700px; overflow-y: auto;}
    .warning {color: #ff6b6b;}
    .error {color: #ff0000; padding: 10px; background-color: #ffeeee; border-radius: 5px;}
""") as demo:
    gr.Markdown("# 🤖 聊天记录实体关系识别系统")

    with gr.Tab("📄 文本分析"):
        input_text = gr.Textbox(lines=6, label="输入文本")
        model_type = gr.Radio(["bert", "chatglm"], value="bert", label="选择模型")
        btn = gr.Button("开始分析")
        out1 = gr.Textbox(label="识别实体")
        out2 = gr.Textbox(label="识别关系")
        out3 = gr.HTML(label="知识图谱")  # 使用HTML组件显示文本格式的知识图谱
        out4 = gr.Textbox(label="耗时")
        btn.click(fn=process_text, inputs=[input_text, model_type], outputs=[out1, out2, out3, out4])

    with gr.Tab("🗂 文件分析"):
        file_input = gr.File(file_types=[".txt", ".json"])
        file_btn = gr.Button("上传并分析")
        fout1, fout2, fout3, fout4, fout5 = gr.Textbox(), gr.Textbox(), gr.Textbox(), gr.Textbox(), gr.File(label="下载知识图谱图片")
        file_btn.click(fn=process_file, inputs=[file_input, model_type], outputs=[fout1, fout2, fout3, fout4, fout5])

    with gr.Tab("📊 模型评估"):
        eval_file = gr.File(label="上传标注 JSON")
        eval_model = gr.Radio(["bert", "chatglm"], value="bert")
        eval_btn = gr.Button("开始评估")
        eval_output = gr.Textbox(label="评估结果", lines=5)
        eval_btn.click(lambda f, m: evaluate_ner_model(convert_telegram_json_to_eval_format(f.name), m),
                       [eval_file, eval_model], eval_output)

    with gr.Tab("✏️ 自动标注"):
        raw_file = gr.File(label="上传 Telegram 原始 JSON")
        auto_model = gr.Radio(["bert", "chatglm"], value="bert")
        auto_btn = gr.Button("自动标注")
        marked_texts = gr.Textbox(label="标注结果", lines=20)
        download_btn = gr.Button("💾 下载标注文件")
        auto_btn.click(fn=auto_annotate, inputs=[raw_file, auto_model], outputs=marked_texts)
        download_btn.click(fn=save_json, inputs=marked_texts, outputs=gr.File())

    with gr.Tab("📂 数据管理"):
        gr.Markdown("### 数据集导入")
        dataset_path = gr.Textbox(
            value="D:/云边智算/暗语识别/filtered_results",
            label="数据集路径"
        )
        import_btn = gr.Button("导入数据集到数据库")
        import_output = gr.Textbox(label="导入日志")
        import_btn.click(fn=lambda: import_dataset(dataset_path.value), outputs=import_output)

demo.launch(server_name="0.0.0.0", server_port=7860)