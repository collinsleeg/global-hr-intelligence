#!/usr/bin/env python3
"""
知识问答服务 - Web版（Flask）- 重新设计版
绿色主色调 + 出海元素 + 高级感 + 科技感
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import chromadb
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv
import anthropic
import jieba
from openai import OpenAI

# 设置 Transformers 离线模式以使用本地缓存的模型
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

load_dotenv(override=True)

app = Flask(__name__)
CORS(app)

# 配置
DB_PATH = "knowledge_db"
COLLECTION_NAME = "country_employment_guides"

# 初始化
client = None
collection = None
claude_client = None

 def init_services():
      """初始化服务"""
      global client, collection, claude_client

      # 初始化ChromaDB
      client = chromadb.PersistentClient(path=DB_PATH)

      # 使用与构建时相同的embedding函数
      openai_key = os.getenv('OPENAI_API_KEY')
      if openai_key:
          embedding_func = embedding_functions.OpenAIEmbeddingFunction(
              api_key=openai_key,
              model_name="text-embedding-3-small"
          )
      else:
          # 使用 ONNX 版本，不需要 sentence-transformers
          embedding_func = ONNXMiniLM_L6_V2()

      collection = client.get_collection(
          name=COLLECTION_NAME,
          embedding_function=embedding_func
      )

      # 初始化Claude
      claude_client = anthropic.Anthropic(
          api_key=os.getenv('ANTHROPIC_API_KEY')
      )

      print("✓ 服务初始化完成")


def query_knowledge_base(question, top_k=3):
    """查询知识库 - 智能混合检索（兼容旧接口）"""
    result = query_knowledge_base_with_status(question, top_k)
    return result.get('contexts', [])


def query_knowledge_base_with_status(question, top_k=3):
    """查询知识库 - 智能混合检索，返回详细状态信息"""
    import re
    import jieba

    # 支持的国家列表（知识库中有数据的国家）- 实际43个
    supported_countries = ['英国', '美国', '德国', '法国', '日本', '韩国', '新加坡', '中国香港', '中国台湾',
                           '巴西', '阿根廷', '墨西哥', '加拿大', '澳大利亚', '新西兰', '印度', '泰国',
                           '越南', '印度尼西亚', '菲律宾', '马来西亚', '土耳其', '沙特阿拉伯', '阿联酋',
                           '意大利', '西班牙', '荷兰', '比利时', '瑞士', '瑞典', '丹麦', '挪威',
                           '波兰', '俄罗斯', '南非', '埃及', '以色列', '卡塔尔',
                           '哈萨克斯坦', '乌兹别克斯坦', '吉尔吉斯斯坦', '塔吉克斯坦', '土库曼斯坦',
                           '吉尔吉斯共和国', '加纳', '匈牙利', '卢森堡', '保加利亚',
                           '拉脱维亚', '斯洛伐克', '秘鲁', '罗马尼亚', '阿尔及利亚',
                           '多米尼加共和国', '尼日利亚', '哥伦比亚', '哥斯达黎加',
                           '希腊', '马耳他', '巴基斯坦']
    
    # 国家名称别名映射（简称 -> 标准名）
    country_aliases = {
        '印尼': '印度尼西亚',
        '大马': '马来西亚',
        'UK': '英国',
        'USA': '美国',
        'US': '美国',
        'America': '美国',
        '德国': '德国',
        'Deutschland': '德国',
        '法国': '法国',
        '日本': '日本',
        '韩国': '韩国',
        '俄国': '俄罗斯',
        '澳洲': '澳大利亚',
    }
    
    # 常见的国家名列表（用于检测用户是否询问了不在支持列表中的国家）
    all_country_keywords = supported_countries + [
        '中国', '中国大陆', '朝鲜', '蒙古', '缅甸', '老挝', '柬埔寨', '伊朗', '伊拉克', '叙利亚', '约旦', '黎巴嫩',
        '哈萨克斯坦', '乌兹别克斯坦', '吉尔吉斯斯坦', '塔吉克斯坦', '土库曼斯坦',
        '也门', '阿曼', '科威特', '巴林', '卡塔尔', '利比亚', '突尼斯', '阿尔及利亚', '摩洛哥', '苏丹', '埃塞俄比亚',
        '肯尼亚', '坦桑尼亚', '乌干达', '赞比亚', '津巴布韦', '博茨瓦纳', '纳米比亚', '安哥拉', '莫桑比克', '马达加斯加',
        '毛里求斯', '塞舌尔', '尼日利亚', '加纳', '科特迪瓦', '塞内加尔', '喀麦隆', '刚果', '卢旺达', '布隆迪',
        '冰岛', '爱尔兰', '葡萄牙', '希腊', '奥地利', '芬兰', '卢森堡', '捷克', '斯洛伐克', '匈牙利', '罗马尼亚',
        '保加利亚', '塞尔维亚', '克罗地亚', '斯洛文尼亚', '乌克兰', '白俄罗斯', '立陶宛', '拉脱维亚', '爱沙尼亚',
        '巴基斯坦', '孟加拉', '斯里兰卡', '尼泊尔', '不丹', '马尔代夫', '阿富汗', '乌兹别克斯坦', '土库曼斯坦',
        '吉尔吉斯斯坦', '塔吉克斯坦', '格鲁吉亚', '阿塞拜疆', '亚美尼亚', '韩国', '朝鲜', '文莱', '老挝', '东帝汶',
        '巴布亚新几内亚', '斐济', '汤加', '萨摩亚', '瓦努阿图', '所罗门群岛', '基里巴斯', '瑙鲁', '帕劳', '图瓦卢',
        '古巴', '牙买加', '海地', '多米尼加', '巴哈马', '巴巴多斯', '特立尼达和多巴哥', '格林纳达', '圣卢西亚',
        '圣文森特和格林纳丁斯', '安提瓜和巴布达', '圣基茨和尼维斯', '伯利兹', '危地马拉', '洪都拉斯', '萨尔瓦多',
        '尼加拉瓜', '哥斯达黎加', '巴拿马', '哥伦比亚', '委内瑞拉', '厄瓜多尔', '秘鲁', '玻利维亚', '巴拉圭', '乌拉圭',
        '智利', '圭亚那', '苏里南', '法属圭亚那', '马尔维纳斯群岛', '格陵兰', '百慕大', '波多黎各', '关岛',
        '美属维尔京群岛', '英属维尔京群岛', '安圭拉', '蒙特塞拉特', '特克斯和凯科斯群岛', '开曼群岛',
        '阿鲁巴', '库拉索', '荷属圣马丁', '法属圣马丁', '瓜德罗普', '马提尼克', '留尼汪', '马约特', '法属波利尼西亚',
        '新喀里多尼亚', '瓦利斯和富图纳', '托克劳', '纽埃', '库克群岛', '皮特凯恩群岛', '圣诞岛', '科科斯群岛',
        '诺福克岛', '赫德岛和麦克唐纳群岛', '法属南部领地', '布韦岛', '南乔治亚和南桑威奇群岛', '英属印度洋领地',
        '安道尔', '摩纳哥', '列支敦士登', '圣马力诺', '梵蒂冈', '马耳他', '塞浦路斯', '摩尔多瓦', '黑山',
        '北马其顿', '波斯尼亚和黑塞哥维那', '阿尔巴尼亚', '科索沃', '直布罗陀', '根西岛', '泽西岛', '马恩岛',
        '法罗群岛', '奥兰群岛', '斯瓦尔巴群岛', '扬马延岛', '新西伯利亚群岛', '法兰士约瑟夫地群岛',
        '喀麦隆', '中非', '乍得', '刚果共和国', '刚果民主共和国', '赤道几内亚', '加蓬', '圣多美和普林西比',
        '科摩罗', '吉布提', '厄立特里亚', '索马里', '南苏丹', '贝宁', '布基纳法索', '佛得角', '冈比亚',
        '几内亚', '几内亚比绍', '利比里亚', '马里', '毛里塔尼亚', '尼日尔', '塞拉利昂', '多哥', '莱索托',
        '斯威士兰', '马拉维', '科摩罗', '马约特', '留尼汪', '圣赫勒拿', '阿森松', '特里斯坦-达库尼亚',
        '西撒哈拉', '索马里兰', '马耳他骑士团', '北塞浦路斯', '南奥塞梯', '阿布哈兹', '纳戈尔诺-卡拉巴赫',
        '德涅斯特河沿岸', '卢甘斯克', '顿涅茨克', '克里米亚', '塞瓦斯托波尔', '科索沃', '巴勒斯坦',
        '中华民国', '香港', '台湾', '澳门'
    ]

    target_country = None
    
    # 首先检查标准国家名
    for country in supported_countries:
        if country in question:
            target_country = country
            break
    
    # 然后检查别名
    if not target_country:
        for alias, standard in country_aliases.items():
            if alias in question:
                target_country = standard
                print(f"通过别名 '{alias}' 识别到国家: {standard}")
                break
    
    # 检查是否询问了不在支持列表中的国家
    if not target_country:
        for country in all_country_keywords:
            if country in question:
                print(f"问题中提到了不在支持列表中的国家 '{country}'")
                return {
                    'contexts': [],
                    'status': 'no_country',
                    'country': country
                }
    
    # 检查是否包含明显的"测试"或虚构内容
    test_keywords = ['火星', '月球', '测试', 'abcdefg', '不存在', '虚拟', '假的', '虚构', '幻想']
    for kw in test_keywords:
        if kw in question:
            print(f"检测到测试关键词 '{kw}'")
            return {
                'contexts': [],
                'status': 'fictional',
                'country': ''
            }

    # 如果指定了国家，先按国家过滤
    if target_country:
        print(f"检测到目标国家: {target_country}")
        # 获取该国所有文档
        country_docs = collection.get(
            where={'country': target_country},
            limit=100  # 获取该国所有文档
        )

        if not country_docs['documents']:
            # 没有该国数据
            print(f"知识库中没有 {target_country} 的数据")
            return {
                'contexts': [],
                'status': 'no_content',
                'country': target_country
            }

        # 提取问题关键词（包含分词和保留原始问题中的重要术语）
        keywords = list(jieba.cut(question))
        allowed_single_chars = ['年', '假', '税', '金', '费', '期']
        keywords = [k for k in keywords
                   if k not in ['什么', '哪些', '如何', '怎么', '多少', '为什么', '是否', '有没有', '的', '了', '吗', '呢', target_country, '？']
                   and (len(k) > 1 or k in allowed_single_chars)]

        # 额外检查问题中的HR关键术语（完整词组）
        hr_terms_in_question = []
        if '年假' in question:
            hr_terms_in_question.append('年假')
        if '试用期' in question or 'probation' in question.lower():
            hr_terms_in_question.append('试用期')
        if '工作时长' in question or '工作时间' in question:
            hr_terms_in_question.extend(['工作时长', '工作时间'])
        if '加班' in question:
            hr_terms_in_question.append('加班')
        if '工资' in question or '薪资' in question or '最低' in question:
            hr_terms_in_question.extend(['工资', '薪资', '最低'])
        if '合同' in question:
            hr_terms_in_question.append('合同')
        if '休假' in question or '假期' in question:
            hr_terms_in_question.extend(['休假', '假期'])
        if '社保' in question or '保险' in question:
            hr_terms_in_question.extend(['社保', '保险'])
        if '解雇' in question or '辞退' in question or '离职' in question:
            hr_terms_in_question.extend(['解雇', '辞退', '离职'])
        if '招聘' in question or '雇佣' in question:
            hr_terms_in_question.extend(['招聘', '雇佣'])
        if '个税' in question or '所得税' in question:
            hr_terms_in_question.extend(['个税', '所得税'])
        if '福利' in question:
            hr_terms_in_question.append('福利')
        if '工时' in question or '工时' in question:
            hr_terms_in_question.append('工时')
        if '病假' in question:
            hr_terms_in_question.append('病假')
        if '产假' in question:
            hr_terms_in_question.append('产假')
        if '陪产假' in question:
            hr_terms_in_question.append('陪产假')
        if '育儿假' in question:
            hr_terms_in_question.append('育儿假')
        if '法定节假日' in question or '公共假期' in question:
            hr_terms_in_question.extend(['法定节假日', '公共假期'])
        if '调休' in question:
            hr_terms_in_question.append('调休')
        if '遣散费' in question or '赔偿金' in question:
            hr_terms_in_question.extend(['遣散费', '赔偿金'])
        if '竞业禁止' in question or '保密协议' in question:
            hr_terms_in_question.extend(['竞业禁止', '保密协议'])
        if '工会' in question:
            hr_terms_in_question.append('工会')
        if '歧视' in question:
            hr_terms_in_question.append('歧视')
        if '安全' in question and '健康' in question:
            hr_terms_in_question.extend(['安全', '健康'])
        if '工伤' in question:
            hr_terms_in_question.append('工伤')
        if '移民' in question or '签证' in question or '工作许可' in question or '工作签证' in question:
            hr_terms_in_question.extend(['移民', '签证', '工作许可', '工作签证'])
        if '养老金' in question or '退休金' in question:
            hr_terms_in_question.extend(['养老金', '退休金'])
        if '医疗' in question:
            hr_terms_in_question.append('医疗')
        if '奖金' in question or '年终奖' in question or '十三薪' in question:
            hr_terms_in_question.extend(['奖金', '年终奖', '十三薪'])
        if '津贴' in question or '补贴' in question:
            hr_terms_in_question.extend(['津贴', '补贴'])
        if '报销' in question:
            hr_terms_in_question.append('报销')
        if '培训' in question:
            hr_terms_in_question.append('培训')
        if '绩效' in question:
            hr_terms_in_question.append('绩效')
        if '考勤' in question:
            hr_terms_in_question.append('考勤')
        if '远程工作' in question or '居家办公' in question:
            hr_terms_in_question.extend(['远程工作', '居家办公'])
        if '灵活工作' in question:
            hr_terms_in_question.append('灵活工作')
        if '最低工资' in question or '底薪' in question:
            hr_terms_in_question.extend(['最低工资', '底薪'])
        if '薪酬' in question:
            hr_terms_in_question.append('薪酬')
        if '待遇' in question:
            hr_terms_in_question.append('待遇')
        if '劳动' in question or '劳工' in question:
            hr_terms_in_question.extend(['劳动', '劳工'])
        if '雇佣' in question:
            hr_terms_in_question.append('雇佣')
        if '就业' in question:
            hr_terms_in_question.append('就业')
        if 'HR' in question or '人力资源' in question:
            hr_terms_in_question.extend(['HR', '人力资源'])
        if '合规' in question:
            hr_terms_in_question.append('合规')
        if '法律' in question and '劳动' in question:
            hr_terms_in_question.append('劳动法')
        if '法规' in question and ('劳动' in question or '雇佣' in question):
            hr_terms_in_question.append('劳动法规')
            
        # 如果问题中没有任何HR相关关键词，则认为不相关
        if not hr_terms_in_question:
            print(f"问题 '{question}' 不包含任何HR相关关键词，返回irrelevant")
            return {
                'contexts': [],
                'status': 'irrelevant',
                'country': target_country
            }

        # 对该国文档进行关键词评分
        scored_docs = []
        for i, (doc, meta) in enumerate(zip(country_docs['documents'], country_docs['metadatas'])):
            keyword_score = sum(1 for kw in keywords if kw in doc)

            # 特殊关键词加分：只对问题中提到的概念加分
            bonus = 0
            for term in hr_terms_in_question:
                if term in doc:
                    if term == '年假':
                        bonus += 25
                    elif '试用期' in term or 'probation' in term.lower():
                        bonus += 20
                    elif term == '加班':
                        bonus += 20
                    elif '工作' in term:
                        bonus += 15
                    elif '工资' in term or '薪资' in term or '最低' in term:
                        bonus += 15
                    elif '合同' in term:
                        bonus += 12

            # OCR内容包含问题时询问的关键词时，额外加分
            if meta.get('type') == 'ocr':
                # 只有当OCR包含完整的HR关键术语时才加分
                has_hr_term = any(term in doc for term in hr_terms_in_question)
                if has_hr_term:
                    # 但如果article有更高匹配，不给予OCR额外优势
                    bonus += 5  # 降低OCR加分

            total_score = keyword_score * 10 + bonus

            if total_score > 0:  # 只保留有相关性的
                scored_docs.append({
                    'doc': doc,
                    'metadata': meta,
                    'score': total_score
                })

        # 按得分排序
        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        
        # 设置最低相关性阈值 - 如果最高分低于阈值，说明问题与该国内容不相关
        MIN_RELEVANCE_THRESHOLD = 15  # 最低相关性阈值
        if not scored_docs or scored_docs[0]['score'] < MIN_RELEVANCE_THRESHOLD:
            print(f"{target_country} 的相关文档与问题相关性太低")
            return {
                'contexts': [],
                'status': 'irrelevant',
                'country': target_country
            }

        # 如果关键词匹配的结果太少，补充向量检索结果
        if len(scored_docs) < top_k:
            results = collection.query(
                query_texts=[question],
                n_results=top_k - len(scored_docs),
                where={'country': target_country}
            )
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i]
                scored_docs.append({
                    'doc': doc,
                    'metadata': metadata,
                    'score': 0
                })

        # 取top_k（补充后可能超过）
        scored_docs = scored_docs[:top_k]

        contexts = []
        for item in scored_docs:
            contexts.append({
                'text': item['doc'],
                'country': item['metadata'].get('country', 'Unknown'),
                'source': item['metadata'].get('title', ''),
                'url': item['metadata'].get('url', '')
            })
        
        return {
            'contexts': contexts,
            'status': 'found',
            'country': target_country
        }

    else:
        # 没有指定国家，使用标准向量检索 + 关键词增强
        results = collection.query(query_texts=[question], n_results=min(15, top_k * 5))

        keywords = list(jieba.cut(question))
        keywords = [k for k in keywords if len(k) > 1 and k not in ['什么', '哪些', '如何', '怎么', '多少', '为什么', '是否', '有没有', '的', '了', '吗', '呢']]

        scored_docs = []
        for i, doc in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i]
            keyword_score = sum(1 for kw in keywords if kw in doc)
            rank_score = len(results['documents'][0]) - i
            total_score = keyword_score * 3 + rank_score

            scored_docs.append({
                'doc': doc,
                'metadata': metadata,
                'score': total_score
            })

        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        
        # 设置最低相关性阈值，过滤低质量结果
        MIN_SCORE_THRESHOLD = 12  # 提高最低分数阈值，确保相关性
        
        # 如果最高分低于阈值，说明没有相关结果
        if not scored_docs or scored_docs[0]['score'] < MIN_SCORE_THRESHOLD:
            return {
                'contexts': [],
                'status': 'no_results',
                'country': ''
            }
        
        # 只保留达到阈值的结果
        scored_docs = [doc for doc in scored_docs if doc['score'] >= MIN_SCORE_THRESHOLD][:top_k]

        contexts = []
        for item in scored_docs:
            metadata = item['metadata']
            contexts.append({
                'text': item['doc'],
                'country': metadata.get('country', 'Unknown'),
                'source': metadata.get('title', ''),
                'url': metadata.get('url', '')
            })

        return {
            'contexts': contexts,
            'status': 'found',
            'country': contexts[0]['country'] if contexts else ''
        }

def generate_answer(question, contexts):
    """使用Claude生成答案 - 四部分结构：精准回答 + 更多参考 + 原文段落 + 文章链接"""
    # 构建prompt
    context_text = "\n\n---\n\n".join([
        f"【段落{i+1} - 来源：{ctx['country']} - {ctx['source']}】\n{ctx['text']}"
        for i, ctx in enumerate(contexts)
    ])

    prompt = f"""你是一个专业的国际HR顾问助手。请根据以下从国家用工指南中检索到的相关内容，回答用户的问题。

检索到的相关内容：
{context_text}

用户问题：{question}

回答格式要求（非常重要）：
请只输出以下两部分内容（系统会自动添加带标题的第三、四部分）：

【精准回答】
直接、简洁地回答问题核心，给出关键数据或规定，使用清晰的结构（如列表）。不要写"第一部分"标题。

【更多相关参考】
补充与问题相关的其他重要信息，如适用场景、注意事项、相关法规等。不要写"第二部分"标题。

注意：
- 如果检索内容确实包含答案，就明确回答，不要说"未找到"
- 如果检索内容与问题相关但不完全匹配，也要从中提取有用信息
- 知识库原文段落和原始文章链接会由系统自动添加，你不需要写

回答："""

    # AI 答案生成 - 优先级：DeepSeek > OpenAI > Claude > 智能提取
    import os
    deepseek_key = os.environ.get('DEEPSEEK_API_KEY')
    openai_key = os.environ.get('OPENAI_API_KEY')
    anthropic_key = os.environ.get('ANTHROPIC_API_KEY')

    # 优先使用 DeepSeek（便宜、支持国内支付）
    if deepseek_key:
        try:
            from openai import OpenAI
            deepseek_client = OpenAI(
                api_key=deepseek_key,
                base_url="https://api.deepseek.com"
            )
            response = deepseek_client.chat.completions.create(
                model="deepseek-chat",  # DeepSeek 的对话模型
                max_tokens=2000,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            if response.choices and len(response.choices) > 0:
                answer = response.choices[0].message.content
            else:
                answer = "抱歉，生成答案时出现问题。"
        except Exception as e:
            answer = f"抱歉，DeepSeek 生成答案时出错：{str(e)}"
    # 如果有OpenAI密钥，使用GPT-3.5生成答案
    elif openai_key:
        try:
            openai_client = OpenAI(api_key=openai_key)
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                max_tokens=2000,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            if response.choices and len(response.choices) > 0:
                answer = response.choices[0].message.content
            else:
                answer = "抱歉，生成答案时出现问题。"
        except Exception as e:
            answer = f"抱歉，OpenAI 生成答案时出错：{str(e)}"
    # 如果有API密钥，使用Claude生成答案
    elif anthropic_key:
        try:
            message = claude_client.messages.create(
                model="claude-sonnet-4.5-20240514",
                max_tokens=2000,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            # 安全获取答案
            if message.content and len(message.content) > 0:
                answer = message.content[0].text
            else:
                answer = "抱歉，生成答案时出现问题。"
        except Exception as e:
            answer = f"抱歉，Claude 生成答案时出错：{str(e)}"
    else:
        # 没有API密钥，使用智能提取逻辑
        # 只处理最相关的第一个段落
        if not contexts:
            return "抱歉，未找到相关信息。"

        primary_ctx = contexts[0]
        text = primary_ctx['text']

        # 清理OCR表格格式（删除多余空格）
        text = text.replace('  ', ' ').replace('   ', ' ').strip()

        # 提取问题关键词，用于选择最相关的段落
        question_keywords = list(jieba.cut(question))
        question_keywords = [k for k in question_keywords
                           if len(k) > 1 and k not in ['什么', '哪些', '如何', '怎么', '多少', '为什么', '是否', '有没有', '的', '了', '吗', '呢', '？']]

        # 按句子分割
        sentences = text.replace('。', '。|').replace('；', '；|').split('|')

        # 评分并选择句子
        scored_sentences = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 10:
                continue

            score = 0
            # 包含数字的句子优先（通常包含具体规定）
            if any(c.isdigit() for c in sent):
                score += 3
            # 包含关键词
            score += sum(1 for kw in question_keywords if kw in sent)
            # 常见HR关键词
            hr_keywords = ['工资', '年假', '试用期', '小时', '天', '周', '月', '小时', '美元', '欧元', '英镑']
            score += sum(1 for kw in hr_keywords if kw in sent)

            if score > 0:
                scored_sentences.append((sent, score))

        # 按得分排序，取前5句
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        selected = [s for s, score in scored_sentences[:5]]

        # 如果没找到好的句子，就取前几句
        if not selected:
            selected = [s.strip() for s in sentences[:3] if len(s.strip()) > 20]

        # 构建四部分结构（无API密钥时的简化版）
        answer = "## ✨ 第一部分：精准回答\n\n"
        for sent in selected:
            answer += f"- {sent}\n"
        
        answer += "\n## 📖 第二部分：更多相关参考\n\n"
        answer += "基于检索到的政策内容，建议关注具体实施细节和最新法规更新。\n"

    # 统一添加带emoji的四部分标题（如果API返回的内容没有标题）
    if not answer.startswith('## '):
        # API返回的内容，需要添加标题
        answer = "## ✨ 第一部分：精准回答\n\n" + answer
        # 检查是否有第二部分标记
        if '【更多相关参考】' in answer:
            answer = answer.replace('【更多相关参考】', '## 📖 第二部分：更多相关参考')
        elif '第二部分' not in answer:
            answer += "\n\n## 📖 第二部分：更多相关参考\n\n更多详细信息请参考下方原文段落。"

    # 添加第三部分：知识库原文段落
    if contexts:
        answer += "\n\n---\n\n## 📚 第三部分：知识库原文段落\n\n"
        for i, ctx in enumerate(contexts):
            answer += f"**段落 {i+1}** - {ctx['country']} - {ctx['source']}\n\n"
            # 显示原文（如果太长则截断）
            original_text = ctx['text']
            if len(original_text) > 300:
                original_text = original_text[:300] + "..."
            answer += f"> {original_text}\n\n"

        # 添加第四部分：原始文章链接（直接生成HTML，避免客户端转换问题）
        answer += "---\n\n## 🔗 第四部分：原始文章链接\n\n"
        seen_urls = set()
        for ctx in contexts:
            url = ctx.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                # 直接生成 HTML 链接
                answer += f'- <a href="{url}" target="_blank" style="color:#00726d;text-decoration:none;font-weight:500;">{ctx["country"]} - {ctx["source"]}</a>\n'

    return answer

@app.route('/')
def index():
    """首页 - 全新设计：绿色主题 + 出海元素"""
    html = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>全球用工智能问答 | Global HR Intelligence v2</title>\n        <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">\n        <meta http-equiv="Pragma" content="no-cache">\n        <meta http-equiv="Expires" content="0">
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            :root {
                /* 品牌标准色 */
                --deep-green: #002D28;        /* 深绿主色 */
                --ui-green: #00726d;          /* UI深绿 */
                --sand: #CEA472;              /* 墨金辅助色 */
                --light-green: #E8FFF9;       /* 浅绿背景 */

                /* 功能色 */
                --dark-bg: #002D28;
                --card-bg: #FFFFFF;
                --text-primary: #000000;
                --text-secondary: #666666;
                --border-color: #CCCCCC;
                --gray-80: #333333;
                --gray-60: #666666;
                --gray-20: #CCCCCC;

                /* 阴影 */
                --shadow-sm: 0 2px 8px rgba(0, 45, 40, 0.08);
                --shadow-lg: 0 20px 60px rgba(0, 45, 40, 0.15);
            }

            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                background: linear-gradient(165deg, #F5FAF9 0%, #FFFFFF 50%, #FAF8F5 100%);
                min-height: 100vh;
                padding: 0;
                position: relative;
                overflow-x: hidden;
            }

            /* 背景装饰元素 - 地球网格 */
            body::before {
                content: '';
                position: fixed;
                top: -50%;
                left: -50%;
                width: 200%;
                height: 200%;
                background-image:
                    radial-gradient(circle at 20% 30%, rgba(0, 114, 109, 0.05) 0%, transparent 50%),
                    radial-gradient(circle at 80% 70%, rgba(206, 164, 114, 0.03) 0%, transparent 50%);
                z-index: -1;
                animation: float 20s ease-in-out infinite;
            }

            @keyframes float {
                0%, 100% { transform: translate(0, 0) rotate(0deg); }
                50% { transform: translate(20px, 20px) rotate(5deg); }
            }

            /* 全球网格背景 */
            .globe-grid {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-image: url("data:image/svg+xml,%3Csvg width='60' height='60' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M0 30h60M30 0v60' stroke='%2300726d' stroke-width='0.5' opacity='0.1' fill='none'/%3E%3C/svg%3E");
                opacity: 0.3;
                z-index: -1;
            }

            .container {
                max-width: 1000px;
                margin: 0 auto;
                padding: 40px 20px;
            }

            /* Header */
            .header {
                text-align: center;
                margin-bottom: 50px;
                position: relative;
            }

            .title-section {
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 20px;
                margin-bottom: 12px;
            }

            .globe-icon {
                font-size: 48px;
                animation: rotate 20s linear infinite;
                cursor: pointer;
            }

            .globe-icon:hover {
                transform: scale(1.1);
            }

            @keyframes rotate {
                from { transform: rotate(0deg); }
                to { transform: rotate(360deg); }
            }

            .header h1 {
                font-size: 2.8em;
                font-weight: 700;
                background: linear-gradient(135deg, var(--deep-green), var(--ui-green));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                letter-spacing: -0.5px;
                margin: 0;
            }

            .header .subtitle {
                font-size: 1.2em;
                color: var(--text-secondary);
                font-weight: 400;
                letter-spacing: 0.5px;
            }

            .stats-bar {
                display: flex;
                justify-content: center;
                gap: 40px;
                margin-top: 25px;
                padding: 20px;
                background: rgba(255, 255, 255, 0.7);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                border: 1px solid var(--border-color);
            }

            .stat-item {
                text-align: center;
            }

            .stat-number {
                font-size: 2em;
                font-weight: 700;
                color: var(--ui-green);
            }

            /* 金色强调元素 */
            .stat-item:nth-child(2) .stat-number {
                color: var(--sand);
            }

            .stat-label {
                font-size: 0.9em;
                color: var(--text-secondary);
                margin-top: 5px;
            }

            /* Card */
            .card {
                background: var(--card-bg);
                border-radius: 24px;
                padding: 40px;
                box-shadow: var(--shadow-lg);
                border: 1px solid var(--border-color);
                backdrop-filter: blur(10px);
                position: relative;
                overflow: hidden;
            }

            .card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: linear-gradient(90deg, var(--deep-green), var(--ui-green), var(--sand));
            }

            /* Search Box */
            .search-box {
                position: relative;
                margin-bottom: 30px;
                z-index: 1;
            }

            #question {
                width: 100%;
                padding: 20px 60px 20px 24px;
                border: 2px solid var(--border-color);
                border-radius: 16px;
                font-size: 16px;
                transition: all 0.3s ease;
                background: #FAFAFA;
                font-family: inherit;
            }

            #question:focus {
                outline: none;
                border-color: var(--ui-green);
                background: white;
                box-shadow: 0 0 0 4px rgba(0, 114, 109, 0.1);
            }

            .search-button {
                position: absolute;
                right: 8px;
                top: 50%;
                transform: translateY(-50%);
                padding: 12px 24px;
                background: linear-gradient(135deg, var(--ui-green), var(--deep-green));
                color: white;
                border: none;
                border-radius: 12px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                z-index: 10;
            }

            .search-button:hover {
                transform: translateY(-50%) scale(1.05);
                box-shadow: 0 8px 20px rgba(0, 114, 109, 0.3);
            }

            .search-button:disabled {
                opacity: 0.6;
                cursor: not-allowed;
            }

            /* Examples */
            .examples {
                margin-top: 25px;
            }

            .examples h3 {
                font-size: 14px;
                color: var(--text-secondary);
                margin-bottom: 15px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 1px;
            }

            .example-grid {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 12px;
            }

            .tag {
                padding: 12px 18px;
                background: rgba(0, 114, 109, 0.06);
                border: 1px solid rgba(0, 114, 109, 0.15);
                border-radius: 12px;
                font-size: 14px;
                cursor: pointer;
                transition: all 0.3s ease;
                text-align: center;
                color: var(--text-primary);
            }

            .tag:hover {
                background: var(--ui-green);
                color: white;
                transform: translateY(-2px);
                box-shadow: var(--shadow-sm);
            }

            /* 给部分标签添加金色 */
            .tag:nth-child(3):hover {
                background: var(--sand);
            }

            .tag:nth-child(6):hover {
                background: var(--sand);
            }

            /* Result */
            .result {
                margin-top: 30px;
                animation: fadeIn 0.5s ease;
            }

            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }

            .answer {
                background: linear-gradient(135deg, #FAFFFE 0%, #FFFFFF 100%);
                padding: 30px;
                border-radius: 16px;
                border-left: 4px solid var(--ui-green);
                line-height: 1.8;
                color: var(--text-primary);
                box-shadow: var(--shadow-sm);
                position: relative;
                font-size: 15px;
            }

            .answer::after {
                content: '';
                position: absolute;
                top: 0;
                right: 0;
                width: 80px;
                height: 80px;
                background: radial-gradient(circle at top right, var(--sand), transparent);
                opacity: 0.1;
                border-radius: 0 16px 0 0;
            }

            .answer a {
                color: var(--ui-green);
                text-decoration: none;
                font-weight: 500;
                transition: color 0.3s;
            }

            .answer a:hover {
                color: var(--deep-green);
                text-decoration: underline;
            }

            .answer h3 {
                color: #00726d;
                font-size: 1.3em;
                margin: 24px 0 12px 0;
                padding-bottom: 8px;
                border-bottom: 2px solid #E8FFF9;
            }

            .answer h4 {
                color: #002D28;
                font-size: 1.1em;
                margin: 16px 0 8px 0;
            }

            .answer strong {
                color: #00726d;
                font-weight: 600;
            }

            .answer blockquote {
                margin: 12px 0;
                padding: 12px 16px;
                background: #F5FAF9;
                border-left: 4px solid #CEA472;
                color: #666;
                font-size: 0.95em;
                border-radius: 0 8px 8px 0;
            }

            .answer hr {
                border: none;
                border-top: 1px solid #E8FFF9;
                margin: 20px 0;
            }

            /* Loading */
            .loading {
                text-align: center;
                padding: 40px;
                color: var(--text-secondary);
            }

            .spinner {
                display: inline-block;
                width: 40px;
                height: 40px;
                border: 4px solid rgba(0, 114, 109, 0.1);
                border-top: 4px solid var(--ui-green);
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            .loading-text {
                margin-top: 20px;
                font-size: 16px;
            }

            /* Footer */
            .footer {
                text-align: center;
                margin-top: 50px;
                padding: 30px;
                color: var(--text-secondary);
                font-size: 14px;
            }

            .footer-links {
                display: flex;
                justify-content: center;
                gap: 30px;
                margin-top: 15px;
            }

            .footer-links a {
                color: var(--ui-green);
                text-decoration: none;
                font-weight: 500;
                transition: color 0.3s;
            }

            .footer-links a:hover {
                color: var(--sand);
            }

            /* Responsive */
            @media (max-width: 768px) {
                .title-section {
                    flex-direction: column;
                    gap: 10px;
                }

                .globe-icon {
                    font-size: 40px;
                }

                .header h1 {
                    font-size: 2em;
                }

                .stats-bar {
                    flex-direction: column;
                    gap: 20px;
                }

                .example-grid {
                    grid-template-columns: 1fr;
                }

                .card {
                    padding: 25px;
                }

                .search-button {
                    position: static;
                    transform: none;
                    width: 100%;
                    margin-top: 10px;
                }

                #question {
                    padding: 20px;
                }
            }
        </style>
    </head>
    <body>
        <div class="globe-grid"></div>

        <div class="container">
            <div class="header">
                <div class="title-section">
                    <div class="globe-icon" onclick="window.location.reload()">🌍</div>
                    <h1>全球用工智能问答</h1>
                </div>
                <p class="subtitle">Global Employment Intelligence Platform</p>

                <div class="stats-bar">
                    <div class="stat-item">
                        <div class="stat-number">43</div>
                        <div class="stat-label">覆盖国家</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number">AI</div>
                        <div class="stat-label">智能检索</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number">24/7</div>
                        <div class="stat-label">随时查询</div>
                    </div>
                </div>
            </div>

            <div class="card">
                <div class="search-box">
                    <input
                        type="text"
                        id="question"
                        placeholder="输入您的问题，例如：巴西的年假是多少天？"
                    />
                    <button class="search-button" onclick="askQuestion()">
                        🔍 查询
                    </button>
                </div>

                <div class="examples">
                    <h3>💡 热门问题</h3>
                    <div class="example-grid">
                        <span class="tag" onclick="fillQuestion('巴西的年假是多少天？')">🇧🇷 巴西年假</span>
                        <span class="tag" onclick="fillQuestion('德国的试用期有多长？')">🇩🇪 德国试用期</span>
                        <span class="tag" onclick="fillQuestion('美国的法定假日有多少天？')">🇺🇸 美国假日</span>
                        <span class="tag" onclick="fillQuestion('新加坡的病假规定是什么？')">🇸🇬 新加坡病假</span>
                        <span class="tag" onclick="fillQuestion('法国的年假是多少天？')">🇫🇷 法国的年假</span>
                        <span class="tag" onclick="fillQuestion('澳大利亚的合同期限规定？')">🇦🇺 澳洲合同</span>
                        <span class="tag" onclick="fillQuestion('印度的最低工资标准？')">🇮🇳 印度工资</span>
                        <span class="tag" onclick="fillQuestion('日本的加班政策')">🇯🇵 日本加班</span>
                    </div>
                </div>

                <div id="result" class="result" style="display: none;">
                    <div class="answer" id="answer"></div>
                </div>

                <div id="loading" class="loading" style="display: none;">
                    <div class="spinner"></div>
                    <p class="loading-text">AI 正在分析全球用工数据...</p>
                </div>
            </div>

            <div class="footer">
                <p>基于 43 个国家的官方用工政策数据 · 由 Claude AI 提供智能分析</p>
                <div class="footer-links">
                    <a href="https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzkwMDMzMTY2Mg==&action=getalbum&album_id=2668531001535954946" target="_blank">📚 数据来源：Horizons</a>
                </div>
            </div>
        </div>

        <script>
            function fillQuestion(text) {
                console.log('=== fillQuestion 被调用 ===');
                const questionInput = document.getElementById('question');
                if (questionInput) {
                    questionInput.value = text;
                    questionInput.focus();
                    // 自动触发查询
                    askQuestion();
                } else {
                    console.error('找不到 question 输入框');
                }
            }

            // formatAnswer v2 - 支持链接转换
            function formatAnswer(answer) {
                let lines = answer.split(String.fromCharCode(10));
                let result = [];
                let inList = false;
                let listItems = [];
                
                for (let i = 0; i < lines.length; i++) {
                    let line = lines[i];
                    let trimmed = line.trim();
                    
                    // 跳过空行处理，但保留一个
                    if (!trimmed) {
                        if (inList) {
                            result.push('</div>');
                            inList = false;
                            listItems = [];
                        }
                        continue;
                    }
                    
                    // 分隔线 ---
                    if (trimmed === '---') {
                        if (inList) {
                            result.push('</div>');
                            inList = false;
                        }
                        result.push('<hr style="border:none;border-top:1px solid #E8FFF9;margin:20px 0;">');
                        continue;
                    }
                    
                    // 标题 ##
                    if (trimmed.startsWith('## ')) {
                        if (inList) {
                            result.push('</div>');
                            inList = false;
                        }
                        result.push('<h3 style="color:#00726d;font-size:1.3em;margin:20px 0 12px 0;padding-bottom:8px;border-bottom:2px solid #E8FFF9;">' + trimmed.substring(3) + '</h3>');
                        continue;
                    }
                    if (trimmed.startsWith('### ')) {
                        if (inList) {
                            result.push('</div>');
                            inList = false;
                        }
                        result.push('<h4 style="color:#002D28;font-size:1.1em;margin:16px 0 8px 0;">' + trimmed.substring(4) + '</h4>');
                        continue;
                    }
                    
                    // 引用块 >
                    if (trimmed.startsWith('> ')) {
                        if (inList) {
                            result.push('</div>');
                            inList = false;
                        }
                        result.push('<blockquote style="margin:12px 0;padding:12px 16px;background:#F5FAF9;border-left:4px solid #CEA472;color:#666;font-size:0.95em;">' + trimmed.substring(2) + '</blockquote>');
                        continue;
                    }
                    
                    // 先检查是否是列表项（在转换行内格式前，避免链接被分割）
                    // 数字列表 1. 2. 3.
                    let numMatch = trimmed.match(/^(\d+)\.\s+(.+)$/);
                    // 无序列表 - * •
                    let bulletMatch = trimmed.match(/^[-\*•]\s+(.+)$/);
                    
                    // 处理行内格式
                    // Markdown 链接 [text](url) - 匹配到右括号结束
                    line = line.replace(/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g, '<a href="$2" target="_blank" style="color:#00726d;text-decoration:none;font-weight:500;">$1</a>');
                    
                    // 加粗 **文本**
                    line = line.replace(/\*\*([^*]+)\*\*/g, '<strong style="color:#00726d;">$1</strong>');
                    
                    // 移除代码块符号和中文方括号
                    line = line.replace(/`/g, '').replace(/【/g, '').replace(/】/g, '');
                    
                    // 处理数字列表
                    if (numMatch) {
                        if (!inList) {
                            result.push('<div style="margin:12px 0;">');
                            inList = true;
                        }
                        // 提取列表项内容（去掉序号）
                        let content = line.replace(/^\s*\d+\.\s*/, '');
                        result.push('<div style="margin:6px 0;padding-left:20px;"><span style="color:#00726d;font-weight:600;">' + numMatch[1] + '.</span> ' + content + '</div>');
                        continue;
                    }
                    
                    // 处理无序列表
                    if (bulletMatch) {
                        if (!inList) {
                            result.push('<div style="margin:12px 0;">');
                            inList = true;
                        }
                        // 提取列表项内容（去掉 bullet）
                        let content = line.replace(/^\s*[-\*•]\s*/, '');
                        result.push('<div style="margin:6px 0;padding-left:20px;"><span style="color:#CEA472;margin-right:8px;">•</span>' + content + '</div>');
                        continue;
                    }
                    
                    // 普通段落
                    if (inList) {
                        result.push('</div>');
                        inList = false;
                    }
                    result.push('<p style="margin:10px 0;line-height:1.8;">' + line + '</p>');
                }
                
                if (inList) {
                    result.push('</div>');
                }
                
                return result.join('');
            }

            async function askQuestion() {
                console.log('=== askQuestion 被调用 ===');
                const questionInput = document.getElementById('question');
                const resultDiv = document.getElementById('result');
                const loadingDiv = document.getElementById('loading');
                const answerDiv = document.getElementById('answer');
                
                if (!questionInput || !resultDiv || !loadingDiv || !answerDiv) {
                    console.error('找不到必要的DOM元素');
                    alert('页面加载错误，请刷新页面重试');
                    return;
                }
                
                const question = questionInput.value.trim();
                console.log('问题:', question);

                if (!question) {
                    alert('请输入问题');
                    questionInput.focus();
                    return;
                }

                console.log('开始查询...');

                // 显示加载
                resultDiv.style.display = 'none';
                loadingDiv.style.display = 'block';

                try {
                    console.log('发送 fetch 请求到 /api/ask');
                    const response = await fetch('/api/ask', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ question })
                    });

                    console.log('响应状态:', response.status);

                    if (!response.ok) {
                        throw new Error('HTTP错误: ' + response.status);
                    }

                    const data = await response.json();
                    console.log('响应数据:', data);

                    if (data.error) {
                        answerDiv.innerHTML = '<div style="color: #dc3545; padding: 10px; background: #f8d7da; border-radius: 8px;">错误: ' + data.error + '</div>';
                    } else if (data.not_found) {
                        // 知识库无答案，显示 Deepseek 求助按钮
                        console.log('知识库未找到答案，状态:', data.status, '国家:', data.country);
                        answerDiv.innerHTML = renderNotFoundPrompt(question, data.status, data.country);
                    } else if (data.answer) {
                        console.log('原始答案长度:', data.answer.length);
                        // 格式化答案：美化显示
                        const formatted = formatAnswer(data.answer);
                        console.log('格式化后长度:', formatted.length);
                        // 添加 Deepseek 拓展搜索按钮
                        const deepseekButton = renderDeepseekButton(question, '拓展搜索');
                        answerDiv.innerHTML = formatted + deepseekButton;
                    } else {
                        answerDiv.innerHTML = '<div style="color: #856404; padding: 10px; background: #fff3cd; border-radius: 8px;">未获取到答案</div>';
                    }

                    loadingDiv.style.display = 'none';
                    resultDiv.style.display = 'block';
                    console.log('查询完成');

                } catch (error) {
                    console.error('查询错误:', error);
                    answerDiv.innerHTML = '<div style="color: #dc3545; padding: 10px; background: #f8d7da; border-radius: 8px;">发生错误: ' + error.message + '</div>';
                    loadingDiv.style.display = 'none';
                    resultDiv.style.display = 'block';
                }
            }

            // 渲染知识库无答案时的提示界面
            function renderNotFoundPrompt(question, status, country) {
                let title = '未找到相关答案';
                let description = '';
                let icon = '🔍';
                
                if (status === 'no_country') {
                    icon = '🌍';
                    title = `${country} 暂未收录`;
                    description = `目前知识库暂未收录 <strong style="color: #00726d;">${country}</strong> 的用工指南数据。`;
                } else if (status === 'no_content') {
                    icon = '📚';
                    title = `${country} 数据缺失`;
                    description = `知识库中暂无 <strong style="color: #00726d;">${country}</strong> 的详细用工指南数据。`;
                } else if (status === 'irrelevant') {
                    icon = '❓';
                    title = `${country} 暂无相关内容`;
                    description = `知识库中收录了 <strong style="color: #00726d;">${country}</strong> 的用工指南，但未找到与您问题相关的内容。`;
                } else {
                    description = '在 <strong style="color: #00726d;">43 个国家</strong> 的用工指南知识库中，没有找到与您问题相关的内容。';
                }
                
                return `
                <div style="text-align: center; padding: 40px 20px;">
                    <div style="font-size: 48px; margin-bottom: 20px;">${icon}</div>
                    <h3 style="color: #002D28; font-size: 1.3em; margin-bottom: 16px;">${title}</h3>
                    <p style="color: #666; font-size: 15px; line-height: 1.6; margin-bottom: 24px; max-width: 500px; margin-left: auto; margin-right: auto;">
                        ${description}
                    </p>
                    <div style="background: linear-gradient(135deg, #E8FFF9 0%, #F5FAF9 100%); border-radius: 16px; padding: 24px; margin: 20px 0; border: 1px solid rgba(0, 114, 109, 0.15);">
                        <div style="font-size: 32px; margin-bottom: 12px;">🤖</div>
                        <p style="color: #333; font-size: 14px; margin-bottom: 16px;">
                            您可以通过 Deepseek AI 进行<strong style="color: #00726d;">联网搜索</strong>获取答案
                        </p>
                        <button onclick='callDeepseekSearch("${question.replace(/"/g, '\\"')}")' 
                                style="background: linear-gradient(135deg, #00726d, #002D28); 
                                       color: white; border: none; padding: 14px 32px; 
                                       border-radius: 12px; font-size: 15px; font-weight: 600; 
                                       cursor: pointer; transition: all 0.3s ease;
                                       box-shadow: 0 4px 15px rgba(0, 114, 109, 0.3);">
                            ✨ 求助 Deepseek 联网搜索
                        </button>
                    </div>
                    <p style="color: #999; font-size: 12px; margin-top: 16px;">
                        💡 提示：您也可以尝试换个问法，或询问其他国家/地区的用工政策
                    </p>
                </div>
                `;
            }

            // 渲染 Deepseek 拓展搜索按钮（用于正确答案底部）
            function renderDeepseekButton(question, buttonText = '拓展搜索更多内容') {
                return `
                <div style="margin-top: 30px; padding: 20px; background: linear-gradient(135deg, #E8FFF9 0%, #F5FAF9 100%); border-radius: 16px; border: 1px solid rgba(0, 114, 109, 0.15); text-align: center;">
                    <div style="display: flex; align-items: center; justify-content: center; gap: 10px; margin-bottom: 12px;">
                        <span style="font-size: 24px;">🤖</span>
                        <span style="color: #002D28; font-weight: 600; font-size: 14px;">想要了解更多？</span>
                    </div>
                    <p style="color: #666; font-size: 13px; margin-bottom: 15px;">
                        通过 Deepseek AI 联网搜索获取更多相关信息
                    </p>
                    <button onclick='callDeepseekSearch("${question.replace(/"/g, '\\"')}")' 
                            style="background: linear-gradient(135deg, #00726d, #002D28); 
                                   color: white; border: none; padding: 12px 28px; 
                                   border-radius: 10px; font-size: 14px; font-weight: 600; 
                                   cursor: pointer; transition: all 0.3s ease;
                                   box-shadow: 0 4px 15px rgba(0, 114, 109, 0.3);">
                        🔍 ${buttonText}
                    </button>
                </div>
                `;
            }

            // 调用 Deepseek 联网搜索
            async function callDeepseekSearch(question) {
                console.log('=== 调用 Deepseek 搜索 ===');
                const resultDiv = document.getElementById('result');
                const loadingDiv = document.getElementById('loading');
                const answerDiv = document.getElementById('answer');
                
                // 显示加载状态
                resultDiv.style.display = 'none';
                loadingDiv.style.display = 'block';
                
                // 更新加载文本
                const loadingText = loadingDiv.querySelector('.loading-text');
                if (loadingText) {
                    loadingText.textContent = 'Deepseek AI 正在联网搜索全球用工数据...';
                }

                try {
                    const response = await fetch('/api/deepseek', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ question })
                    });

                    if (!response.ok) {
                        throw new Error('HTTP错误: ' + response.status);
                    }

                    const data = await response.json();
                    console.log('Deepseek 响应:', data);

                    if (data.error) {
                        answerDiv.innerHTML = '<div style="color: #dc3545; padding: 10px; background: #f8d7da; border-radius: 8px;">错误: ' + data.error + '</div>';
                    } else if (data.answer) {
                        // 格式化 Deepseek 答案，添加来源标记
                        const formattedAnswer = formatDeepseekAnswer(data.answer);
                        answerDiv.innerHTML = formattedAnswer;
                    } else {
                        answerDiv.innerHTML = '<div style="color: #856404; padding: 10px; background: #fff3cd; border-radius: 8px;">Deepseek 未能生成答案</div>';
                    }

                } catch (error) {
                    console.error('Deepseek 调用错误:', error);
                    answerDiv.innerHTML = '<div style="color: #dc3545; padding: 10px; background: #f8d7da; border-radius: 8px;">调用 Deepseek 时出错: ' + error.message + '</div>';
                } finally {
                    loadingDiv.style.display = 'none';
                    resultDiv.style.display = 'block';
                    // 恢复加载文本
                    const loadingText = loadingDiv.querySelector('.loading-text');
                    if (loadingText) {
                        loadingText.textContent = 'AI 正在分析全球用工数据...';
                    }
                }
            }

            // 格式化 Deepseek 答案
            function formatDeepseekAnswer(answer) {
                // 添加 Deepseek 来源标记
                let formatted = `
                <div style="background: linear-gradient(135deg, #E8FFF9 0%, #F5FAF9 100%); padding: 16px 20px; border-radius: 12px; margin-bottom: 20px; border-left: 4px solid #00726d;">
                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                        <span style="font-size: 20px;">🤖</span>
                        <span style="color: #002D28; font-weight: 600; font-size: 14px;">Deepseek AI 联网搜索</span>
                        <span style="background: #00726d; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px;">联网</span>
                    </div>
                    <p style="color: #666; font-size: 12px; margin: 0;">以下内容通过 Deepseek AI 联网搜索生成，仅供参考</p>
                </div>
                `;
                
                // 使用相同的 formatAnswer 函数处理内容
                formatted += formatAnswer(answer);
                
                // 添加免责声明
                formatted += `
                <div style="margin-top: 24px; padding: 16px; background: #FFF8F0; border-radius: 12px; border-left: 4px solid #CEA472;">
                    <p style="color: #886644; font-size: 12px; margin: 0; line-height: 1.6;">
                        <strong style="color: #CEA472;">⚠️ 免责声明：</strong>
                        以上内容由 AI 联网搜索生成，可能存在信息滞后或不准确的情况。
                        重要决策前请核实官方最新政策或咨询专业法律顾问。
                    </p>
                </div>
                `;
                
                return formatted;
            }

            // 确保DOM加载完成后再绑定事件
            document.addEventListener('DOMContentLoaded', function() {
                document.getElementById('question').addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') {
                        askQuestion();
                    }
                });
            });
        </script>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route('/api/ask', methods=['POST'])
def ask():
    """API: 回答问题"""
    try:
        data = request.json
        question = data.get('question', '')

        if not question:
            return jsonify({'error': '问题不能为空'}), 400

        # 查询知识库，同时获取状态信息
        result = query_knowledge_base_with_status(question, top_k=3)
        contexts = result.get('contexts', [])
        status = result.get('status', 'not_found')
        country = result.get('country', '')

        if not contexts:
            return jsonify({
                'not_found': True,
                'status': status,  # 'no_country' | 'no_content' | 'irrelevant'
                'country': country,
                'answer': '',
                'sources': []
            })

        # 生成答案
        answer = generate_answer(question, contexts)

        return jsonify({
            'answer': answer,
            'sources': contexts
        })

    except Exception as e:
        print(f"错误: {str(e)}")
        return jsonify({'error': str(e)}), 500


def call_deepseek_search(question):
    """调用 Deepseek 进行联网搜索并生成答案"""
    deepseek_key = os.environ.get('DEEPSEEK_API_KEY')
    if not deepseek_key:
        return "抱歉，Deepseek 服务暂时不可用。"
    
    try:
        from openai import OpenAI
        deepseek_client = OpenAI(
            api_key=deepseek_key,
            base_url="https://api.deepseek.com"
        )
        
        prompt = f"""你是一个专业的国际HR顾问助手。请回答用户关于全球用工政策的问题。

用户问题：{question}

请通过联网搜索获取最新信息，并以清晰、专业的格式回答。如果无法确定具体数据，请说明信息来源的不确定性。

回答格式要求：
1. 直接给出答案要点
2. 使用简洁的列表或段落
3. 如有相关法规或政策依据，请简要说明
4. 注明信息来源（如官网、法规等）

回答："""
        
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": prompt
            }],
            extra_body={"search": True}  # 启用联网搜索
        )
        
        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content
        else:
            return "抱歉，Deepseek 未能生成有效答案。"
            
    except Exception as e:
        print(f"Deepseek 调用错误: {str(e)}")
        return f"抱歉，调用 Deepseek 时出错：{str(e)}"


@app.route('/api/deepseek', methods=['POST'])
def deepseek_search():
    """API: 使用 Deepseek 联网搜索回答问题"""
    try:
        data = request.json
        question = data.get('question', '')

        if not question:
            return jsonify({'error': '问题不能为空'}), 400

        # 调用 Deepseek 进行联网搜索
        answer = call_deepseek_search(question)

        return jsonify({
            'answer': answer,
            'source': 'deepseek_search'
        })

    except Exception as e:
        print(f"Deepseek API 错误: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("="*60)
    print("启动全球用工智能问答服务（全新设计）")
    print("="*60)

    try:
        init_services()
        # 从环境变量获取端口（Render 会使用 PORT 环境变量）
        port = int(os.environ.get('PORT', 5002))
        print(f"\n✓ 服务已启动: http://0.0.0.0:{port}")
        print("  按 Ctrl+C 停止服务\n")
        app.run(host='0.0.0.0', port=port, debug=False)
    except Exception as e:
        print(f"\n✗ 启动失败: {str(e)}")
        print("  请确保已运行: python build_knowledge_base.py")
