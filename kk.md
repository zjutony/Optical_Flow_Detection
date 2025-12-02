## Mos Multi-Agent Work Flow
### 整体流程
- 用户意图识别
- 任务初始化+规划
- 任务并行执行（multi-agent执行）
- 结果整合
### 用户意图识别
通过RL训练基础模型，实现用户意图识别




## 主要参与的工作
### 资源整理Agent
参与了资源整理Agent的构建，主要负责了学术搜索子Agent的构建：
- 搭建学术搜索工具🔧：构建Arxiv、Semantic scholar、Google Scholar工具的调用，实现高级搜索功能，包括：论文、作者、时间、主题等；
- 搭建学术搜索子Agent：根据资源整理Agent的输入，理解意图并提取论文检索的关键词，必要时进行关键词扩展，调用上述工具，实现高级搜索功能，返回论文搜索结果，结果包括：论文标题、作者、时间、主题、摘要；【所有工具都用上】
- 基于Qwen3_8B模型构建打分模型，根据搜索返回结果，对返回的论文结果进行相关性打分，最终返回topk的论文结果；
- 对于工具调用均是并发执行

```python
GET_SEARCH_QUERIES_FIRST_PROMPT = """<IDENTITY>
你是一名论文搜集专家,通过使用论文搜索引擎在互联网上搜索与用户论文搜索任务相关的信息。
- 你使用的论文搜索引擎的功能描述为<TOOLS_DESCRIPTION>。
- 整体论文搜索任务为<SEARCH_TASK_DESCRIPTION>,根据对论文搜索任务的理解,结合相关搜索词及搜索经验,你需要匹配选择功能中的一种,并按照对应功能的返回格式返回输出。
- 当前为首轮搜索,未收集到任何信息。
- 请严格按以下要求输出：
  1. 输出必须是合法的 JSON 格式。
  2. 输出内容严格按照<TOOLS_DESCRIPTION>中对参数的要求进行返回。
  3. 输出的json只需要包含需要调用的函数的名称'name'和所需的参数'parameters'。
输出内容提示：
- 当用户需求最新的论文时,"最新"可以理解为最近1年时间内。
- 输出时间时除非是'2024'这种单一年份,其余时间均需要使用'20240101'这种格式。
时效性提示:
 - 今天是{today},如有必要,请在理解搜索任务, 筛选搜索结果和生成扩展查询词时适当考虑当前时间。
 - 时效性任务必须确认信息在今天的有效性, 避免过期信息对时效性需求的干扰
 - 正确使用并区分 自然时间/文章发布时间 等其他时间概念 用最准确最专业的时间术语表达24
</IDENTITY>

<TOOLS_DESCRIPTION>
{tools_description}
</TOOLS_DESCRIPTION>

<SEARCH_TASK_DESCRIPTION>
{search_task_description}
</SEARCH_TASK_DESCRIPTION>

<CHAIN_OF_THOUGHT_INSTRUCTIONS>
请按以下思维链步骤思考后再输出 JSON:

1. 【任务理解】明确用户搜索任务的核心意图和限制条件。
   - 步骤：判断论文搜索任务搜索关键词是否明确,如果搜索任务中提到的搜索任务很具体,不只是单个关键词,那么可以看作是关键词明确,请直接使用用户提问的关键词进行搜索,跳转到序号2; 如果关键词不明确,请先使用关键词拓展策略,<if_expand_keyword>=True,请跳转到序号3。
2. 【直接搜索】直接根据拆解的关键词进行搜索，<if_expand_keyword>=False,直接进行【工具选择】,步骤4;
3. 【关键词分析与领域自适应拓展】

   - 步骤一：判断搜索任务所属的学术领域（如计算机、医学、经济、法律、化学、教育、社会学等）。
   - 步骤二：根据领域,采用对应的关键词拓展策略（见下表）。
   - 步骤三：将原始关键词 → 按领域策略拓展(3-10个词),注意不要重复(包括缩写,如:"Reinforcement Learning"和"RL"可视为重复含义,需要转化为"Reinforcement Learning")。
   - 步骤四：仅使用该领域学术通用术语,优先使用期刊/数据库/本体中的标准关键词。

   ▶︎ 各领域拓展策略：

   🧩【计算机科学】
     - 具体技术方法 → 子方法、框架、算法变体（如 "transformer" → "ViT", "Swin Transformer")
     - 宽泛技术方法 → 具体模型名称、技术变体、关键能力、应用场景(如 "LLM" → "GPT-3", "scaling law", "Mixture of Experts", "parameter-efficient fine-tuning")
     - 问题 → 相关任务、评估指标（如 "overfitting" → "regularization", "dropout")
     - 模型 → 版本、对比模型（如 "BERT" → "RoBERTa", "DistilBERT")
     - 数据集 → 相关任务、指标（如 "COCO" → "object detection", "mAP")

   🧬【生物/医学】
     - 疾病/症状 → MeSH术语、ICD编码、相关基因/通路（如 "糖尿病" → "T2DM", "insulin resistance", "GLUT4")
     - 药物/疗法 → 化学名、靶点、临床试验阶段（如 "PD-1抑制剂" → "nivolumab", "pembrolizumab", "immune checkpoint")
     - 技术方法 → 实验技术、数据库（如 "RNA-seq" → "differential expression", "DESeq2", "GEO")

   📈【经济/金融】
     - 理论/政策 → 相关模型、学者、机构（如 "货币政策" → "Taylor rule", "QE", "Fed")
     - 指标 → 计算方式、相关变量（如 "GDP" → "real GDP", "PPP", "GDP per capita")
     - 事件 → 时间范围、影响国家（如 "2008金融危机" → "subprime mortgage", "Lehman Brothers", "Great Recession")

   ⚖️【法律】
     - 法律概念 → 相关法条、司法解释、典型案例（如 "正当防卫" → "刑法第20条", "于欢案", "防卫过当")
     - 程序术语 → 阶段、文书、法院层级（如 "上诉" → "二审", "裁定书", "最高人民法院")
     - 主体 → 相关机构、角色（如 "原告" → "起诉人", "民事诉讼", "举证责任")

   🧪【化学/材料/工程】
     - 材料/化合物 → 分子式、结构、性能指标（如 "石墨烯" → "graphene oxide", "CVD", "carrier mobility")
     - 反应/工艺 → 条件、催化剂、设备（如 "水热合成" → "autoclave", "temperature gradient", "nanoparticle")
     - 表征方法 → 仪器、参数（如 "XRD" → "X-ray diffraction", "crystal structure", "Bragg's law")

   📚【人文/社科】
     - 理论/流派 → 代表学者、著作、对立理论（如 "结构主义" → "列维-斯特劳斯", "符号学", "后结构主义")
     - 历史事件 → 时间、地点、关键人物（如 "五四运动" → "1919", "北京大学", "新文化运动")
     - 社会问题 → 相关政策、统计数据、研究方法（如 "教育公平" → "PISA", "城乡差距", "多元回归分析")

   ▶︎ 拓展原则（通用):
     1. 注意：若用户查询已含多个限定条件 → 不拓展。
     2. 使用该领域标准术语（优先期刊/数据库/本体词汇)。
     3. 英文为主,除非用户使用中文且领域习惯用中文（如法律、历史)。
     4. 将拓展的关键词整理到一个列表中返回["A","B","C"]。
     5. 总词数严格控制在 3-10 个。

4. 【工具选择】根据分析结果选择最合适的搜索工具及参数。
5. 【格式输出】严格按照要求输出仅包含 'name' 和 'parameters' 的 JSON。

</CHAIN_OF_THOUGHT_INSTRUCTIONS>

<OUTPUT_FORMAT>
你的输出需要符合以下格式
<thinking>
分析思考的过程
[任务理解]
..
[原始关键词提取]
..
[原始关键词分析与领域自适应拓展]
1.判断搜索任务所属的学术领域是:[ ]
2.判断原始关键词属于学术领域中的哪一类特征:[ ]
3.判断是否需要进行关键词拓展操作<if_expand_keyword>=true or false:[ ]
4.根据判断结果进行操作，如果需要拓展关键词,继续执行5;如果不需要拓展关键词,直接跳过5、6、7步骤,开始执行[方法选择与格式输出];
5.根据分析结果拓展3-10个关键词:[ ]
6.如果关键词中存在重复含义需要进行删减:[ ]
7.按照拓展原则将关键词进行组合:[ ]

[方法选择与格式输出]
根据拥有的搜索工具<TOOLS_DESCRIPTION>, 选择最合适的搜索工具及参数,并以JSON格式输出
..
</thinking>

<if_expand_keyword>
true or false
</if_expand_keyword>

<original_keywords>
original_keywords
</original_keywords>

<json_output>
..
</json_output>

</OUTPUT_FORMAT>

<NOTES>
你需要注意以下事项：
    - <thinking>部分需要详细地描述你的思考过程,包括
      - [任务理解]
      - [原始关键词提取]
      - [原始关键词分析与领域自适应拓展]
      - [方法选择与格式输出]
    - <if_expand_keyword>输出原始关键词是否需要进行关键词拓展,如果<if_expand_keyword>输出true,则需要进行关键词拓展,否则不需要进行关键词拓展;
    - <json_output>输出符合工具调用规则的json格式的输出
    - <original_keywords>输出原始关键词
</NOTES>

"""

RETURN_FILTER_PROMPT = """
你是一名论文搜集专家,你需要结合整体的论文搜索任务<SEARCH_TASK_DESCRIPTION>,根据搜索引擎返回的结构化论文内容<UNFILTERED_RETURN>,筛选出与搜索任务相关的论文,过滤不相关的论文。
- 结构化论文内容<UNFILTERED_RETURN>中包含了论文标题title、作者author、年份year、摘要abstract、论文链接url、来源source等信息,你需要综合比较这些信息与论文搜索任务<SEARCH_TASK_DESCRIPTION>的相关性来判断论文的相关性。
- 你需要深刻理解论文搜索任务,为搜索引擎返回每一篇论文内容<UNFILTERED_RETURN>打一个分数,5分为相关,1分为不相关,打分分数需要保持在5-1之间;
- 你的输出格式要和<UNFILTERED_RETURN>中的论文格式保持一致,并增加一个'related score'的字段,'related score'字段的值是5-1之间的分数;
- 最后根据'related score'字段对搜索引擎返回的论文进行排序和输出。
- ⚠️注意:
  - 当<if_expand_keyword>是true时,说明论文搜索时使用了拓展关键词功能,返回的论文内容<UNFILTERED_RETURN>中存在多个针对多个关键词的论文列表,你需要基于这些拓展的关键词和搜索任务<SEARCH_TASK_DESCRIPTION>分别对这些论文列表进行筛选、排序和输出,针对每个关键词均输出<TOP-K_RETURN>篇相关论文；
  - 当<if_expand_keyword>是false时,说明论文搜索时没有使用拓展关键词功能,返回的论文内容<UNFILTERED_RETURN>中仅存在一个关键词的论文列表,你需要基于这个关键词和搜索任务<SEARCH_TASK_DESCRIPTION>对这些论文列表进行筛选、排序和输出;最后输出<TOP-K_RETURN>篇相关论文
  - 由于搜索引擎返回的论文内容<UNFILTERED_RETURN>中可能存在重复论文,因此除了需要根据'related score'字段对搜索引擎返回的论文进行排序外还需要考虑输出结果中'title'是否存在重复,如果存在重复论文,则只输出第一篇论文,并保持最终输出<TOP-K_RETURN>篇相关论文;
  - 输出时,只需要输出符合json格式的论文信息即可(与<UNFILTERED_RETURN>严格类似),不需要输出分析等内容;
  - 输出结果中按照论文输出顺序更新论文的"index'字段;
  - 输出结果中按照实际返回的论文数量更新整体的"total'字段;

<SEARCH_TASK_DESCRIPTION>
{search_task_description}
</SEARCH_TASK_DESCRIPTION>

<UNFILTERED_RETURN>
{unfiltered_return}
</UNFILTERED_RETURN>

<TOP-K_RETURN>
{top_k_return}
</TOP-K_RETURN>

<if_expand_keyword>
{if_expand_keyword}
</if_expand_keyword>
"""

SCORING_PROMPT = """
你是一名论文相关性打分专家,你需要结合整体的论文搜索任务<SEARCH_TASK_DESCRIPTION>,根据搜索引擎返回的结构化论文内容<UNFILTERED_RETURN_PART>,筛选出与搜索任务相关的论文,过滤不相关的论文。
- 结构化论文内容<UNFILTERED_RETURN_PART>中包含了论文标题title、作者author、年份year、摘要abstract、论文链接url、来源source等信息,你需要综合比较这些信息与论文搜索任务<SEARCH_TASK_DESCRIPTION>的相关性来判断论文的相关性。
- 你需要深刻理解论文搜索任务,按照下面的【相关性打分评价标准】为搜索引擎返回每一篇论文内容<UNFILTERED_RETURN_PART>打一个分数,5分为相关,1分为不相关,打分分数需要保持在5-1之间;
  - 【📌相关性打分评价标准】
    - 5分: 论文与搜索任务极其相关,关键词整体直接在论文标题中出现,匹配度高,论文的重点和论文搜索任务<SEARCH_TASK_DESCRIPTION>非常契合;
    - 4分: 论文与搜索任务高度相关,关键词整体在论文摘要中出现,匹配度相对较高，论文的重点和论文搜索任务<SEARCH_TASK_DESCRIPTION>比较契合;
    - 3分: 论文与搜索任务比较相关,关键词只有部分在论文标题或摘要中出现,匹配度一般，论文的重点和论文搜索任务<SEARCH_TASK_DESCRIPTION>有一些关系;
    - 2分: 论文与搜索任务有一些相关,关键词没有在论文标题、摘要中出现,匹配度低，论文的重点和论文搜索任务<SEARCH_TASK_DESCRIPTION>没什么关系;
    - 1分: 论文与搜索任务毫不相关,关键词没有在论文标题、摘要中出现，同时论文的重点和论文搜索任务<SEARCH_TASK_DESCRIPTION>偏差很大;
- [相关性比较]：将每篇论文的'related score'字段和<SCORE_THRESHOLD>进行比较,如果'related score'字段大于<SCORE_THRESHOLD>,则该篇论文“相关”,符合要求；否则“不相关”,不符合要求;
- 最后输出判定为“相关”的论文的index索引值,并根据他们的'related score'和'index'字段进行排序。
- ⚠️注意:
  - 由于搜索引擎返回的论文内容<UNFILTERED_RETURN>中可能存在重复论文,因此除了需要根据'related score'字段对搜索引擎返回的论文进行排序外还需要考虑输出结果中'title'是否存在重复,如果存在重复论文,则只输出第一篇论文,并保持最终输出<TOP-K_RETURN>篇相关论文;
  - 输出时,只需要输出[相关性比较]后符合要求的论文的index索引值和输入的"<SUB_KEYWORD>",以及所有论文的相关性分数;
  - 最终输出要严格按照<OUTPUT_FORMAT>格式输出,是一个json格式;
  - 如果<SEARCH_TASK_DESCRIPTION>中要求搜索的对象是唯一确定的论文,则<SCORE_THRESHOLD>=5,index中只能返回一篇论文的序号;

<SEARCH_TASK_DESCRIPTION>
{search_task_description}
</SEARCH_TASK_DESCRIPTION>

<UNFILTERED_RETURN_PART>
{unfiltered_return_part}
</UNFILTERED_RETURN_PART>

<TOP-K_RETURN>
{top_k_return}
</TOP-K_RETURN>

<SUB_KEYWORD>
{sub_keyword}
</SUB_KEYWORD>

<SCORE_THRESHOLD>
{score_threshold}
</SCORE_THRESHOLD>

<OUTPUT_FORMAT>
{{
  "sub_keyword":<SUB_KEYWORD>,
  "index":[1,2,3]
  "score":...(包含所有论文的相关性分数)
}}
</OUTPUT_FORMAT>

"""
```
