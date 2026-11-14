"""`ego_pod.compiler.pipeline` 的 Conversation_Corpus JSONL 往返属性测试。

本文件实现 design.md「Correctness Properties」中的 **Property 11**，对应
tasks.md 任务 6.17：

* **Property 11: corpus_jsonl 往返一致** ——
  把 :class:`~ego_pod.compiler.classify.SampleCandidate` 列表写入 JSONL 再
  逐行解析回来，得到的对象与原列表语义等价。

**Validates: Requirements 6.4, 6.5**

往返链路有三个层级，本文件逐层覆盖：

* **单行层（R6-AC4 单行内不含未转义换行）** ——
  :func:`serialize_sample` → :func:`parse_corpus_line` 还原出语义相等的样本；
  且序列化结果本身不含字面 ``\\n``（嵌入字段中的换行被 ``json`` 转义为
  ``\\n`` 转义序列），保证整文件能以 LF 安全切分。
* **整文件层（R6-AC4 行间分隔符 LF）** ——
  :func:`render_corpus` → :func:`parse_corpus` 对一个样本列表对称读写。
* **字节/编码层（R6-AC5 UTF-8 无 BOM）** ——
  :func:`write_corpus` 写出的文件以 UTF-8 编码、不含 BOM、行间分隔符为 LF，
  且读回后解析所得样本列表与原列表相等。

"语义等价"的定义（与 :func:`parse_sample_dict` 实现一致）：``instruction`` /
``input`` / ``output`` / ``category`` 四个字段逐一相等；``augmented`` 标记不写入
JSON，而是由 ``instruction`` 是否以 ``[AUG]`` 前缀开头无损恢复（R7-AC5、
Property 10）。因此生成器令每条样本满足不变量
``augmented == is_augmented_instruction(instruction)``，使 ``SampleCandidate``
的数据类相等比较（含 ``augmented`` 字段）在往返后严格成立。

生成策略刻意混入非 ASCII 文本（中日文、带变音符的拉丁字母、希腊字母、emoji）
与 JSON 敏感/特殊字符（双引号、反斜杠、花括号、方括号、冒号、逗号、各类换行
与控制字符、行/段分隔符 U+2028/U+2029），以充分覆盖序列化转义与 UTF-8 编码的
输入空间。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import hypothesis.strategies as st
from hypothesis import HealthCheck, given, settings

from ego_pod.compiler.augment import AUGMENT_PREFIX, is_augmented_instruction
from ego_pod.compiler.classify import VALID_CATEGORIES, SampleCandidate
from ego_pod.compiler.pipeline import (
    parse_corpus,
    parse_corpus_line,
    render_corpus,
    serialize_sample,
    write_corpus,
)

# ---------------------------------------------------------------------------
# 文本生成策略 — 非 ASCII + JSON 敏感字符 + 特殊空白/控制字符
# ---------------------------------------------------------------------------

#: 五个合法 Data_Category 的规范（小写）形式。
_CATEGORIES = sorted(VALID_CATEGORIES)

#: JSON 敏感字符、各类换行/控制字符与非 ASCII 字符的集合，刻意制造转义压力。
_SPECIAL_CHARS = (
    '"\\/'          # 双引号、反斜杠、正斜杠（json 转义相关）
    "{}[]:,"        # JSON 结构字符（嵌入字符串后不应破坏解析）
    "\b\f\n\r\t"    # 退格/换页/换行/回车/制表（均被 json 转义为 \X 序列）
    "\x00\x1f\x7f"  # 控制字符（json 以 \uXXXX 转义）
    "\u2028\u2029"  # 行分隔符 / 段分隔符（非 U+000A，不应触发按 \n 切分）
    " 　"           # 半角空格 + 全角空格（U+3000）
    "中文字符日本語のテキスト"  # CJK
    "éàüçñ"         # 带变音符的拉丁字母
    "λΩβ"           # 希腊字母
    "🎉😀🌟"        # emoji（BMP 外码点）
)

# 策略 A：从完整 Unicode 空间采样（hypothesis 默认排除代理区 Cs，UTF-8 安全）。
_broad_text = st.text(max_size=120)

# 策略 B：从精选的特殊/非 ASCII 字符集合逐字符采样，密集触发转义路径。
_special_text = st.text(alphabet=st.sampled_from(list(_SPECIAL_CHARS)), max_size=60)

# 策略 C：广义文本与特殊文本拼接，覆盖"普通字符夹杂特殊字符"的混合场景。
_field_text = st.one_of(
    _broad_text,
    _special_text,
    st.builds(lambda a, b, c: a + b + c, _broad_text, _special_text, _broad_text),
)


@st.composite
def sample_candidates(draw: st.DrawFn) -> SampleCandidate:
    """生成单条 :class:`SampleCandidate`，字段含非 ASCII 与特殊字符。

    随机决定是否给 ``instruction`` 加上 ``[AUG]`` 前缀，从而同时覆盖原始样本
    与增强样本两类；``augmented`` 标记始终与前缀保持一致（不变量
    ``augmented == is_augmented_instruction(instruction)``），以保证含
    ``augmented`` 字段的数据类相等比较在往返后成立。
    """
    instruction = draw(_field_text)
    if draw(st.booleans()):
        instruction = AUGMENT_PREFIX + instruction
    return SampleCandidate(
        instruction=instruction,
        input_field=draw(_field_text),
        output=draw(_field_text),
        category=draw(st.sampled_from(_CATEGORIES)),
        augmented=is_augmented_instruction(instruction),
    )


# ---------------------------------------------------------------------------
# Property 11: 单行往返（serialize_sample → parse_corpus_line）
# ---------------------------------------------------------------------------


@given(sample=sample_candidates())
@settings(max_examples=400, deadline=None)
def test_serialize_parse_line_round_trip(sample: SampleCandidate) -> None:
    """单条样本经 ``serialize_sample`` → ``parse_corpus_line`` 后语义等价。

    同时断言序列化结果不含字面换行符（R6-AC4「单行内不含未转义的换行符」），
    保证整文件可按 LF 安全切分。

    **Property 11: corpus_jsonl 往返一致**

    **Validates: Requirements 6.4, 6.5**
    """
    line = serialize_sample(sample)

    # R6-AC4：单行内不含字面 LF（字段中的换行被 json 转义为 \n 转义序列）。
    assert "\n" not in line

    parsed = parse_corpus_line(line)

    # 四个数据字段逐一相等。
    assert parsed.instruction == sample.instruction
    assert parsed.input_field == sample.input_field
    assert parsed.output == sample.output
    assert parsed.category == sample.category
    # augmented 由 [AUG] 前缀恢复，与原样本一致。
    assert parsed.augmented == sample.augmented
    # 数据类整体相等（覆盖全部字段，含 augmented）。
    assert parsed == sample


# ---------------------------------------------------------------------------
# Property 11: 整文件往返（render_corpus → parse_corpus）
# ---------------------------------------------------------------------------


@given(samples=st.lists(sample_candidates(), max_size=25))
@settings(max_examples=300, deadline=None)
def test_render_parse_list_round_trip(samples: list[SampleCandidate]) -> None:
    """样本列表经 ``render_corpus`` → ``parse_corpus`` 后与原列表语义等价。

    含空列表场景：``render_corpus([]) == ""``，``parse_corpus("")`` 返回 ``[]``。

    **Property 11: corpus_jsonl 往返一致**

    **Validates: Requirements 6.4, 6.5**
    """
    text = render_corpus(samples)
    assert parse_corpus(text) == samples


# ---------------------------------------------------------------------------
# Property 11: 字节/编码层往返（write_corpus → 读回字节）
# ---------------------------------------------------------------------------


@given(samples=st.lists(sample_candidates(), max_size=20))
@settings(
    max_examples=80,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_write_corpus_utf8_no_bom_round_trip(samples: list[SampleCandidate]) -> None:
    """``write_corpus`` 写出 UTF-8 无 BOM、行间 LF 的文件，读回后解析等价。

    覆盖 R6-AC5（UTF-8 无 BOM）与 R6-AC4（行间分隔符 LF）的字节级保证：

    * 文件字节不以 UTF-8 BOM（``EF BB BF``）开头；
    * 内容可作为 UTF-8 解码，且与 ``render_corpus`` 输出逐字节一致；
    * 不含 CRLF（``\\r\\n``）——字段中的回车被 json 转义为 ``\\r`` 文本而非字面
      CR 字节，写入时 ``newline="\\n"`` 亦关闭平台换行转换；
    * 读回解析所得样本列表与原列表相等。

    **Property 11: corpus_jsonl 往返一致**

    **Validates: Requirements 6.4, 6.5**
    """
    with tempfile.TemporaryDirectory() as tmp:
        corpus_path = Path(tmp) / ".ego" / "corpus.jsonl"
        returned = write_corpus(samples, corpus_path)
        assert returned == corpus_path

        raw = corpus_path.read_bytes()
        # R6-AC5：不含 UTF-8 BOM。
        assert not raw.startswith(b"\xef\xbb\xbf")
        # R6-AC4：无 CRLF。
        assert b"\r\n" not in raw

        text = raw.decode("utf-8")
        # 写出内容与 render_corpus 输出严格一致。
        assert text == render_corpus(samples)
        # 解析回的样本列表与原列表相等。
        assert parse_corpus(text) == samples
