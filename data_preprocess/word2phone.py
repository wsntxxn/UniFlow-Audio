import os
import json
import re
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from collections import Counter


from montreal_forced_aligner.g2p.generator import PyniniConsoleGenerator

# 初始化 G2P（可在主程序启动时调用一次）
g2p = PyniniConsoleGenerator(
    g2p_model_path='english_us_arpa',
    strict_graphemes=False,
    num_pronunciations=1,
    include_bracketed=False
)
g2p.setup()
g2p_error_cnt=0
g2p_error_list=[]

def g2p_resolve(word):
    """调用 G2P 生成发音（用于处理 OOV）"""
    try:
        result = g2p.rewriter(word.lower())
        if result and result[0][0]:
            return result[0][0].split()
    except Exception:
        return None
    return None


def text_norm(s):
    """
    文本规范化（保留 don't, it's 等词内单引号，删除作为引号/括号的单引号，并去除其它标点）：
    1. 小写化
    2. 保留字母间的撇号 (e.g. don't)
    3. 删除不在字母间的撇号（作引号或孤立时）
    4. 删除其它常见标点（.,;!?()[]-"“”等）
    5. 合并多余空格
    """
    s = s.lower()

    # 先把字母间（a'b）形式的撇号临时替换为占位，避免被后续删除
    # 支持 ASCII ' 和 Unicode ’、‘
    APOST = "<<<APOST>>>"  # 占位字符串（保证不会出现在句子里）
    s = re.sub(r"(?<=[A-Za-z0-9])['\u2019\u2018](?=[A-Za-z0-9])", APOST, s)

    # 删除所有其余的单引号（这些是引号或孤立的）
    s = re.sub(r"['\u2019\u2018]", " ", s)

    # 删除其它标点（保留我们已用占位保护的内部撇号）
    s = re.sub(r"[,\.\!\?\;\:\(\)\[\]\"“”\-]", " ", s)

    # 恢复内部撇号为 ASCII apostrophe（也可恢复为原字符）
    s = s.replace(APOST, "'")

    # 合并多余空格
    s = " ".join(s.split())

    return s


# ---------------- 载入词映射 ----------------
def load_word2phone(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------- 核心转换 ----------------
def sentence_to_phones(sentence, word2phones):
    """
    句子转 phone：
    1. 对原句做拆分，保留标点位置信息用于后面添加 sil
    2. 标点位置添加 sil
    3. 句子首尾添加 sil
    """
    original_sentence = sentence  # 保存原始句子
    sentence = text_norm(sentence)

    phone_sequence = ["sil"]  # 起始停顿
    oov_list = []

    # 拆分原始句子用于判断标点位置

    tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[.,;!?]", original_sentence)

    for token in tokens:
        if re.match(r"[.,;!?]", token):  # 标点
            phone_sequence.append("sil")
        else:
            word = text_norm(token)  # 规范化词
                
            if word not in word2phones:
                g2p_ph = g2p_resolve(word)
                if g2p_ph:
                    phone_sequence.extend(g2p_ph)
                else:
                    phone_sequence.append("spn")  # 实在无法处理，用轻微停顿
                    g2p_error_cnt+=1
                    g2p_error_list.append(word)
                    
                oov_list.append(word)

            else:
                pron, _ = max(word2phones[word].items(), key=lambda x: x[1])
                phone_sequence.extend(pron.split())

    if phone_sequence[-1] != 'sil':
        phone_sequence.append("sil")  # 结束停顿
    return phone_sequence, oov_list


def convert_sentence(sentence, word2phones):
    phones, oov_list = sentence_to_phones(sentence, word2phones)
    return " ".join(phones), oov_list


# ---------------- OOV 统计 ----------------
def process_oov_single(args):
    lab_file, word2phones = args
    try:
        with open(lab_file, "r", encoding="utf-8") as f:
            sentence_raw = f.read().strip()

        sentence_norm = text_norm(sentence_raw)
        _, oov_list = sentence_to_phones(sentence_raw, word2phones)

        return {
            "file": lab_file,
            "sentence_raw": sentence_raw,
            "sentence_norm": sentence_norm,
            "oov": oov_list
        }
    except Exception:
        return {"file": lab_file, "sentence_raw": "", "sentence_norm": "", "oov": []}


def validate_oov_parallel(root_dir, word2phones, num_workers=None):
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    lab_files = [
        os.path.join(dp, f)
        for dp, _, files in os.walk(root_dir)
        for f in files if f.endswith(".lab")
    ]

    print(f"🔍 扫描 OOV，共 {len(lab_files)} 文件，使用 {num_workers} 进程")

    oov_counter = Counter()
    oov_details = []
    args_list = [(f, word2phones) for f in lab_files]

    with Pool(num_workers) as pool:
        for result in tqdm(pool.imap_unordered(process_oov_single, args_list), total=len(args_list)):
            if result["oov"]:
                oov_counter.update(result["oov"])
                oov_details.append(result)

    print(f"👉 总 OOV 数（含重复）: {sum(oov_counter.values())}")
    print(f"👉 不重复 OOV 数: {len(oov_counter)}")

    save_oov_path = "oov_words.json"
    with open(save_oov_path, "w", encoding="utf-8") as f:
        json.dump(oov_counter, f, ensure_ascii=False, indent=4)

    save_details_path = "oov_details.json"
    with open(save_details_path, "w", encoding="utf-8") as f:
        json.dump(oov_details, f, ensure_ascii=False, indent=4)

    return oov_counter, oov_details


# ---------------- 处理 lab 文件 ----------------
def process_single_lab(args):
    lab_file, word2phones = args
    try:
        with open(lab_file, "r", encoding="utf-8") as f:
            sentence = f.read().strip()

        phone_str, _ = convert_sentence(sentence, word2phones)
        output_file = lab_file.replace(".lab", ".dict.phone")

        with open(output_file, "w", encoding="utf-8") as out_f:
            out_f.write(phone_str + "\n")
        return True
    except Exception as e:
        return f"❌ {lab_file}: {e}"


def process_lab_files_parallel(root_dir, word2phones, num_workers=None):
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    lab_files = [
        os.path.join(dp, f)
        for dp, _, files in os.walk(root_dir)
        for f in files if f.endswith(".lab")
    ]

    print(f"📊 共 {len(lab_files)} 文件，使用 {num_workers} 进程")

    failures = []
    args_list = [(f, word2phones) for f in lab_files]

    with Pool(num_workers) as pool:
        for result in tqdm(pool.imap_unordered(process_single_lab, args_list), total=len(args_list)):
            if result is not True:
                failures.append(result)

    print("✅ 全部处理完成")
    if failures:
        print(f"⚠️ {len(failures)} 文件失败（示例）:")
        for msg in failures[:5]:
            print(" ", msg)


# ========================= 主函数 =========================
if __name__ == "__main__":
    word2phones = load_word2phone("data/word2phone.json")

    # sentence = """You'll see in a moment what the difference is between 'convenient' and 'inconvenient.' You quite understand it now, don't you?" """
    sentence="""Stagecraft is the art of getting over these and other difficulties, and (if possible) getting over them in a showy manner, so that people will say, "How remarkable his stagecraft is for so young a writer," when otherwise they mightn't have noticed it at all."""
    print("sentence:", sentence)
    phone,OOV=convert_sentence(sentence, word2phones)
    print("norm sentence:", text_norm(sentence))
    print("→ Phone:", phone)
    print("→ OOV:", OOV)

    # root_dir = "/hpc_stor03/sjtu_home/jiahao.mei/data/LibriSpeech-pc/test-clean"


    # # validate_oov_parallel(root_dir, word2phones)



    # process_lab_files_parallel(root_dir, word2phones)

    # print(f'Out of Vocabulary且g2p也无法处理的词汇数量{g2p_error_cnt}, {g2p_error_list}')
