import os
import json
import numpy as np
import pandas as pd
from collections import Counter
from scipy.spatial.distance import jensenshannon
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any

def normalize_responses(options: dict, responses: list) -> list:
    # 양수 옵션만 사용해서 0~1 정규화. None 값이나 숫자가 아닌 값은 건너뜀
    pos_keys = [int(k) for k in options.keys() if int(k) > 0]
    if not pos_keys:
        return []
    min_val, max_val = min(pos_keys), max(pos_keys)

    # min == max일 경우 0.0 처리
    if min_val == max_val:
        normalized = []
        for r in responses:
            if r is None:
                continue
            try:
                r_int = int(r)
                if r_int == min_val:
                    normalized.append(0.0)
            except ValueError:
                continue
        return normalized

    normalized = []
    for r in responses:
        if r is None:
            continue
        try:
            r_int = int(r)
            if r_int < min_val or r_int > max_val:
                continue
            norm = (r_int - min_val) / (max_val - min_val)
            normalized.append(norm)
        except ValueError:
            continue

    return normalized

def add_normalized_to_single_file(data: dict) -> dict:
    for category, items in data.get("WVS", {}).items():
        for item in items:
            options = item.get("Options", {})
            responses = item.get("Responses_preprocessed")
            if responses is None:
                responses = item.get("Responses", [])
            item["Responses_normalized"] = normalize_responses(options, responses)
    return data

def calculate_sim(data: dict, out_csv: str) -> List[Dict[str, Any]]:
    results = []

    # 상위 폴더 보장
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    for category, qlist in data.get("WVS", {}).items():
        for item in qlist:
            qid = item.get("Q_index")
            if not qid:
                continue

            # --- 사람 분포 ---
            human_norm = item.get("KR_normalized")
            if not human_norm:  # None or []
                continue
            try:
                human_counts = {float(e["norm"]): float(e["count"]) for e in human_norm}
            except (KeyError, TypeError, ValueError):
                continue
            human_total = sum(human_counts.values())
            if human_total <= 0:
                continue
            human_probs = {k: v / human_total for k, v in human_counts.items()}

            # --- 모델 분포 ---
            model_samples = item.get("Responses_normalized", [])
            if not model_samples:
                continue
            try:
                model_counts = Counter(float(v) for v in model_samples)
            except (TypeError, ValueError):
                continue
            model_total = sum(model_counts.values())
            if model_total <= 0:
                continue
            model_probs = {k: v / model_total for k, v in model_counts.items()}

            # --- 공통 bins ---
            bins = sorted(set(human_probs.keys()) | set(model_probs.keys()))
            if not bins:
                continue
            human_vec = np.array([human_probs.get(b, 0.0) for b in bins], dtype=float)
            model_vec = np.array([model_probs.get(b, 0.0) for b in bins], dtype=float)

            if human_vec.sum() == 0 or model_vec.sum() == 0:
                continue

            # --- 유사도 ---
            jsd = float(jensenshannon(human_vec, model_vec, base=2) ** 2)  # divergence
            one_minus_jsd = 1.0 - jsd
            cos = float(cosine_similarity(human_vec.reshape(1, -1),
                                          model_vec.reshape(1, -1))[0, 0])

            results.append({
                "Q_index": qid,
                "category": category,
                "jsd": jsd,
                "1-jsd": one_minus_jsd,
                "cosine": cos
            })

    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"✅ Saved similarity results to {out_csv}")
    return results

def calculate_sim_mean(results: List[Dict[str, Any]], out_csv: str) -> None:
    df = pd.DataFrame(results)
    if df.empty:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"⚠️ No rows. Wrote empty summary to {out_csv}")
        return

    # model 컬럼이 있으면 포함, 없으면 category만
    group_cols = ["category"] + (["model"] if "model" in df.columns else [])
    summary = (
        df.groupby(group_cols, as_index=False)
          .agg({"jsd": "mean", "1-jsd": "mean", "cosine": "mean"})
    )

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    summary.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"✅ Saved summary file: {out_csv}")

def run(data: dict, out_dir: str, mode: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    data_norm = add_normalized_to_single_file(data)
    full_csv = os.path.join(out_dir, f"kovalplus_similarity_score_full_{mode}.csv")
    mean_csv = os.path.join(out_dir, f"kovalplus_similarity_score_{mode}.csv")
    results = calculate_sim(data_norm, full_csv)
    calculate_sim_mean(results, mean_csv)
