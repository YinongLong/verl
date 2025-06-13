# Copyright 2025 TCL-Research team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import re
import requests
from collections import defaultdict
from concurrent import futures
from requests.exceptions import Timeout, RequestException
from typing import Callable, Optional

import torch
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.reward_score import default_compute_score


def request_remote_service(service_url, data_items, timeout=5.0):
    """
    请求部署的一致性判别模型，判断采样的思维链和json结果之间的关系
    """
    ans = []
    for i, session, cot, se_dict in data_items:
        try:
            response = requests.post(
                url=service_url,
                json={
                    'session': json.dumps(session, ensure_ascii=False),
                    'cot': cot,
                    'se_dict_str': json.dumps(se_dict, ensure_ascii=False)
                },
                timeout=timeout
            )
            re_dict = response.json()
            ans.append((i, re_dict['cons_flag']))
        except (Timeout, RequestException):
            ans.append((i, '一致'))
    return ans


def concurrent_requests(container, service_url, num_threads):
    """
    并发请求远程服务
    """
    num_items = len(container)
    num_base = num_items // num_threads
    num_resi = num_items % num_threads

    def get_bs():
        nonlocal num_resi
        bs = num_base
        if num_resi:
            bs += 1
            num_resi -= 1
        return bs

    ans = []
    with futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures_arr = []

        capacity = []
        bs = get_bs()
        for i, (session, cot, se_dict) in enumerate(container):
            capacity.append((i, session, cot, se_dict))
            if len(capacity) < bs:
                continue
            future = executor.submit(
                request_remote_service, service_url, capacity
            )
            futures_arr.append(future)
            capacity = []
            bs = get_bs()
        if capacity:
            future = executor.submit(
                request_remote_service, service_url, capacity
            )
            futures_arr.append(future)
            capacity = []

        for future in futures.as_completed(futures_arr):
            part_ans = future.result()
            ans.extend(part_ans)
    ans.sort(key=lambda item: item[0])
    ans = [item[-1] for item in ans]
    return ans


def extract_think_answer(solution_str):
    """
    按照预期抽取模型回复中的推理部分和答案结果部分
    """
    pattern = r'^\s*<analysis>(.+?)</analysis>\s*<answer>(.+?)</answer>\s*$'
    se_ans = re.search(pattern, solution_str, re.DOTALL)
    if se_ans is None:
        return None, None

    think_part = se_ans.group(1)
    answer_part = se_ans.group(2)
    if re.findall(r'<analysis>|</analysis>|<answer>|</answer>', think_part):
        return None, None

    if re.findall(r'<analysis>|</analysis>|<answer>|</answer>', answer_part):
        return None, None

    think_part = think_part.strip()
    answer_part = answer_part.strip()
    if not think_part or not answer_part:
        return None, None

    return think_part, answer_part


def get_se_dict(answer_str):
    """
    检测答案是否为合法的json结构
    """
    if answer_str is None:
        return None

    try:
        answer_json = json.loads(answer_str)
    except Exception:
        return None

    if not isinstance(answer_json, dict) or not answer_json:
        return None

    return answer_json


class CCSRewardManager:
    """
    针对中控强化学习的奖励定制化处理
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int,
        compute_score: Optional[Callable] = None,
        reward_fn_key: str = 'data_source',
        service_url: str = 'http://172.16.0.2:60025/consistency/qwen3_4b',
        num_threads: int = 128
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.service_url = service_url
        self.num_threads = num_threads

    def request_service(self, container):
        """
        请求远程服务，判断CoT和json结果的一致性
        """
        return concurrent_requests(container, self.service_url, self.num_threads)

    def judge(self, data):
        """
        judge the consistency between CoT and answer
        """
        res = []
        container = []
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            cot, se_dict_str = extract_think_answer(response_str)
            se_dict = get_se_dict(se_dict_str)
            if cot is None or se_dict is None:
                res.append((cot, se_dict, -1))  # -1 代表推理结果格式错误，不需要判断一致性
            else:
                res.append((cot, se_dict, len(container)))
                extra_info = data_item.non_tensor_batch.get('extra_info', None)
                assert extra_info is not None
                session = json.loads(extra_info['session'])
                container.append((session, cot, se_dict))
        flags = self.request_service(container)
        return res, flags

    def __call__(self, data: DataProto, return_dict: bool = False):
        """
        We will expand this function gradually based on the available datasets
        """

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            if return_dict:
                return {'reward_tensor': data.batch['rm_scores']}
            else:
                return data.batch['rm_scores']

        # 请求远程服务，判断模型推理思维链和json结果的一致性关系
        ext_res, consist_flags = self.judge(data)
        
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            cot, prd_se_dict, c_flag_idx = ext_res[i]
            c_flag = '未知' if c_flag_idx == -1 else consist_flags[c_flag_idx]

            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            score = self.compute_score(
                data_source=data_source,
                cot=cot,
                prd_se_dict=prd_se_dict,
                cons_flag=c_flag,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
