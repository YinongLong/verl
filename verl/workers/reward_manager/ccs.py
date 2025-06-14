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
import collections
import json
from collections import defaultdict
from typing import Callable, Optional

import torch
from doraemon.formatters import consistency
from doraemon.formatters.utils import extract_think_answer
from doraemon.inference import service
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register


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


@register("ccs")
class CCSRewardManager:
    """
    customize our own reward manager
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int,
        compute_score: Optional[Callable] = None,
        reward_fn_key: str = 'data_source',
        tokenizer_dir: Optional[str] = None,
        service_conf: Optional[str] = None
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.tokenizer_dir = tokenizer_dir
        self.service_conf = service_conf

        self.data_formatter = consistency.DataFormatter(tokenizer_dir)

    def request_service(self, container, n=1, temperature=0.0):
        """
        请求远程服务，判断CoT和json结果的一致性
        """
        completions = service.get_completions(container, self.service_conf, n=n, temperature=temperature)
        flags = []
        for com_item in completions:
            counter = collections.Counter()
            for choice in com_item.choices:
                if choice.finish_reason != 'stop':
                    continue
                flag = choice.text
                if flag not in {'一致', '不一致'}:
                    continue
                counter[flag] += 1
            if not counter:
                flag = '一致'
            else:
                flag = counter.most_common()[0][0]
            flags.append(flag)
        return flags

    def judge(self, data):
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
                res.append((cot, se_dict, -1))  # -1 represents error
            else:
                extra_info = data_item.non_tensor_batch.get('extra_info', None)
                assert extra_info is not None
                session = json.loads(extra_info['session'])
                try:
                    prompt = self.data_formatter.format(session, cot, se_dict)
                except Exception:
                    res.append((cot, se_dict, -1))
                    continue
                res.append((cot, se_dict, len(container)))
                container.append(prompt)
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
                print("[consistency]", c_flag)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
