# Copyright 2023-2024 SGLang Team
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
# ==============================================================================
"""The baseclass of a backend for reasoner grammar-guided constrained decoding."""

import logging
from typing import List, Optional, Tuple

import torch

from .base_grammar_backend import (
    INVALID_GRAMMAR_OBJ,
    BaseGrammarBackend,
    BaseGrammarObject,
)

logger = logging.getLogger(__name__)


class ReasonerGrammarObject(BaseGrammarObject):
    def __init__(self, grammar: BaseGrammarObject, think_end_id):
        super().__init__()
        self.grammar = grammar
        self.think_end_id = think_end_id
        self.is_in_reasoning = False  # Disable enforce reasoning
        self.is_think_started = False
        self.think_token = 0

    def accept_token(self, token: int):
        if self.is_in_reasoning or self.is_think_started:
            self.think_token += 1

        """
        special_end_tokens = set([163585, 163586, 163591, 163593, 163596, 163599])
        if (self.is_in_reasoning or self.is_think_started) and (
            token in special_end_tokens
        ):
            logging.error("DAMN SGLANG accept special end token {token} during think")
        """

        if token == 163606:
            # logging.error("DAMN SGLANG accept think start token")
            self.is_think_started = True
            self.is_in_reasoning = True
            self.think_token = 0
            return

        if token == self.think_end_id:
            # logging.error(f"DAMN SGLANG accept think end token {self.think_token}")
            self.is_think_started = False
            self.is_in_reasoning = False
            return

        if not self.is_in_reasoning and token != self.think_end_id:
            self.grammar.accept_token(token)

    def allocate_vocab_mask(
        self, vocab_size: int, batch_size: int, device
    ) -> torch.Tensor:
        """
        if self.is_think_started:
            ret = torch.zeros(batch_size, vocab_size, dtype=torch.bool, device=device)
            logging.error(f"FUCK SGLANG allocate_vocab_mask in ReasonerGrammarObject {ret.dtype}")
            return ret
        """
        return self.grammar.allocate_vocab_mask(vocab_size, batch_size, device)

    def set_token_allowed(
        self,
        bitmask: torch.Tensor,
        token_id: int,
        batch_idx: int = 0,
        allowed: bool = True,
    ):
        element_idx = token_id // 32
        bit_idx = token_id % 32
        current_value = bitmask[batch_idx, element_idx].item()

        if allowed:
            new_value = current_value | (1 << bit_idx)
        else:
            new_value = current_value & ~(1 << bit_idx)
        bitmask[batch_idx, element_idx] = new_value

    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
        vocab_mask[idx].fill_(-1)
        if self.is_think_started and (self.think_token < 2048):
            # logging.error(f"DAMN SGLANG think length {self.think_token}")
            token_ids = [
                163606,
                163595,
                163597,
                163598,
                163599,
                163596,
            ]
            for token_id in token_ids:
                self.set_token_allowed(vocab_mask, token_id, idx, False)
            special_end_tokens = [163585, 163586, 163591, 163593, 163596, 163599]
            for token_id in special_end_tokens:
                self.set_token_allowed(vocab_mask, token_id, idx, False)
            return

        if (self.is_in_reasoning or self.is_think_started) and (
            self.think_token >= 2048
        ):
            vocab_mask[idx].fill_(0)
            # logging.error(f"DAMN SGLANG think too long {self.think_token}")
            token_ids = [
                163607,
            ]
            for token_id in token_ids:
                self.set_token_allowed(vocab_mask, token_id, idx, True)
            return

        if not self.is_in_reasoning:
            self.grammar.fill_vocab_mask(vocab_mask, idx)

    def move_vocab_mask(self, vocab_mask: torch.Tensor, device) -> torch.Tensor:
        return self.grammar.move_vocab_mask(vocab_mask, device)

    @property
    def apply_vocab_mask(self):
        return self.grammar.apply_vocab_mask

    def copy(self) -> BaseGrammarObject:
        return ReasonerGrammarObject(self.grammar.copy(), self.think_end_id)

    @property
    def finished(self):
        return self.grammar.finished

    @finished.setter
    def finished(self, finished):
        self.grammar.finished = finished

    def try_jump_forward(self, tokenizer):
        return self.grammar.try_jump_forward(tokenizer)

    def jump_forward_str_state(self, helper):
        return self.grammar.jump_forward_str_state(helper)

    def jump_and_retokenize(
        self, old_output_ids: List[int], new_output_ids: List[int], next_state: int
    ):
        return self.grammar.jump_and_retokenize(
            old_output_ids, new_output_ids, next_state
        )


class ReasonerGrammarBackend(BaseGrammarBackend):
    def __init__(self, grammar_backend: BaseGrammarBackend, think_end_id):
        super().__init__()
        self.grammar_backend = grammar_backend
        self.think_end_id = think_end_id

    def _init_value_dispatch(self, key: Tuple[str, str]) -> Optional[BaseGrammarObject]:
        ret = self.grammar_backend._init_value_dispatch(key)
        # avoid wrapping invalid grammar, so that the scheduler can detect it
        if ret is None or ret is INVALID_GRAMMAR_OBJ:
            return ret
        return ReasonerGrammarObject(ret, self.think_end_id)
