from abc import abstractmethod

from torch import cat, full, gather, logsumexp, long, randn, tensor, Tensor, unsqueeze, zeros
from torch.nn import Module, Parameter

from utils.data.tags import BIOTag


CRF_CONSTRAINT_VALUE: float = -1e4


class BaseCRF(Module):
    def __init__(self, tag_vocabulary: dict[str, int]):
        super().__init__()

        self.tags_to_indices: dict[str, int] = tag_vocabulary

        # We initialize and prepare the matrix of alpha values or viterbi variables for calls to the CRF.
        # We make this a parameter so that it moves with the model, but we do not update its weights,
        #   as it is fixed.
        self.initial_algorithm_state: Parameter = Parameter(
            full((1, len(self.tags_to_indices)), CRF_CONSTRAINT_VALUE), requires_grad=False
        )   # (1 x T)

        # We initialize START_TAG such that it has all the score.
        self.initial_algorithm_state[0][self.tags_to_indices[BIOTag.START.value]] = 0.0

        # We also prepare the initial tag as a tensor for the model's use in score_chunk.
        self.initial_tag_tensor: Parameter = Parameter(
            tensor([self.tags_to_indices[BIOTag.START.value]], dtype=long), requires_grad=False
        )

        # We also prepare a tensor which holds the initial state for scoring.
        self.initial_score: Parameter = Parameter(zeros(1), requires_grad=False)

        # We create a matrix of transition parameters. Entry i, j is the score of transitioning *to* i *from* j.
        self.transitions: Parameter = Parameter(randn(len(self.tags_to_indices), len(self.tags_to_indices)))

        # These two statements enforce the constraint that we never transfer to the start tag and from the stop tag.
        self.transitions.data[self.tags_to_indices[BIOTag.START.value], :] = CRF_CONSTRAINT_VALUE
        self.transitions.data[:, self.tags_to_indices[BIOTag.STOP.value]] = CRF_CONSTRAINT_VALUE

    def get_initial_algorithm_state(self):
        return self.initial_algorithm_state

    def get_initial_score(self):
        return self.initial_score

    def get_initial_tag_tensor(self):
        return self.initial_tag_tensor

    @staticmethod
    def max_log_sum_exp(current_tensor: Tensor) -> Tensor:
        max_scores, _ = current_tensor.max(-1)   # Should produce a 1-by-n tensor of max scores.
        # We unsqueeze the max_scores value in order to properly broadcast it across the input tensor.
        return max_scores + logsumexp(current_tensor - unsqueeze(max_scores, -1), dim=-1)

    @abstractmethod
    def compute_forward(self, chunk_features: Tensor) -> Tensor:
        return NotImplemented

    @abstractmethod
    def viterbi_decode(self, chunk_features: Tensor) -> tuple[Tensor, list[int]]:
        return NotImplemented

    @abstractmethod
    def score_chunk(self, features: Tensor, tags: Tensor) -> Tensor:
        return NotImplemented


class CRF(BaseCRF):
    def __init__(self, tag_vocabulary: dict[str, int]):
        super(CRF, self).__init__(tag_vocabulary)

    def compute_forward(self, chunk_features: Tensor) -> Tensor:
        alphas = self.get_initial_algorithm_state()

        # Iterate through the chunk. The input vectors are each of shape T.
        # Since dependencies are across time-steps, this must be done sequentially.
        for word_feature in chunk_features:
            # We treat the word feature's values as emission scores.
            # We use unsqueeze in order to broadcast to each column in the transition matrix accordingly.
            emission_scores: Tensor = word_feature.unsqueeze(-1)   # (T x 1)

            # The ith entry of next_tag_var is the value for the edge (i -> next_tag) before we do log-sum-exp.
            next_tag_variables: Tensor = alphas + self.transitions + emission_scores   # (1 x T) + (T x T) + (T x 1)

            alphas = self.max_log_sum_exp(next_tag_variables).view(1, -1)

        terminal_variable = alphas + self.transitions[self.tags_to_indices[BIOTag.STOP.value]]   # (1 x T) + (1 x T)
        alpha: Tensor = self.max_log_sum_exp(terminal_variable)
        return alpha

    def viterbi_decode(self, chunk_features: Tensor) -> tuple[Tensor, list[int]]:
        back_pointers: list[list[int]] = []

        # Initialize the viterbi variables in log space.
        viterbi_variables: Tensor = self.get_initial_algorithm_state()

        for word_feature in chunk_features:
            next_tag_variables: Tensor = viterbi_variables + self.transitions  # (1 x T) + (T x T) = (T x T)
            best_tag_ids: Tensor = next_tag_variables.argmax(-1)  # (T)

            # Now add in the emission scores, and assign forward_var to the set of viterbi variables we just computed.
            best_tag_viterbi_variables: Tensor = \
                gather(next_tag_variables, -1, best_tag_ids.unsqueeze(-1)).transpose(0, 1)
            viterbi_variables = (best_tag_viterbi_variables + word_feature).view(1, -1)
            back_pointers.append(best_tag_ids.tolist())

        # We transition to STOP_TAG.
        terminal_variable: Tensor = viterbi_variables + self.transitions[self.tags_to_indices[BIOTag.STOP.value]]
        best_tag_id: int = terminal_variable.argmax().item()
        path_score: Tensor = terminal_variable[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path: list[int] = [best_tag_id]
        for traced_back_pointer in reversed(back_pointers):
            best_tag_id = traced_back_pointer[best_tag_id]
            best_path.append(best_tag_id)

        # We pop off the start tag so that it isn't returned.
        best_path.pop()
        # We reverse the path to make it in order of the appropriate tags.
        best_path.reverse()
        return path_score, best_path

    def score_chunk(self, features: Tensor, tags: Tensor) -> Tensor:
        score: Tensor = self.get_initial_score()
        initial_tag_tensor = self.get_initial_tag_tensor()
        tags = cat([initial_tag_tensor, tags])
        for i, feature in enumerate(features):
            score = score + self.transitions[tags[i + 1], tags[i]] + feature[tags[i + 1]]
        score = score + self.transitions[self.tags_to_indices[BIOTag.STOP.value], tags[-1]]
        return score
