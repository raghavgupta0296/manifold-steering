import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class LlamaModel:
    small_model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
    # small_model_name = "meta-llama/Llama-3.2-1B-Instruct"
    paper_model_name = "meta-llama/Llama-3.1-8B"

    def __init__(self, model_name: str = small_model_name):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

    def load(self) -> "LlamaModel":
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
        )
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        self.model.eval()
        return self

    def _get_answer_token_ids(
        self,
        answer_values: list[str],
        output_prefix: str,
    ) -> list[list[int]]:
        if self.tokenizer is None:
            raise ValueError("Call .load() before tokenizing answers.")

        answer_token_ids = []
        for answer in answer_values:
            canonical = output_prefix + answer
            candidates = [canonical]
            if canonical.startswith(" "):
                candidates.append(answer)
                candidates.append(" " + answer.lower())
            else:
                candidates.append(" " + answer)
            if answer.lower() != answer:
                candidates.append(answer.lower())

            ids_for_answer = []
            seen_strings = set()
            for candidate in candidates:
                if candidate in seen_strings:
                    continue
                seen_strings.add(candidate)
                token_ids = self.tokenizer.encode(candidate, add_special_tokens=False)
                if len(token_ids) == 1 and token_ids[0] not in ids_for_answer:
                    ids_for_answer.append(token_ids[0])

            if not ids_for_answer:
                raise ValueError(f"No single-token variants found for {answer!r}.")
            answer_token_ids.append(ids_for_answer)

        return answer_token_ids

    def _get_answer_probabilities(
        self,
        logits: torch.Tensor,
        answer_token_ids: list[list[int]],
        full_vocab_softmax: bool,
        include_other: bool,
    ) -> torch.Tensor:
        device = logits.device
        full_probs = F.softmax(logits.float(), dim=-1)

        answer_probs = torch.zeros(
            full_probs.shape[0],
            len(answer_token_ids),
            device=device,
        )
        for answer_index, ids in enumerate(answer_token_ids):
            ids_tensor = torch.tensor(ids, device=device)
            answer_probs[:, answer_index] = full_probs[:, ids_tensor].sum(dim=-1)

        if not full_vocab_softmax:
            answer_probs = answer_probs / answer_probs.sum(
                dim=-1,
                keepdim=True,
            ).clamp(min=1e-10)

        if include_other:
            other = (1.0 - answer_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0)
            answer_probs = torch.cat([answer_probs, other], dim=-1)

        return answer_probs

    def gather_activations(
        self,
        prompts: list[str],
        layers: list[int],
        token_position: int = -1,
        batch_size: int = 8,
    ) -> dict[int, torch.Tensor]:
        """
        Gather residual-stream activations at multiple layers and one token position.

        HF hidden states include embeddings at index 0, so layer 0 maps to
        hidden_states[1], layer 1 maps to hidden_states[2], and so on.
        """
        results = self.gather_activations_and_prediction_probabilities(
            prompts=prompts,
            layers=layers,
            answer_values=[],
            token_position=token_position,
            batch_size=batch_size,
        )
        return results["activations"]

    def gather_prediction_probabilities(
        self,
        prompts: list[str],
        answer_values: list[str],
        output_prefix: str = " ",
        full_vocab_softmax: bool = True,
        include_other: bool = False,
        batch_size: int = 8,
    ) -> torch.Tensor:
        """
        Gather next-token probabilities for answer concepts.

        Like the reference repo, this first softmaxes over the full vocabulary,
        then sums common token variants for each concept, e.g. " Monday",
        "Monday", and " monday".
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Call .load() before gathering prediction probabilities.")

        answer_token_ids = self._get_answer_token_ids(answer_values, output_prefix)

        device = next(self.model.parameters()).device
        probabilities = []

        with torch.no_grad():
            for start in range(0, len(prompts), batch_size):
                batch_prompts = prompts[start : start + batch_size]
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                ).to(device)

                outputs = self.model(**inputs)
                batch_probs = self._get_answer_probabilities(
                    logits=outputs.logits[:, -1, :],
                    answer_token_ids=answer_token_ids,
                    full_vocab_softmax=full_vocab_softmax,
                    include_other=include_other,
                )

                probabilities.append(batch_probs.detach().cpu())

        return torch.cat(probabilities, dim=0)

    def gather_activations_and_prediction_probabilities(
        self,
        prompts: list[str],
        layers: list[int],
        answer_values: list[str],
        token_position: int = -1,
        output_prefix: str = " ",
        full_vocab_softmax: bool = True,
        include_other: bool = False,
        batch_size: int = 8,
    ) -> dict[str, dict[int, torch.Tensor] | torch.Tensor | None]:
        """
        Gather activations and next-token answer probabilities in one forward pass.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Call .load() before gathering model outputs.")

        answer_token_ids = (
            self._get_answer_token_ids(answer_values, output_prefix)
            if answer_values
            else None
        )
        device = next(self.model.parameters()).device
        activation_batches = {layer: [] for layer in layers}
        probability_batches = []

        with torch.no_grad():
            for start in range(0, len(prompts), batch_size):
                batch_prompts = prompts[start : start + batch_size]
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                ).to(device)

                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                )

                for layer in layers:
                    batch_activations = outputs.hidden_states[layer + 1][
                        :, token_position, :
                    ]
                    activation_batches[layer].append(batch_activations.detach().cpu())

                if answer_token_ids is not None:
                    batch_probs = self._get_answer_probabilities(
                        logits=outputs.logits[:, -1, :],
                        answer_token_ids=answer_token_ids,
                        full_vocab_softmax=full_vocab_softmax,
                        include_other=include_other,
                    )
                    probability_batches.append(batch_probs.detach().cpu())

        activations = {
            layer: torch.cat(batches, dim=0)
            for layer, batches in activation_batches.items()
        }
        probabilities = (
            torch.cat(probability_batches, dim=0) if probability_batches else None
        )

        return {
            "activations": activations,
            "probabilities": probabilities,
        }

    def gather_steered_prediction_probabilities(
        self,
        prompts: list[str],
        answer_values: list[str],
        layer: int,
        steering_vectors: torch.Tensor,
        coefficient: float,
        token_position: int = -1,
        output_prefix: str = " ",
        full_vocab_softmax: bool = True,
        include_other: bool = False,
        batch_size: int = 8,
    ) -> torch.Tensor:
        """
        Add steering vectors to one residual-stream layer and score answers.

        The hook is placed on decoder layer `layer`, matching the hidden-state
        convention used elsewhere: layer 28 means hidden_states[29].
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Call .load() before gathering model outputs.")

        if steering_vectors.ndim == 1:
            steering_vectors = steering_vectors.unsqueeze(0).expand(len(prompts), -1)
        if steering_vectors.shape[0] != len(prompts):
            raise ValueError("Expected one steering vector per prompt.")

        answer_token_ids = self._get_answer_token_ids(answer_values, output_prefix)
        device = next(self.model.parameters()).device
        module = self.model.model.layers[layer]
        probabilities = []

        with torch.no_grad():
            for start in range(0, len(prompts), batch_size):
                batch_prompts = prompts[start : start + batch_size]
                batch_vectors = steering_vectors[start : start + batch_size].to(device)

                def add_steering(_module, _inputs, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    steered = hidden.clone()
                    addition = batch_vectors.to(dtype=hidden.dtype) * coefficient
                    steered[:, token_position, :] = (
                        steered[:, token_position, :] + addition
                    )
                    if isinstance(output, tuple):
                        return (steered, *output[1:])
                    return steered

                handle = module.register_forward_hook(add_steering)
                try:
                    inputs = self.tokenizer(
                        batch_prompts,
                        return_tensors="pt",
                        padding=True,
                    ).to(device)
                    outputs = self.model(**inputs)
                finally:
                    handle.remove()

                batch_probs = self._get_answer_probabilities(
                    logits=outputs.logits[:, -1, :],
                    answer_token_ids=answer_token_ids,
                    full_vocab_softmax=full_vocab_softmax,
                    include_other=include_other,
                )
                probabilities.append(batch_probs.detach().cpu())

        return torch.cat(probabilities, dim=0)

    def gather_pca_feature_steered_prediction_probabilities(
        self,
        prompts: list[str],
        answer_values: list[str],
        layer: int,
        pca_mean: torch.Tensor,
        pca_rotation: torch.Tensor,
        feature_steering_vectors: torch.Tensor,
        coefficient: float,
        token_position: int = -1,
        output_prefix: str = " ",
        full_vocab_softmax: bool = True,
        include_other: bool = False,
        batch_size: int = 8,
    ) -> torch.Tensor:
        """
        Add steering vectors in PCA feature space and reconstruct activations.

        This mirrors the reference repo's featurizer idea in a small form:
        activation -> PCA features + residual error -> add feature vector ->
        inverse PCA while preserving the residual error.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Call .load() before gathering model outputs.")

        if feature_steering_vectors.ndim == 1:
            feature_steering_vectors = feature_steering_vectors.unsqueeze(0).expand(
                len(prompts),
                -1,
            )
        if feature_steering_vectors.shape[0] != len(prompts):
            raise ValueError("Expected one feature steering vector per prompt.")

        answer_token_ids = self._get_answer_token_ids(answer_values, output_prefix)
        device = next(self.model.parameters()).device
        module = self.model.model.layers[layer]
        pca_mean = pca_mean.to(device)
        pca_rotation = pca_rotation.to(device)
        probabilities = []

        with torch.no_grad():
            for start in range(0, len(prompts), batch_size):
                batch_prompts = prompts[start : start + batch_size]
                batch_vectors = feature_steering_vectors[
                    start : start + batch_size
                ].to(device)

                def add_feature_steering(_module, _inputs, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    steered = hidden.clone()
                    position_activations = steered[:, token_position, :]
                    mean = pca_mean.to(dtype=hidden.dtype)
                    rotation = pca_rotation.to(dtype=hidden.dtype)
                    feature_delta = batch_vectors.to(dtype=hidden.dtype) * coefficient

                    centered = position_activations - mean
                    base_features = centered @ rotation
                    base_error = centered - base_features @ rotation.T
                    steered_features = base_features + feature_delta
                    steered[:, token_position, :] = (
                        mean + steered_features @ rotation.T + base_error
                    )

                    if isinstance(output, tuple):
                        return (steered, *output[1:])
                    return steered

                handle = module.register_forward_hook(add_feature_steering)
                try:
                    inputs = self.tokenizer(
                        batch_prompts,
                        return_tensors="pt",
                        padding=True,
                    ).to(device)
                    outputs = self.model(**inputs)
                finally:
                    handle.remove()

                batch_probs = self._get_answer_probabilities(
                    logits=outputs.logits[:, -1, :],
                    answer_token_ids=answer_token_ids,
                    full_vocab_softmax=full_vocab_softmax,
                    include_other=include_other,
                )
                probabilities.append(batch_probs.detach().cpu())

        return torch.cat(probabilities, dim=0)

    def gather_replaced_prediction_probabilities(
        self,
        prompts: list[str],
        answer_values: list[str],
        layer: int,
        replacement_activations: torch.Tensor,
        token_position: int = -1,
        output_prefix: str = " ",
        full_vocab_softmax: bool = True,
        include_other: bool = False,
        batch_size: int = 8,
    ) -> torch.Tensor:
        """
        Replace one residual-stream position with precomputed activations.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Call .load() before gathering model outputs.")

        if replacement_activations.ndim == 1:
            replacement_activations = replacement_activations.unsqueeze(0).expand(
                len(prompts),
                -1,
            )
        if replacement_activations.shape[0] != len(prompts):
            raise ValueError("Expected one replacement activation per prompt.")

        answer_token_ids = self._get_answer_token_ids(answer_values, output_prefix)
        device = next(self.model.parameters()).device
        module = self.model.model.layers[layer]
        probabilities = []

        with torch.no_grad():
            for start in range(0, len(prompts), batch_size):
                batch_prompts = prompts[start : start + batch_size]
                batch_replacements = replacement_activations[
                    start : start + batch_size
                ].to(device)

                def replace_activation(_module, _inputs, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    replaced = hidden.clone()
                    replaced[:, token_position, :] = batch_replacements.to(
                        dtype=hidden.dtype,
                    )
                    if isinstance(output, tuple):
                        return (replaced, *output[1:])
                    return replaced

                handle = module.register_forward_hook(replace_activation)
                try:
                    inputs = self.tokenizer(
                        batch_prompts,
                        return_tensors="pt",
                        padding=True,
                    ).to(device)
                    outputs = self.model(**inputs)
                finally:
                    handle.remove()

                batch_probs = self._get_answer_probabilities(
                    logits=outputs.logits[:, -1, :],
                    answer_token_ids=answer_token_ids,
                    full_vocab_softmax=full_vocab_softmax,
                    include_other=include_other,
                )
                probabilities.append(batch_probs.detach().cpu())

        return torch.cat(probabilities, dim=0)

    def gather_pca_feature_replaced_prediction_probabilities(
        self,
        prompts: list[str],
        answer_values: list[str],
        layer: int,
        pca_mean: torch.Tensor,
        pca_rotation: torch.Tensor,
        replacement_features: torch.Tensor,
        token_position: int = -1,
        output_prefix: str = " ",
        full_vocab_softmax: bool = True,
        include_other: bool = False,
        batch_size: int = 8,
    ) -> torch.Tensor:
        """
        Replace PCA features and reconstruct, preserving PCA residual error.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Call .load() before gathering model outputs.")

        if replacement_features.ndim == 1:
            replacement_features = replacement_features.unsqueeze(0).expand(
                len(prompts),
                -1,
            )
        if replacement_features.shape[0] != len(prompts):
            raise ValueError("Expected one replacement feature vector per prompt.")

        answer_token_ids = self._get_answer_token_ids(answer_values, output_prefix)
        device = next(self.model.parameters()).device
        module = self.model.model.layers[layer]
        pca_mean = pca_mean.to(device)
        pca_rotation = pca_rotation.to(device)
        probabilities = []

        with torch.no_grad():
            for start in range(0, len(prompts), batch_size):
                batch_prompts = prompts[start : start + batch_size]
                batch_replacements = replacement_features[
                    start : start + batch_size
                ].to(device)

                def replace_features(_module, _inputs, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    replaced = hidden.clone()
                    position_activations = replaced[:, token_position, :]
                    mean = pca_mean.to(dtype=hidden.dtype)
                    rotation = pca_rotation.to(dtype=hidden.dtype)
                    target_features = batch_replacements.to(dtype=hidden.dtype)

                    centered = position_activations - mean
                    base_features = centered @ rotation
                    base_error = centered - base_features @ rotation.T
                    replaced[:, token_position, :] = (
                        mean + target_features @ rotation.T + base_error
                    )

                    if isinstance(output, tuple):
                        return (replaced, *output[1:])
                    return replaced

                handle = module.register_forward_hook(replace_features)
                try:
                    inputs = self.tokenizer(
                        batch_prompts,
                        return_tensors="pt",
                        padding=True,
                    ).to(device)
                    outputs = self.model(**inputs)
                finally:
                    handle.remove()

                batch_probs = self._get_answer_probabilities(
                    logits=outputs.logits[:, -1, :],
                    answer_token_ids=answer_token_ids,
                    full_vocab_softmax=full_vocab_softmax,
                    include_other=include_other,
                )
                probabilities.append(batch_probs.detach().cpu())

        return torch.cat(probabilities, dim=0)


if __name__ == "__main__":
    llama = LlamaModel().load()
    print(f"Loaded model: {llama.model_name}")

    import ipdb

    ipdb.set_trace()
