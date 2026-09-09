"""Use an actual tiny HF/PEFT model to check generation distribution, not just token replay."""
import unittest
from unittest.mock import patch
import torch
from training_pipeline.hf_exact_token_sampler import ExactTokenGenerationConfig, HFExactTokenPolicySampler
from training_pipeline.audit_hf_exact_token_sampler import _build_model, TinyTokenizer


class RawSamplingDistributionTests(unittest.TestCase):
    def test_truncated_or_tempered_sampling_is_rejected(self):
        for kwargs in ({'temperature': .7}, {'top_p': .9}, {'do_sample': False}):
            with self.assertRaisesRegex(ValueError, 'raw_softmax'):
                ExactTokenGenerationConfig(**kwargs).validate()

    def test_actual_hf_scores_equal_raw_logits_despite_inherited_processors(self):
        model = _build_model()
        model.generation_config.top_k = 1
        model.generation_config.top_p = .1
        model.generation_config.repetition_penalty = 2
        model.generation_config.suppress_tokens = [0, 3]
        sampler = HFExactTokenPolicySampler(model, TinyTokenizer(), config=ExactTokenGenerationConfig(max_new_tokens=2), device='cpu')
        original_generate = model.generate
        observed = []
        def capture(**kwargs):
            with torch.no_grad():
                raw = model(input_ids=kwargs['input_ids'], attention_mask=kwargs['attention_mask']).logits[:, -1, :]
            generated = original_generate(**kwargs, return_dict_in_generate=True, output_scores=True)
            observed.append((raw, generated.scores[0]))
            return generated.sequences
        with patch.object(model, 'generate', side_effect=capture):
            _, info = sampler.generate('distribution check', adapter_name='lora_rca', sample_index=0, group_id='synthetic-test')
        self.assertEqual(len(observed), 1)
        self.assertTrue(torch.allclose(observed[0][0], observed[0][1], atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.isfinite(observed[0][1]).all())
        self.assertEqual(info['sampling_contract'], 'raw_softmax_v1')


if __name__ == '__main__':
    unittest.main()
