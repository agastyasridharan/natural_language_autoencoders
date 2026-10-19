import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import Gemma2Config, Gemma2ForCausalLM, Qwen2Config, Qwen2Model

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"EasyNLA"))
from nla.utils.cross_model import resolve_critic_depth, validate_backbone
from nla.utils.hooks import register_karvonen_hook


class SwapTests(unittest.TestCase):
    def test_independent_depth(self):
        self.assertEqual(resolve_critic_depth(20,32,32),32)
        self.assertEqual(resolve_critic_depth(20,None),21)
        with self.assertRaises(ValueError): resolve_critic_depth(20,32)
        with self.assertRaises(ValueError): resolve_critic_depth(20,31,32)

    def test_width_family_guards(self):
        cfg=SimpleNamespace(d_model=3584,nla_model_type="gemma2",reconstructor_num_layers=32)
        validate_backbone(cfg,SimpleNamespace(hidden_size=3584,model_type="gemma2",num_hidden_layers=42))
        with self.assertRaises(ValueError):
            validate_backbone(cfg,SimpleNamespace(hidden_size=3840,model_type="gemma3",num_hidden_layers=48))

    def test_truncated_qwen_extracts_block_output(self):
        torch.manual_seed(0)
        cfg=Qwen2Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=4,
                        num_attention_heads=4,num_key_value_heads=2)
        model=Qwen2Model(cfg).eval(); captured=[]
        handle=model.layers[2].register_forward_hook(lambda m,a,o:captured.append(o[0] if isinstance(o,tuple) else o))
        ids=torch.tensor([[1,2,3,4]])
        with torch.no_grad(): model(input_ids=ids,use_cache=False)
        handle.remove(); model.layers=torch.nn.ModuleList(list(model.layers)[:3]); model.config.num_hidden_layers=3
        model.norm=torch.nn.Identity()
        with torch.no_grad(): short=model(input_ids=ids,use_cache=False).last_hidden_state
        torch.testing.assert_close(short,captured[0],atol=1e-6,rtol=1e-6)

    def test_gemma_injection_uses_post_block_full_residual(self):
        torch.manual_seed(0)
        cfg=Gemma2Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=3,
            num_attention_heads=4,num_key_value_heads=2,head_dim=8,query_pre_attn_scalar=8,
            attn_logit_softcapping=50.0,final_logit_softcapping=30.0,sliding_window=32)
        model=Gemma2ForCausalLM(cfg).eval(); captured=[]
        ids=torch.tensor([[1,2,3,4,5]])
        h=model.model.layers[1].register_forward_hook(lambda m,a,o:captured.append(o[0] if isinstance(o,tuple) else o))
        with torch.no_grad(): model(input_ids=ids,use_cache=False)
        original=captured.pop().clone(); h.remove()
        v=torch.randn(1,32); vref=[v]
        register_karvonen_hook(model,vref,3,2,4,layer_idx=1)
        h=model.model.layers[1].register_forward_hook(lambda m,a,o:captured.append(o[0] if isinstance(o,tuple) else o))
        with torch.no_grad(): model(input_ids=ids,use_cache=False)
        actual=captured.pop(); expected=original.clone()
        expected[0,2]+=original[0,2].norm()*v[0]/v[0].norm()
        torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-5)
        h.remove()

    def test_patcher_dependencies_and_idempotence(self):
        spec=importlib.util.spec_from_file_location("patch",ROOT/"EasyNLA/utils/patch_vllm_lens.py")
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        old=module.HUNKS
        try:
            module.HUNKS=[("x = 1","x = 2"),("x = 2","x = 3")]
            result,count=module.patched_source("x = 1\n")
            self.assertEqual(result,"x = 3\n");self.assertEqual(count,2)
        finally: module.HUNKS=old
        # Exercise the exact installed upstream worker fixture if available.
        fixture=Path('/tmp/nla-swap-vllm-worker.py')
        if fixture.exists():
            one,count=module.patched_source(fixture.read_text())
            two,count2=module.patched_source(one)
            self.assertEqual(one,two);self.assertEqual(count2,0)


if __name__ == "__main__": unittest.main()
