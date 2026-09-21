from types import SimpleNamespace
import unittest
import torch
from release_lab.label_head import install,score


class Tiny(torch.nn.Module):
    def __init__(self,tied=False):
        super().__init__();self.input=torch.nn.Embedding(47,8);self.head=torch.nn.Linear(8,47,bias=True)
        if tied:self.head.weight=self.input.weight
        self.requires_grad_(False)
    def get_output_embeddings(self):return self.head
    def get_input_embeddings(self):return self.input
    def set_output_embeddings(self,value):self.head=value
    def forward(self,hidden,**unused):return SimpleNamespace(logits=self.head(hidden))


class LabelHeadTests(unittest.TestCase):
    def test_native_logits_probabilities_and_hidden_gradients_are_preserved(self):
        torch.manual_seed(4);model=Tiny();ids=[17,3,45,8]
        hidden=torch.randn(2,1,8,requires_grad=True)
        original=model(hidden).logits[:,-1,ids]
        expected_p=original.softmax(-1).detach()
        original.square().sum().backward();expected_gradient=hidden.grad.clone();hidden.grad=None
        receipt=install(model,ids)
        actual=score(model,{'hidden':hidden},torch.tensor(ids),torch.ones(2,4,dtype=torch.bool))
        actual.square().sum().backward()
        torch.testing.assert_close(actual.softmax(-1),expected_p,rtol=1e-6,atol=1e-7)
        torch.testing.assert_close(hidden.grad,expected_gradient,rtol=1e-6,atol=1e-7)
        self.assertGreater(receipt['projection_bytes_removed'],0)
        self.assertFalse(any(p.requires_grad for p in model.head.parameters()))

    def test_unsafe_reuse_ties_and_ordering_are_rejected(self):
        with self.assertRaises(ValueError):install(Tiny(tied=True),[1,2])
        with self.assertRaises(ValueError):install(Tiny(),[1,1])
        with self.assertRaises(ValueError):install(Tiny().half(),[1,2])
        model=Tiny();install(model,[3,7])
        with self.assertRaises(ValueError):install(model,[3,7])
        with self.assertRaises(ValueError):score(model,{'hidden':torch.zeros(1,1,8)},torch.tensor([7,3]),torch.ones(1,2,dtype=torch.bool))


if __name__=='__main__':unittest.main()
